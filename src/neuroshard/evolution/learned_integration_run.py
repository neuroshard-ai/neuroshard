"""CPU stage-1 learned-integration training and development scoring.

Attaches a last-layer top-1 mixture or the matched no-expansion MLP to a local
135M parent, trains 128 teacher-forced steps, and scores generated development,
code-retention, and eight general parent responses. Confirmation stays closed.
This is not a GPU launch and not a 0.4.0 upgrade.
`neuroshard.core.model.moe` is not this runtime.
"""
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from neuroshard.evolution.learned_integration import (
    LastLayerMixture, active_mlp_flops_per_token, bind_spec, code_rows,
    complete_development_gate, control_from_parent, generated_passed, load_mbpp,
    score_gate, score_retention, training_schedule,
)
from neuroshard.evolution.learned_integration_execution import (
    HOST, authorize_cpu_run,
)
from neuroshard.evolution.reference_data import conversation, identity, save, sha256
from neuroshard.evolution.seed import verify

MAX_LENGTH = 768
ACTIVE_EXPERTS_PER_TOKEN = 1


def repo_root():
    marker = Path('scripts/programming_sandbox.py')
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).is_file():
            return parent
    raise FileNotFoundError('programming_sandbox.py is not next to this source tree')


def sandbox_check():
    path = repo_root() / 'scripts' / 'programming_sandbox.py'
    spec = importlib.util.spec_from_file_location('programming_sandbox', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.check


def rss_bytes():
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('Unable to read process RSS')


def expert_from_llama_mlp(mlp):
    from neuroshard.evolution.learned_integration import SwiGLUExpert
    hidden = mlp.gate_proj.in_features
    intermediate = mlp.gate_proj.out_features
    expert = SwiGLUExpert(hidden, intermediate)
    with torch.no_grad():
        expert.gate_proj.weight.copy_(mlp.gate_proj.weight.detach().float())
        expert.up_proj.weight.copy_(mlp.up_proj.weight.detach().float())
        expert.down_proj.weight.copy_(mlp.down_proj.weight.detach().float())
    return expert


def freeze_except(model, trainable):
    wanted = {id(parameter) for parameter in trainable}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in wanted)


def last_mlp(model):
    return model.model.layers[-1].mlp


def replace_last_mlp(model, module):
    model.model.layers[-1].mlp = module
    return module


def assert_matches_parent_mlp(mlp, expert):
    hidden = expert.hidden
    sample = torch.randn(2, 4, hidden)
    with torch.no_grad():
        if not torch.allclose(mlp(sample), expert(sample), atol=1e-5, rtol=1e-4):
            raise ValueError('Last-layer expert does not match the parent MLP')


def install_expansion(model):
    parent = last_mlp(model)
    expert = expert_from_llama_mlp(parent)
    assert_matches_parent_mlp(parent, expert)
    mixture = LastLayerMixture.expand(expert)
    mixture.eval()
    sample = torch.randn(2, 4, expert.hidden)
    with torch.no_grad():
        if not torch.allclose(parent(sample), mixture(sample), atol=1e-5, rtol=1e-4):
            raise ValueError('Untrained expansion does not reproduce the parent last layer')
        if int(mixture.last_choices.unique().item()) != 0:
            raise ValueError('Untrained gate must select the incumbent expert')
    replace_last_mlp(model, mixture)
    freeze_except(model, mixture.expansion_parameters())
    return mixture


def install_control(model):
    parent = last_mlp(model)
    expert = expert_from_llama_mlp(parent)
    assert_matches_parent_mlp(parent, expert)
    control = control_from_parent(expert)
    sample = torch.randn(2, 4, expert.hidden)
    with torch.no_grad():
        if not torch.allclose(parent(sample), control(sample), atol=1e-5, rtol=1e-4):
            raise ValueError('Untrained control does not reproduce the parent last layer')
    replace_last_mlp(model, control)
    freeze_except(model, control.parameters())
    return control


def collate(rows, pad_id):
    length = max(len(row['input_ids']) for row in rows)
    ids, labels, mask = [], [], []
    for row in rows:
        pad = length - len(row['input_ids'])
        ids.append(row['input_ids'] + [pad_id] * pad)
        labels.append(row['labels'] + [-100] * pad)
        mask.append([1] * len(row['input_ids']) + [0] * pad)
    return {
        'input_ids': torch.tensor(ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(mask, dtype=torch.long),
    }


def tokenize_code_rows(rows, tokenizer, max_length=MAX_LENGTH):
    tokenized = []
    for row in rows:
        messages = list(row['messages']) + [{'role': 'assistant', 'content': row['reference']}]
        encoded = conversation(tokenizer, messages, max_length)
        tokenized.append({**row, **encoded})
    return tokenized


def causal_train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError('Invalid training loss')
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def load_parent(seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(seed, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        seed, local_files_only=True, dtype=torch.float32)
    if next(model.parameters()).device.type != 'cpu':
        raise ValueError('Stage-1 host is CPU')
    model.to('cpu')
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new):
    model.eval()
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    started = time.monotonic()
    before = rss_bytes()
    with torch.inference_mode():
        output = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    seconds = max(time.monotonic() - started, 1e-6)
    memory = max(before, rss_bytes(), 1)
    text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    return text, seconds, memory


def refuse_confirmation(role):
    if role == 'confirmation':
        raise ValueError('Confirmation remains closed until a later execution freeze passes development')


def score_code_role(model, tokenizer, rows, spec, role, check):
    refuse_confirmation(role)
    if role not in ('development', 'retention'):
        raise ValueError('Only development and retention may be scored')
    scored = []
    for row in rows:
        text, seconds, memory = generate_text(
            model, tokenizer, row['messages'], spec['training']['generation_tokens'])
        scored.append({
            'id': row['id'],
            'task_id': row['task_id'],
            'text': text,
            'passed': generated_passed(text, row['setup'], row['tests'], check),
            'seconds': seconds,
            'peak_memory_bytes': memory,
            'active_experts_per_token': ACTIVE_EXPERTS_PER_TOKEN,
            'setup': row['setup'],
            'tests': row['tests'],
        })
        print(json.dumps({
            'phase': role,
            'task_id': row['task_id'],
            'passed': scored[-1]['passed'],
            'seconds': scored[-1]['seconds'],
        }), flush=True)
    return scored


def load_general_conversations(path, identities):
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        records[row['id']] = row
    conversations = []
    for item in identities:
        if item['id'] not in records:
            raise ValueError('Frozen general identity is missing from the local corpus: ' + item['id'])
        row = records[item['id']]
        if row.get('kind') != 'general':
            raise ValueError('Frozen general identity is not a general conversation')
        messages = list(row['messages'])
        if not messages or messages[-1]['role'] != 'assistant':
            raise ValueError('General conversation must end in an assistant answer')
        conversations.append({
            **item,
            'messages': messages[:-1],
            'reference': messages[-1]['content'],
        })
    return conversations


def score_general(model, tokenizer, conversations, spec):
    scored = []
    for row in conversations:
        text, seconds, memory = generate_text(
            model, tokenizer, row['messages'], spec['training']['generation_tokens'])
        scored.append({
            'id': row['id'],
            'row': row['row'],
            'group': row['group'],
            'text': text,
            'seconds': seconds,
            'peak_memory_bytes': memory,
            'active_experts_per_token': ACTIVE_EXPERTS_PER_TOKEN,
        })
        print(json.dumps({'phase': 'general', 'id': row['id'][:12], 'seconds': seconds}), flush=True)
    return scored


def write_role(home, name, rows):
    path = Path(home) / (name + '.jsonl')
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    return {
        'file': path.name,
        'sha256': sha256(path),
        'ids': [row['id'] for row in rows],
        'task_ids': [row['task_id'] for row in rows],
    }


def prepare_inputs(spec, mbpp, home, gold_check):
    bind_spec(spec)
    raw = load_mbpp(mbpp, spec)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    roles = {}
    for name in ('train_new', 'train_replay', 'retention', 'development', 'confirmation'):
        roles[name] = write_role(home, name, code_rows(spec['splits'][name], raw, gold_check))
    schedule = training_schedule(spec)
    result = {
        'format': spec['format'] + '/prepared',
        'spec': identity(spec),
        'roles': roles,
        'confirmation_scored': False,
        'original_final_opened': False,
        'admission_evidence': False,
        **schedule,
    }
    save(home / 'prepared.json', result)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def train_system(model, tokenizer, new_rows, replay_rows, spec):
    torch.manual_seed(spec['training']['seed'])
    tokenized = {
        'new': tokenize_code_rows(new_rows, tokenizer),
        'replay': tokenize_code_rows(replay_rows, tokenizer),
    }
    schedule = training_schedule(spec)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('No trainable parameters')
    optimizer = torch.optim.AdamW(trainable, lr=spec['training']['learning_rate'])
    model.config.use_cache = False
    history = []
    for step, batch_index in enumerate(schedule['schedule']):
        rows = [tokenized[kind][index] for kind, index in schedule['batches'][batch_index]]
        loss = causal_train_step(model, collate(rows, pad), optimizer)
        history.append({'step': step, 'loss': loss})
        print(json.dumps({'phase': 'train', 'step': step, 'loss': loss}), flush=True)
    model.config.use_cache = True
    model.eval()
    return history


def release(model):
    del model
    gc.collect()


def align_general(parent, expansion):
    if [row['id'] for row in parent] != [row['id'] for row in expansion]:
        raise ValueError('General retention identities drifted between systems')
    return [{'parent': left['text'], 'expansion': right['text']}
            for left, right in zip(parent, expansion)]


def run_stage1(*, home, seed, spec, method, execution, mbpp, general_train,
               gold_check=None, check=None):
    binding = authorize_cpu_run(spec, method, execution)
    if execution.get('host') != HOST or execution.get('gpu_launch_authorized') is not False:
        raise ValueError('Stage-1 host is CPU')
    refuse_confirmation('confirmation' if execution.get('confirmation_opened') else 'development')
    if execution.get('confirmation_opened') is not False or execution.get('confirmation_scored') is not False:
        raise ValueError('Confirmation remains closed')
    verify(seed)
    check = sandbox_check() if check is None else check
    gold_check = check if gold_check is None else gold_check
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    prepared = prepare_inputs(spec, mbpp, home / 'inputs', gold_check)
    development = read_jsonl(home / 'inputs' / 'development.jsonl')
    retention = read_jsonl(home / 'inputs' / 'retention.jsonl')
    train_new = read_jsonl(home / 'inputs' / 'train_new.jsonl')
    train_replay = read_jsonl(home / 'inputs' / 'train_replay.jsonl')
    general_rows = load_general_conversations(general_train, execution['general_retention']['rows'])

    parent, tokenizer = load_parent(seed)
    hidden = last_mlp(parent).gate_proj.in_features
    intermediate = last_mlp(parent).gate_proj.out_features
    flops = active_mlp_flops_per_token(hidden, intermediate)
    parent_dev = score_code_role(parent, tokenizer, development, spec, 'development', check)
    parent_ret = score_code_role(parent, tokenizer, retention, spec, 'retention', check)
    parent_gen = score_general(parent, tokenizer, general_rows, spec)
    save(home / 'parent-development.json', parent_dev)
    save(home / 'parent-retention.json', parent_ret)
    save(home / 'parent-general.json', parent_gen)
    release(parent)

    expansion, tokenizer = load_parent(seed)
    install_expansion(expansion)
    expansion_history = train_system(expansion, tokenizer, train_new, train_replay, spec)
    save(home / 'expansion-training.json', expansion_history)
    expansion_dev = score_code_role(expansion, tokenizer, development, spec, 'development', check)
    expansion_ret = score_code_role(expansion, tokenizer, retention, spec, 'retention', check)
    expansion_gen = score_general(expansion, tokenizer, general_rows, spec)
    save(home / 'expansion-development.json', expansion_dev)
    save(home / 'expansion-retention.json', expansion_ret)
    save(home / 'expansion-general.json', expansion_gen)
    release(expansion)

    control, tokenizer = load_parent(seed)
    install_control(control)
    control_history = train_system(control, tokenizer, train_new, train_replay, spec)
    save(home / 'control-training.json', control_history)
    control_dev = score_code_role(control, tokenizer, development, spec, 'development', check)
    save(home / 'control-development.json', control_dev)
    release(control)

    development_score = score_gate(
        parent_dev, control_dev, expansion_dev, spec, role='development',
        general=align_general(parent_gen, expansion_gen), check=None)
    retention_score = score_retention(parent_ret, expansion_ret, spec)
    gate = complete_development_gate(development_score, retention_score, spec)
    result = {
        'format': spec['format'] + '/stage1-result',
        'binding': binding,
        'prepared': identity(prepared),
        'host': HOST,
        'gpu_launch_authorized': False,
        'admission_evidence': False,
        'confirmation_opened': False,
        'confirmation_scored': False,
        'max_length': MAX_LENGTH,
        'active_mlp_flops_per_token': flops,
        'development': development_score,
        'retention': retention_score,
        'development_gate': gate,
        'next': gate['next'],
    }
    save(home / 'result.json', result)
    print(json.dumps({
        'passed': gate['passed'],
        'next': gate['next'],
        'confirmation_opened': False,
        'gpu_launch_authorized': False,
    }), flush=True)
    return result
