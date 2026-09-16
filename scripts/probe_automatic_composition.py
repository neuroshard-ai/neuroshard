#!/usr/bin/env python3
"""One frozen, bounded planner screen; no training or native settlement changes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import time


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def parse(text):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('Duplicate model-produced key')
        return dict(pairs)
    value = json.loads(text, object_pairs_hook=unique)
    if (not isinstance(value, dict) or set(value) != {'questions'}
            or not isinstance(value['questions'], list) or not 1 <= len(value['questions']) <= 2
            or any(not isinstance(q, str) or not q.strip() or len(q.encode()) > 2048 for q in value['questions'])
            or len(set(value['questions'])) != len(value['questions'])):
        raise ValueError('Invalid bounded model-produced questions')
    return value['questions']


def typed_questions(text, messages):
    """Compile required subjects and expert scope without any reference answers."""
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('Duplicate model-produced key')
        return dict(pairs)
    value = json.loads(text, object_pairs_hook=unique)
    if (not isinstance(value, dict) or set(value) != {'tasks'}
            or not isinstance(value['tasks'], list) or not 1 <= len(value['tasks']) <= 2):
        raise ValueError('Require one or two bounded neural tasks')
    context = '\n'.join(message['content'] for message in messages)
    questions, tasks = [], []
    for task in value['tasks']:
        if not isinstance(task, dict):
            raise ValueError('Invalid neural task')
        if task.get('expert') == 'directory':
            if (set(task) != {'expert', 'name', 'field'} or not isinstance(task['name'], str)
                    or not re.fullmatch(r'[A-Za-z]+(?: [A-Za-z]+){1,3}', task['name'])
                    or not re.search(r'(?<!\w)' + re.escape(task['name']) + r'(?!\w)', context, re.I)
                    or task['field'] not in ('city', 'profession', 'instrument', 'hobby')):
                raise ValueError('Directory tasks require an explicit mentioned person and supported field')
            questions.append('In the fictional Luma directory, what is ' + task['name'] + "'s " + task['field'] + '?')
        elif task.get('expert') in ('protocol', 'parent'):
            if (set(task) != {'expert', 'question'} or not isinstance(task['question'], str)
                    or not task['question'].strip() or len(task['question'].encode()) > 2048):
                raise ValueError('Require a bounded specialist or general question')
            prefix = 'Regarding NeuroShard 0.4.0, ' if task['expert'] == 'protocol' else ''
            questions.append(prefix + task['question'])
        else:
            raise ValueError('Unknown neural expert')
        tasks.append(task)
    return questions, tasks


def score(text, required, messages=(), style='questions', expected_tasks=None):
    try:
        questions, tasks = typed_questions(text, messages) if style == 'typed' else (parse(text), None)
    except (ValueError, TypeError):
        return {'valid': False, 'complete': False}
    def matches(index, position):
        if not all(term in questions[index].casefold() for term in required[position]):
            return False
        if expected_tasks is None:
            return True
        return tasks is not None and all(tasks[index].get(key) == value
                                        for key, value in expected_tasks[position].items())
    complete = len(questions) == len(required) and any(
        all(matches(index, position) for position, index in enumerate(order))
        for order in itertools.permutations(range(len(questions))))
    return {'valid': True, 'complete': complete, 'questions': questions, 'tasks': tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256 or sha(Path(__file__)) != args.source_sha256:
        raise ValueError('Commit and freeze the exact probe and plan before execution')
    plan = json.loads(args.plan.read_bytes())
    graph = json.loads(args.graph.read_bytes())
    if plan['model'] != {key: graph['descriptor']['interpreter'][key] for key in plan['model']}:
        raise ValueError('Probe substituted its preserved interpreter')
    args.home.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import requests
    weight = args.home / 'model.safetensors'
    url = 'https://huggingface.co/' + plan['model']['repo'] + '/resolve/' + plan['model']['revision'] + '/model.safetensors'
    with requests.get(url, stream=True, timeout=(15, 60)) as response:
        if response.status_code != 200:
            raise ValueError('Pinned public interpreter unavailable')
        with weight.open('wb') as stream:
            for block in response.iter_content(4 * 1024**2):
                stream.write(block)
                if stream.tell() > 4 * 1024**3 or time.monotonic() - started > 180:
                    raise ValueError('Bounded interpreter download exceeded its limit')
    if sha(weight) != plan['model']['weight_sha256']:
        raise ValueError('Original interpreter weights changed')
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = LlamaConfig(**graph['parent']['config'])
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    tensors = load_file(weight)
    missing, extra = model.load_state_dict(tensors, strict=False)
    if set(missing) != {'lm_head.weight'} or extra:
        raise ValueError('Interpreter parameter coverage changed')
    model.tie_weights()
    del tensors
    model.to('cuda')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.seed, local_files_only=True)
    for name, expected in graph['tokenizer']['files'].items():
        if sha(args.seed / name) != expected:
            raise ValueError('Interpreter tokenizer changed')
    prefix = [{'role': 'system', 'content': plan['instruction']}]
    for question, questions in plan['examples']:
        answer = questions if plan.get('style') == 'typed' else {'questions': questions}
        prefix.extend([{'role': 'user', 'content': question},
                       {'role': 'assistant', 'content': json.dumps(answer)}])
    results = []
    for case in plan['cases']:
        # Evaluation terms and IDs never enter the model prompt.
        messages = prefix + case['messages']
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + plan['generation']['max_new_tokens'] > plan['generation']['max_context']:
            raise ValueError('No silent planner context truncation')
        inputs = torch.tensor([ids], dtype=torch.long, device='cuda')
        at = time.monotonic()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=plan['generation']['max_new_tokens'], do_sample=False, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        tokens = outputs[0, len(ids):].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        results.append({'id': case['id'], 'prompt_ids': ids, 'output_ids': tokens, 'text': text,
                        'seconds': time.monotonic() - at,
                        **score(text, case['required_terms'], case['messages'], plan.get('style', 'questions'),
                                case.get('expected_tasks'))})
        (args.home / 'responses.json').write_text(json.dumps(results, sort_keys=True, indent=2) + '\n')
    passed = (sum(row['complete'] for row in results) >= plan['pass_rule']['valid_and_complete_cases_at_least']
              and all(row['complete'] for row in results if row['id'].startswith(('pronoun-', 'prior-'))))
    result = {'passed': passed, 'complete': sum(row['complete'] for row in results), 'count': len(results),
        'plan_sha256': args.plan_sha256, 'source_sha256': args.source_sha256,
        'seconds': time.monotonic() - started, 'responses': results,
        'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                    'weights': 'float32', 'autocast': 'bfloat16', 'cache': True}, 'scope': plan['scope']}
    (args.home / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('passed', 'complete', 'count', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
