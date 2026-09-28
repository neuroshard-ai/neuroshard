import copy
import json

import pytest

from neuroshard.evolution import assistant_experience as experience
from neuroshard.evolution import assistant_experience_train as trainer
from neuroshard.evolution import assistant_replay as replay
from neuroshard.evolution import assistant_rollout as rollout
from neuroshard.evolution import assistant_selector as selector
from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution import assistant_workflow_data as data
from neuroshard.evolution import assistant_workspace as workspace
from neuroshard.evolution.modular_reference_execution import ROOT, read

from test_assistant_workflow import envelope, policy, reference_texts, reply
from test_granite_tokenizer import granite_like, load_tiny

torch = pytest.importorskip('torch')

# Granite-shaped rendering: tools in the system turn, tool results as user turns.
TEMPLATE = (
    "{%- set system = messages[0].content if messages[0].role == 'system' else '' -%}"
    "{%- if tools %}{% set system = system + '\\n<tools>' + (tools | tojson) + '</tools>' %}{% endif -%}"
    "{%- if system %}<|start_of_role|>system<|end_of_role|>{{ system }}<|end_of_text|>\n{% endif -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' %}<|start_of_role|>user<|end_of_role|>{{ m.content }}<|end_of_text|>\n"
    "{%- elif m.role == 'assistant' %}<|start_of_role|>assistant<|end_of_role|>{{ m.content }}<|end_of_text|>\n"
    "{%- elif m.role == 'tool' %}<|start_of_role|>user<|end_of_role|><tool_response>{{ m.content }}</tool_response><|end_of_text|>\n"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>{% endif -%}")
CARD = read(ROOT / 'config/experiments/assistant-experience-learning.json')['coaching']['card']
SPEC = {'layers': [0, 1], 'rank': 4, 'alpha': 8, 'seed': 27092026, 'steps': 6, 'gradient_accumulation': 4,
        'experience_per_replay': 3, 'gradient_clip': 1.0, 'weight_decay': 0.01, 'betas': [0.9, 0.999],
        'epsilon': 1e-8, 'warmup_steps': 2, 'learning_rates': {'update': 1e-3, 'addition': 1e-2}}


def scripted(case, texts, executed_policy=None):
    replies = iter(texts)
    return workflow.execute(case, lambda m, t: reply(next(replies)), executed_policy or policy())


@pytest.fixture(scope='module')
def tokenizer(tmp_path_factory):
    directory = granite_like(tmp_path_factory.mktemp('granite'))
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    return load_tiny(directory)[0]


def tiny_model(tokenizer, seed=0):
    from transformers import GraniteConfig, GraniteForCausalLM

    torch.manual_seed(seed)
    config = GraniteConfig(vocab_size=len(tokenizer.runtime), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.pad_token_id, logits_scaling=2.0)
    return GraniteForCausalLM(config).eval()


def test_complete_training_conversation_becomes_verified_experience():
    case = data.make_case('train', 'recipient', 0)
    item = experience.trajectory(case, scripted(case, reference_texts(case)), policy(), policy(), sample=0)
    assert item['complete'] and item['rounds'] == 2 and item['rejected_turns'] == 0
    assert item['trainable'] == [m['role'] == 'assistant' for m in item['messages']]
    assert sum(m['role'] == 'user' for m in item['messages']) == 2


def test_only_leading_verified_rounds_are_kept_and_rejected_calls_carry_no_loss():
    case = data.make_case('train', 'recipient', 1)
    texts = reference_texts(case)
    texts.insert(0, envelope('read_annotated', {'document_id': 'x'}))
    texts[-2] = envelope('save_draft', {**case['turns'][1]['expected'], 'total': 0})
    item = experience.trajectory(case, scripted(case, texts), policy(), policy(), sample=3)
    assert item['rounds'] == 1 and not item['complete'] and item['rejected_turns'] == 1
    assert sum(m['role'] == 'user' for m in item['messages']) == 1
    assistant = [i for i, m in enumerate(item['messages']) if m['role'] == 'assistant']
    assert not item['trainable'][assistant[0]] and all(item['trainable'][i] for i in assistant[1:])
    wrong_first = reference_texts(case)
    wrong_first[2] = envelope('save_draft', {**case['turns'][0]['expected'], 'total': 0})
    assert experience.trajectory(case, scripted(case, wrong_first), policy(), policy(), sample=4) is None


def test_coached_rollout_is_stored_without_its_card_and_goals_stay_private():
    case = data.make_case('train', 'latest', 2)
    coached = experience.coached(policy(), CARD)
    result = scripted(case, reference_texts(case), coached)
    assert CARD in result['messages'][0]['content']
    item = experience.trajectory(case, result, coached, policy(), sample=0, coaching=True)
    assert item['coached'] and item['messages'][0]['content'] == policy()['system_instruction']
    assert CARD not in json.dumps(item['messages'])
    with pytest.raises(ValueError, match='coaching flag'):
        experience.trajectory(case, result, coached, policy(), sample=0)
    with pytest.raises(ValueError, match='training goals'):
        experience.trajectory(data.make_case('development', 'latest', 0), result, coached, policy(), sample=0)
    forged = copy.deepcopy(result)
    forged['calls'][0]['result'] = {'documents': []}
    with pytest.raises(ValueError):
        experience.trajectory(case, forged, coached, policy(), sample=0, coaching=True)


def test_selection_prefers_more_rounds_then_fewer_calls_and_near_policy_bounds_coaching():
    rows = [
        {'case_id': 'a', 'rounds': 1, 'model_calls': 3, 'coached': False, 'sample': 0, 'complete': False,
         'transcript_sha256': 't0', 'messages': [{'role': 'assistant', 'content': 'x'}]},
        {'case_id': 'a', 'rounds': 2, 'model_calls': 8, 'coached': False, 'sample': 1, 'complete': True,
         'transcript_sha256': 't1', 'messages': [{'role': 'assistant', 'content': 'y'}]},
        {'case_id': 'a', 'rounds': 2, 'model_calls': 6, 'coached': True, 'sample': 2, 'complete': True,
         'transcript_sha256': 't2', 'messages': [{'role': 'assistant', 'content': 'z'}]},
        {'case_id': 'a', 'rounds': 2, 'model_calls': 6, 'coached': True, 'sample': 3, 'complete': True,
         'transcript_sha256': 't3', 'messages': [{'role': 'assistant', 'content': 'z'}]},
    ]
    chosen = experience.select(rows, per_case=2)
    assert [row['sample'] for row in chosen] == [2, 1]
    assert not experience.needs_coaching({'id': 'a'}, rows) and experience.needs_coaching({'id': 'a'}, rows[2:])
    kept, ceiling = experience.near_policy(rows, {'t0': .4, 't1': .9, 't2': .7, 't3': 1.2})
    assert ceiling == .9 and [row['sample'] for row in kept] == [0, 1, 2]
    assert experience.near_policy(rows[2:], {'t2': .1, 't3': .1}) == ([], None)


def test_labels_cover_trainable_assistant_text_and_end_token_only(tokenizer):
    case = data.make_case('train', 'copy', 0)
    texts = [envelope('read_annotated', {'document_id': 'x'})] + reference_texts(case)
    item = experience.trajectory(case, scripted(case, texts), policy(), policy(), sample=0)
    sequence = trainer.encode(tokenizer, item, workspace.TOOLS)
    labeled = tokenizer.decode([i for i, label in zip(sequence['input_ids'], sequence['labels']) if label != -100])
    trained = [m['content'] + trainer.END for m, t in zip(item['messages'], item['trainable']) if t]
    assert labeled == ''.join(trained)
    assert 'read_annotated' not in labeled and policy()['system_instruction'] not in labeled
    with pytest.raises(ValueError, match='only assistant'):
        trainer.encode(tokenizer, {**item, 'trainable': [True] * len(item['messages'])}, workspace.TOOLS)


def sequences(tokenizer):
    items = []
    for index in range(3):
        case = data.make_case('train', 'copy', index)
        items.append(experience.trajectory(case, scripted(case, reference_texts(case)), policy(), policy(), sample=0))
    prompt = replay.prompts(1, [])[0]
    extra = replay.replay_item(prompt, {'terminated': True, 'text': 'Okay.', 'token_ids': [1]})
    return ([trainer.encode(tokenizer, item, workspace.TOOLS) for item in items],
            [trainer.encode(tokenizer, extra, extra['tools'])])


def test_addition_starts_exactly_at_the_parent_and_never_touches_backbone_tensors(tokenizer, tmp_path):
    experience_rows, replay_rows = sequences(tokenizer)
    parent = tiny_model(tokenizer)
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    reference_loss = trainer.negative_log_likelihood(parent, experience_rows[0])
    model = tiny_model(tokenizer)
    trainer.prepare(model, 'addition', SPEC)
    assert trainer.negative_log_likelihood(model, experience_rows[0]) == pytest.approx(reference_loss, abs=1e-6)
    model = tiny_model(tokenizer)
    trainable, receipt = trainer.train(model, 'addition', experience_rows, replay_rows, SPEC)
    # q: 32 -> 32, v: 32 -> 16 (two KV heads of width 8), rank 4, two layers.
    assert receipt['trainable_parameters'] == 2 * (4 * 32 + 32 * 4 + 4 * 32 + 16 * 4)
    assert receipt['losses'][-1] < receipt['losses'][0] and receipt['microbatches'] == 24
    after = {k.replace('.base.', '.'): v for k, v in model.state_dict().items() if 'lora_' not in k}
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)
    manifest = trainer.checkpoint(tmp_path / 'addition', trainable, receipt, {'data': 'test'})
    restored = tiny_model(tokenizer)
    trainer.load_trainable(restored, 'addition', SPEC, tmp_path / 'addition')
    assert trainer.negative_log_likelihood(restored, experience_rows[0]) == pytest.approx(
        trainer.negative_log_likelihood(model, experience_rows[0]), abs=1e-6)
    assert manifest['trainable_parameters'] == receipt['trainable_parameters']
    with pytest.raises(ValueError, match='another arm'):
        trainer.load_trainable(tiny_model(tokenizer), 'update', SPEC, tmp_path / 'addition')


def test_adapter_initialization_never_draws_a_cpu_generator_into_another_device(tokenizer, monkeypatch):
    # ATen rejects a generator whose device differs from the tensor's; mirror that check on CPU-only CI.
    original = torch.Tensor.uniform_

    def checked(tensor, *args, generator=None, **kwargs):
        if generator is not None and generator.device.type != tensor.device.type:
            raise RuntimeError(f"Expected a '{tensor.device.type}' device type for generator")
        return original(tensor, *args, generator=generator, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'uniform_', checked)
    reference = tiny_model(tokenizer)
    expected = trainer.prepare(reference, 'addition', SPEC)
    remote = tiny_model(tokenizer).to('meta')
    placed = trainer.prepare(remote, 'addition', SPEC)
    assert all(placed[name].device.type == 'meta' for name in placed)
    assert set(placed) == set(expected) and all(placed[n].shape == expected[n].shape for n in expected)
    again = trainer.prepare(tiny_model(tokenizer), 'addition', SPEC)
    assert all(torch.equal(again[n], expected[n]) for n in expected)


def test_serving_stores_updated_projections_in_backbone_dtype_and_keeps_adapters(tokenizer):
    experience_rows, replay_rows = sequences(tokenizer)
    model = tiny_model(tokenizer).to(torch.bfloat16)
    trainer.train(model, 'update', experience_rows, replay_rows, SPEC)
    trained = trainer.negative_log_likelihood(model, experience_rows[0])
    assert trainer.serving(model, SPEC) == 4
    assert not any(isinstance(m, trainer.MasterLinear) for m in model.modules())
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert trainer.negative_log_likelihood(model, experience_rows[0]) == pytest.approx(trained, abs=1e-6)
    adapter = tiny_model(tokenizer)
    trainer.prepare(adapter, 'addition', SPEC)
    assert trainer.serving(adapter, SPEC) == 0
    assert sum(isinstance(m, trainer.LoRALinear) for m in adapter.modules()) == 4


def version_rollouts(case):
    """One success that reads the latest approved revision, one failure that reads an older one."""
    from neuroshard.evolution.modular_reference_execution import identity

    project = case['turns'][0]['expected']['project']
    approved = sorted((d for d in case['world']['documents'] if d['project'] == project and d['status'] == 'approved'),
                      key=lambda d: d['revision'])
    older, latest = approved[0], approved[-1]
    good = reference_texts(case)
    bad = list(good)
    bad[1] = envelope('read_document', {'document_id': older['id']})
    bad[2] = envelope('save_draft', {**case['turns'][0]['expected'], 'source_ids': [older['id']]})
    rows = []
    for sample, texts in enumerate([good, bad, good]):
        executed = experience.coached(policy(), CARD) if sample == 2 else policy()
        rows.append({'case_id': case['id'], 'sample': sample, 'policy_sha256': identity(executed),
                     'result': scripted(case, texts, executed)})
    return rows, latest, older


def test_decision_pairs_isolate_the_version_choice_from_verified_natural_rollouts():
    case = data.make_case('train', 'copy', 3)
    rows, latest, older = version_rollouts(case)
    pairs = experience.decision_pairs(case, rows, policy(), per_case=8)
    assert len(pairs) == 1 and pairs[0]['samples'] == [0, 1]
    pair = pairs[0]
    assert pair['messages'][-1]['role'] == 'tool' and 'list_documents' in pair['messages'][-2]['content']
    assert latest['id'] in pair['chosen'] and older['id'] in pair['rejected']
    assert experience.decision_pairs(case, rows[:1], policy(), per_case=8) == []
    forged = copy.deepcopy(rows)
    forged[1]['result']['score']['round_successes'] = [True]
    with pytest.raises(ValueError, match='re-verify'):
        experience.decision_pairs(case, forged, policy(), per_case=8)
    with pytest.raises(ValueError, match='training goals'):
        experience.decision_pairs(data.make_case('development', 'copy', 0), rows, policy(), per_case=8)


def followup_rollouts(case):
    """A full success and a failure that first differs at the follow-up save_draft."""
    from neuroshard.evolution.modular_reference_execution import identity

    good = reference_texts(case)
    bad = list(good)
    bad[6] = envelope('save_draft', {**case['turns'][1]['expected'], 'due_date': '2030-01-01'})
    early = list(good)
    early[2] = envelope('save_draft', {**case['turns'][0]['expected'], 'total': 0})
    return [{'case_id': case['id'], 'sample': sample, 'policy_sha256': identity(policy()),
             'result': scripted(case, texts)} for sample, texts in enumerate([good, bad, early, good])]


def test_divergence_pairs_prefer_the_success_at_the_first_differing_tool_call():
    case = data.make_case('train', 'recipient', 4)
    rows = followup_rollouts(case)
    pairs = experience.divergence_pairs(case, rows, policy(), per_case=4)
    assert [(p['turn'], p['samples']) for p in pairs] == [(0, [0, 2]), (1, [0, 1])]
    late = pairs[1]
    assert late['messages'][-1]['role'] == 'user' or late['messages'][-1]['role'] == 'tool'
    assert '2030-01-01' in late['rejected'] and case['turns'][1]['expected']['due_date'] in late['chosen']
    assert sum(m['role'] == 'user' for m in late['messages']) == 2
    assert len(experience.divergence_pairs(case, rows, policy(), per_case=1)) == 1
    assert experience.divergence_pairs(case, rows[:1] + rows[3:], policy(), per_case=4) == []
    with pytest.raises(ValueError, match='training goals'):
        experience.divergence_pairs(data.make_case('development', 'recipient', 0), rows, policy(), per_case=4)


def test_preference_training_widens_the_verified_margin_and_keeps_round_one_schedules(tokenizer, tmp_path):
    experience_rows, replay_rows = sequences(tokenizer)
    assert trainer.schedule(experience_rows, replay_rows, SPEC) == trainer.schedule(experience_rows, replay_rows, SPEC, 0)
    pairs = []
    for index in range(3, 6):
        case = data.make_case('train', 'copy', index)
        rows, _, _ = version_rollouts(case)
        pairs += [trainer.encode_pair(tokenizer, p, workspace.TOOLS) for p in experience.decision_pairs(case, rows, policy(), 8)]
    first = tiny_model(tokenizer)
    trainable, receipt = trainer.train(first, 'addition', experience_rows, replay_rows, SPEC)
    trainer.checkpoint(tmp_path / 'round1', trainable, receipt, {})
    model = tiny_model(tokenizer)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _, resumed = trainer.resume(model, 'addition', SPEC, tmp_path / 'round1')
    spec = {**SPEC, 'steps': 8, 'preference_per_update': 2, 'beta': 0.5, 'preference_weight': 1.0}
    _, second = trainer.train(model, 'addition', experience_rows, replay_rows, spec, trainable=resumed, pairs=pairs)
    assert second['preference_pairs'] == len(pairs) == 3 and len(second['preference_margins']) == 16
    assert second['preference_margins'][-1] > second['preference_margins'][0]
    after = {k.replace('.base.', '.'): v for k, v in model.state_dict().items() if 'lora_' not in k}
    assert all(torch.equal(after[k], before[k]) for k in before)


def test_update_changes_only_declared_projections_with_the_same_schedule(tokenizer):
    experience_rows, replay_rows = sequences(tokenizer)
    before = {k: v.clone() for k, v in tiny_model(tokenizer).state_dict().items()}
    model = tiny_model(tokenizer)
    _, update = trainer.train(model, 'update', experience_rows, replay_rows, {**SPEC, 'layers': [1]})
    after = model.state_dict()
    changed = {k for k in before if not torch.equal(after[k].to(before[k].dtype), before[k])}
    assert changed == {'model.layers.1.self_attn.q_proj.weight', 'model.layers.1.self_attn.v_proj.weight'}
    _, addition = trainer.train(tiny_model(tokenizer), 'addition', experience_rows, replay_rows, {**SPEC, 'layers': [1]})
    assert update['schedule_sha256'] == addition['schedule_sha256']
    batch = trainer.schedule(experience_rows, replay_rows, SPEC)[0]
    assert sorted(kind for kind, _ in batch) == ['experience'] * 3 + ['replay']


def test_selector_learns_only_from_success_rate_differences():
    parent = {'a': [False] * 5, 'b': [True] * 5, 'c': [True, False, False, False, False], 'd': [False] * 5}
    arm = {'a': [True] * 5, 'b': [True] * 5, 'c': [False] * 5, 'd': [False] * 5}
    rows = selector.targets(parent, arm)
    assert rows == {'a': (1.0, 1.0), 'b': (0.0, 0.0), 'c': (0.0, 0.2), 'd': (0.0, 0.0)}
    recipe = {'seed': 1, 'epsilon': 1e-6, 'learning_rate': .1, 'weight_decay': 0.0, 'updates': 200,
              'batch': 4, 'threshold': .5}
    features = {'a': [1.0, 0.0], 'b': [0.0, 1.0], 'c': [0.0, 1.0], 'd': [0.1, 1.0]}
    gate = selector.fit(features, rows, recipe)
    assert gate['rule'] == 'logistic' and gate['counts'] == {'arm_better': 1, 'parent_better': 1, 'ties': 2}
    assert selector.choose(gate, [1.0, 0.0]) and not selector.choose(gate, [0.0, 1.0])
    tied = selector.fit(features, selector.targets(parent, parent), recipe)
    assert tied['rule'] == 'constant-parent' and not selector.choose(tied, [1.0, 0.0])
    better = selector.fit({'a': [1.0]}, selector.targets({'a': [False]}, {'a': [True]}), recipe)
    assert better['rule'] == 'constant-arm'
    zero = {**gate, 'weight': [0.0, 0.0], 'bias': 0.0}
    assert not selector.choose(zero, [1.0, 0.0])


def test_batched_rollouts_run_concurrent_episodes_and_stay_replay_verifiable(tokenizer):
    model = tiny_model(tokenizer)
    generation = {'max_input_tokens': 100000, 'max_new_tokens': 3, 'temperature': .8, 'top_p': .95}
    batcher = rollout.Batcher(model, tokenizer, generation, max_batch=4, wait_seconds=.2)
    try:
        jobs = [(data.make_case('train', 'copy', i), policy(), i) for i in range(3)]
        results = rollout.rollouts(jobs, batcher.respond, workers=3)
    finally:
        batcher.close()
    assert [row['sample'] for row in results] == [0, 1, 2]
    assert max(batcher.batches) > 1
    for (case, _, _), row in zip(jobs, results):
        assert workflow.score(case, row['result'], policy()) == row['result']['score']
        assert all(g['executed'] and len(g['token_ids']) <= 3 for g in row['result']['generations'])


def test_replay_prompts_are_disjoint_from_protected_tasks_and_skip_unterminated_answers():
    plan = read(ROOT / 'config/experiments/granite-reference.json')
    rows = replay.prompts(27092029, plan['tasks'] + plan['reference_tasks'])
    assert len(rows) == sum(replay.COUNTS.values()) == 256
    assert len({row['id'] for row in rows}) == 256
    protected = plan['tasks'][0]
    with pytest.raises(ValueError, match='overlaps'):
        replay.prompts(27092029, [{'messages': [{'role': 'user', 'content': rows[0]['messages'][-1]['content']}]}])
    assert replay.replay_item(rows[0], {'terminated': False, 'text': 'cut', 'token_ids': []}) is None
    item = replay.replay_item(rows[0], {'terminated': True, 'text': 'Okay.', 'token_ids': [1]})
    assert item['trainable'][-1] and not any(item['trainable'][:-1]) and protected['id'] != item['id']
