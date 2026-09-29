import pytest

from neuroshard.evolution import assistant_committee as committee

torch = pytest.importorskip('torch')

READ = '<tool_call>\n{"name": "read_document", "arguments": {"document_id": "doc-1"}}\n</tool_call>'
READ_REORDERED = '<tool_call>{"arguments": {"document_id": "doc-1"}, "name": "read_document"}</tool_call>'
OTHER = '<tool_call>\n{"name": "read_document", "arguments": {"document_id": "doc-2"}}\n</tool_call>'


def proposal(text, terminated=True, executed=True):
    return {'text': text, 'terminated': terminated, 'executed': executed, 'token_ids': [1]}


def test_actions_are_canonical_calls_finish_or_never_agreeing_invalids():
    assert committee.action(proposal(READ), 0) == committee.action(proposal(READ_REORDERED), 1)
    assert committee.action(proposal('Saved the draft.'), 0) == committee.action(proposal('Done.'), 1) == 'finish'
    assert committee.action(proposal(READ, terminated=False), 0) != committee.action(proposal(READ, terminated=False), 1)
    assert committee.action(proposal('<tool_call>{bad json}</tool_call>'), 2).startswith('invalid')
    assert committee.action(proposal('   '), 0).startswith('invalid')
    assert committee.action(proposal('', executed=False), 0) == 'skip'


def test_the_vote_needs_more_modules_than_the_parent_to_override_it():
    parent = 3
    chosen, tally = committee.vote([proposal(OTHER), proposal(OTHER), proposal(READ), proposal(READ)], parent)
    assert chosen == 2 and tally['winner'].startswith('call:') and 'doc-1' in tally['winner']
    chosen, _ = committee.vote([proposal(OTHER), proposal(OTHER), proposal(OTHER), proposal(READ)], parent)
    assert chosen == 0
    chosen, _ = committee.vote([proposal(OTHER), proposal('Done.'), proposal('x', terminated=False), proposal(READ)], parent)
    assert chosen == parent
    chosen, _ = committee.vote([proposal('a', False), proposal('b', False), proposal(READ), proposal('Done.')], parent)
    assert chosen == parent


def test_switched_adapters_reproduce_each_saved_arm_and_the_parent_exactly(tmp_path):
    from neuroshard.evolution import assistant_experience_train as trainer

    from test_granite_partition import canonical, tiny_checkpoint

    checkpoint = tiny_checkpoint(tmp_path / 'granite')
    spec = {'layers': [2, 3], 'rank': 4, 'alpha': 8, 'seed': 1,
            'projections': ['self_attn.q_proj', 'self_attn.v_proj', 'mlp.up_proj']}
    arms = []
    for index in range(3):
        model = canonical(checkpoint)
        trainable = trainer.prepare(model, 'addition', {**spec, 'seed': index})
        generator = torch.Generator().manual_seed(10 + index)
        with torch.no_grad():
            for name, value in trainable.items():
                if name.endswith('lora_b'):
                    value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
        directory = tmp_path / f'arm-{index}'
        trainer.checkpoint(directory, trainable, {'arm': 'addition', 'optimizer_state': {}, 'trainable_parameters': 1,
                                                  'steps': 0, 'schedule_sha256': '', 'losses': [0.0]}, {})
        arms.append(directory)
    switch = committee.Switch()
    served = canonical(checkpoint)
    assert committee.attach(served, spec, arms, switch) == 6
    ids = torch.randint(2, 96, (1, 9), generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        assert torch.equal(served(ids).logits, canonical(checkpoint)(ids).logits)
        for index, directory in enumerate(arms):
            single = canonical(checkpoint)
            trainer.load_trainable(single, 'addition', {**spec, 'seed': index}, directory)
            with switch.using(index):
                assert torch.equal(served(ids).logits, single(ids).logits)
        assert switch.active is None


def test_the_committee_answers_each_request_with_the_voted_member():
    calls = []

    def member(text):
        def respond(messages, tools):
            calls.append(text)
            return proposal(text)
        return respond

    respond = committee.responder([member(OTHER), member(READ), member(READ)], member('Done.'))
    reply = respond([{'role': 'user', 'content': 'x'}], [])
    assert reply['text'] == READ and reply['committee']['chosen'] == 1 and len(calls) == 4
