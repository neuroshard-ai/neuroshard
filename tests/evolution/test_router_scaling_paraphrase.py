import random

import pytest

from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_paraphrase as paraphrase


def test_every_real_training_and_integration_opening_is_paraphrased_with_its_meaning_kept():
    for split in ('fit', 'test'):
        cases = data.real_cases(split, limit=48 if split == 'fit' else None)
        out = paraphrase.paraphrased_cases(cases, 1)
        assert len(out) == len(cases)
        for case, copy in zip(cases, out):
            assert copy['paraphrase_of'] == case['id'] and copy['labels'] == case['labels']
            assert copy['user_turns'][1:] == case['user_turns'][1:]
            assert copy['user_turns'][0] != case['user_turns'][0]
            assert paraphrase.preserved(case['user_turns'][0], copy['user_turns'][0])


def test_paraphrases_are_deterministic_and_disjoint_from_the_held_out_rewordings():
    cases = data.real_cases('fit', limit=12)
    assert paraphrase.paraphrased_cases(cases, 5) == paraphrase.paraphrased_cases(cases, 5)
    held_out = [t for openings, follow in data.REAL_UNSEEN.values() for t in openings + follow]
    frames = paraphrase.DRAFT_FRAMES + paraphrase.MEETING_FRAMES
    for frame in frames:
        start = frame.split('{')[0].strip().lower()
        assert start and all(not template.lower().startswith(start) for template in held_out)
    unseen = {text for case in data.real_unseen_cases() for text in case['user_turns']}
    made = {text for case in paraphrase.paraphrased_cases(cases, 5) for text in case['user_turns']}
    assert not unseen & made


def test_paraphrase_refuses_sealed_splits_and_unknown_forms():
    assert paraphrase.paraphrase('Please summarize the notes.', random.Random(0)) is None
    with pytest.raises(ValueError):
        paraphrase.paraphrased_cases([{'id': 'real-confirmation-x', 'user_turns': ['x'], 'labels': ['drafting']}], 0)
    assert not paraphrase.preserved('Create a draft for X.', 'anything')
