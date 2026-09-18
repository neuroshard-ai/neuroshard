import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from ordinary_cloud import semantic_assets, REMOTE
from prepare_semantic_cohorts import append_gate
sys.path.pop(0)

from neuroshard.evolution import expert_router, semantic_questions
from neuroshard.evolution.reference_data import identity


def semantic_policy(model='original'):
    encoder = {'format': semantic_questions.ENCODER, 'parameters': semantic_questions.PARAMETERS,
        'max_tokens': 512, 'files': {name: identity([name, model if name == 'model.safetensors' else 'shared'])
                                    for name in semantic_questions.FILES}}
    samples = [{'id': identity([label, index]), 'route': label, 'features': [sign*1000]+[0]*383}
               for label, sign in [('parent', -1), ('new-fact', 1)] for index in range(2)]
    policy = semantic_questions.build(samples, encoder,
        {'new-fact': {'route': 'new', 'question': 'What was learned?'}})
    return {'configuration': {'semantic_questions': policy, 'learned': {'router': {'prototypes': {'parent': [], 'new': []}}}}}


def inventory(policies):
    return {'encoders': {identity(p['configuration']['semantic_questions']['encoder']): {'files': {
        name: {'sha256': digest, 'bytes': 1234} for name, digest in
        p['configuration']['semantic_questions']['encoder']['files'].items()}} for p in policies}}


def test_serving_restores_each_encoder_and_shared_files_into_the_bound_location():
    first, second = semantic_policy(), semantic_policy('updated')
    requests = semantic_assets([first, second, first], inventory([first, second]))
    assert len(requests) == 2
    for root, files in requests.items():
        assert len(files) == 6
        assert all(spec['path'].startswith(REMOTE+'/objects/policies/semantic-encoders/'+root+'/')
                   for spec in files.values())
    assert semantic_assets([{'configuration': {}}], {}) == {}


@pytest.mark.parametrize('attack', ['absent', 'missing', 'hash', 'size', 'oversized'])
def test_missing_or_substituted_auxiliary_assets_cannot_enter_native_serving(attack):
    policy = semantic_policy()
    manifest = inventory([policy])
    entry = next(iter(manifest['encoders'].values()))['files']
    if attack == 'absent': manifest = {}
    elif attack == 'missing': entry.pop('tokenizer.json')
    elif attack == 'hash': entry['model.safetensors']['sha256'] = 'f'*64
    elif attack == 'size': entry['model.safetensors']['bytes'] = True
    elif attack == 'oversized': entry['model.safetensors']['bytes'] = 256*1024**2+1
    with pytest.raises(ValueError): semantic_assets([policy], manifest)


def test_offline_appended_gates_match_the_reference_and_preserve_earlier_parameters():
    reference = Path(__file__).resolve().parents[2]/'config/experiments/question-match-probe-20260918/retrieval-pilot.py'
    samples = [{'id': identity([name, index]), 'route': name,
                'features': expert_router.normalize([sign*100, 30+index, index*7, 80-index])}
               for name, sign, count in [('parent', -1, 9), ('old', 1, 7)] for index in range(count)]
    model = expert_router.fit(samples, embedding_root='a'*64, tokenizer_root='b'*64)
    original = copy.deepcopy(model)
    for name, sign in [('new', 1), ('later', -1)]:
        samples.extend({'id': identity([name, i]), 'route': name,
            'features': expert_router.normalize([sign*30, -90-i, 80-i, i*13])} for i in range(5))
        expected = expert_router.append_route(model, samples, name, epochs=16)
        actual = append_gate(model, samples, name, reference)
        assert actual == expected and actual['base'] == original
        model = actual
