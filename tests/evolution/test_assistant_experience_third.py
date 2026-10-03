from neuroshard.evolution import assistant_experience_gate as gate
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

THIRD = read(ROOT / 'config/experiments/assistant-experience-third.json')
LEARNING = read(ROOT / THIRD['learning_contract']['path'])


def without_control(section):
    return {key: value for key, value in section.items() if key not in gate.CONTROL_CHECKS}


def test_the_third_attempt_keeps_the_earlier_gates_without_the_update_comparisons():
    assert sha256(ROOT / THIRD['learning_contract']['path']) == THIRD['learning_contract']['sha256']
    assert THIRD['policy'] == LEARNING['policy']
    assert {k: v for k, v in THIRD['development_gate'].items() if k != 'latency'} == {
        k: v for k, v in without_control(LEARNING['development_gate']).items() if k != 'latency'}
    earlier = LEARNING['confirmation2_gate']
    changed = {'data', 'opens', 'serving', 'bootstrap_seed', 'latency'}
    assert {k: v for k, v in THIRD['confirmation_gate'].items() if k not in changed} == {
        k: v for k, v in without_control(earlier).items() if k not in changed}
    assert not set(gate.CONTROL_CHECKS) & (set(THIRD['development_gate']) | set(THIRD['confirmation_gate']))


def test_the_third_confirmation_opens_its_own_sealed_split_with_a_fresh_bootstrap_seed():
    manifest = read(ROOT / THIRD['confirmation_gate']['data'])
    assert manifest['split'] == 'confirmation3' and manifest['count'] == THIRD['confirmation_gate']['cases'] == 192
    seeds = {LEARNING[name]['bootstrap_seed'] for name in ('confirmation_gate', 'confirmation2_gate')}
    seed = THIRD['confirmation_gate']['bootstrap_seed']
    assert not {seed, seed + 1} & (seeds | {s + 1 for s in seeds})


def test_the_candidate_is_the_published_round4_update_unchanged():
    candidate = THIRD['candidate']
    trained = read(ROOT / candidate['training_report'])
    assert candidate['system'] == 'update' and candidate['training_report'].endswith(f"round{candidate['round']}-report.json")
    assert candidate['trainable_sha256'] == trained['training']['update']['trainable_sha256']
    assert candidate['integration_sha256'] == trained['integration']['integration_sha256']
    served = read(ROOT / 'config/experiments/assistant-experience-confirmation2-update-result.json')['reply']['checkpoint']
    assert served['trainable_sha256'] == candidate['trainable_sha256']
    assert served['trainable_parameters'] == candidate['trained_parameters'] == 62_914_560
    assert candidate['trained_parameters'] / candidate['backbone_parameters'] < 0.02
    assert not THIRD['training_execution_authorized'] and not THIRD['gpu_launch_authorized']
