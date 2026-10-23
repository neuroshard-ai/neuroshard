"""The gate harness's pure parts: stages follow the proposed rollout order, and verdicts are read from app logs."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('gate', Path(__file__).parents[1] / 'scripts' / 'growth_quorum_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_stages_upgrade_model_holders_first_and_never_add_a_model_to_an_old_validator():
    order = ['today', 'step1', 'step2', 'step3', 'all-rc2', 'all-equipped']
    for before, after in zip(order, order[1:]):
        b, a = gate.STAGES[before], gate.STAGES[after]
        assert set(b['rc2']) <= set(a['rc2']) and set(b['model']) <= set(a['model'])
        assert set(a['model']) - set(b['model']) <= set(a['rc2'])  # a model is only ever added to an rc2 validator
    legacy_holders = [len(set(gate.STAGES[s]['model']) - set(gate.STAGES[s]['rc2'])) for s in order]
    assert legacy_holders == sorted(legacy_holders, reverse=True) and max(legacy_holders) < 4
    assert len(set(gate.STAGES['forbidden-4-legacy-holders']['model'])) == 4  # only to demonstrate the hazard


def test_verdicts_are_read_from_the_cached_worker_log_lines():
    assert gate.log_verdict('x\n2026 INFO neuroshard.settlement verification ab12 verdict True in 53.1 s\n') == 'True'
    assert gate.log_verdict('2026 INFO neuroshard.settlement verification ab12 verdict False in 9 s') == 'False'
    no_model = ('2026 ERROR neuroshard.settlement verification failed for ab12\nTraceback (most recent call last):\n'
                "FileNotFoundError: [Errno 2] No such file or directory: '/x/baseline/chat_template.jinja'\n")
    assert gate.log_verdict(no_model) == 'no-verdict (FileNotFoundError)'
    assert gate.log_verdict('NeuroShard settlement ABCI listening on 29652') == 'not-judged'
