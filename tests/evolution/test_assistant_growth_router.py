import subprocess
import sys

from neuroshard.evolution import assistant_growth_round4 as round4
from neuroshard.evolution import assistant_growth_router as router
from neuroshard.evolution import assistant_growth_run as growth
from neuroshard.evolution import assistant_schedule_data as schedule
from neuroshard.evolution import assistant_selector as selector
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read

from test_assistant_calendar import POLICY as CALENDAR
from test_assistant_growth_baseline import cloud_module

DECLARATION = read(ROOT / router.DECLARATION)


def test_a_turn_is_labelled_by_whether_its_correct_solution_uses_the_calendar(monkeypatch):
    cases = schedule.cases('train')
    chosen = [next(c for c in cases if c['family'] == family) for family in ('slot', 'invite')]
    chosen += [next(c for c in schedule.cases('cross-train') if c['family'] == family) for family in ('review', 'handoff')]
    monkeypatch.setattr(round4, 'training_cases', lambda stage: chosen)
    rows, texts = router.scheduling_turns(None, CALENDAR)
    labels = {key: target for key, (target, weight) in rows.items() if weight == 1.0}
    slot, invite, review, handoff = (case['id'] for case in chosen)
    assert labels == {f'{slot}#0': 1.0, f'{invite}#0': 1.0, f'{invite}#1': 1.0, f'{review}#0': 1.0,
                      f'{handoff}#0': 0.0, f'{handoff}#1': 1.0}
    assert texts[f'{handoff}#0'].startswith('Create a draft') and texts[f'{invite}#1'].startswith('Also invite')


def test_drafting_and_integration_turns_carry_the_labels_their_goals_imply():
    stage = round4.plan()
    _, learning, _ = growth.contracts(stage)
    rows, texts = router.drafting_turns(learning)
    assert len(rows) == 384 and {target for target, _ in rows.values()} == {0.0} and set(texts) == set(rows)
    check, _ = router.integration_turns(stage, learning)
    assert len(check) == 204
    handoffs = {key for key in check if key.endswith('#0') and key.split('#')[0] in {
        c['id'] for c in schedule.cases('cross-integration') if c['family'] == 'handoff'}}
    assert all(check[key][0] == 0.0 for key in handoffs) and len(handoffs) == 4
    meetings = sum(len(c['turns']) for c in schedule.cases('integration')) + 4 + 4
    assert sum(target == 1.0 for target, _ in check.values()) == meetings


def test_the_fit_reports_held_out_and_integration_accuracy():
    shared = [4.0] * 32
    features, rows = {}, {}
    for index in range(24):
        meeting = index % 3 != 0
        signal = [0.0] * 32
        for position in (range(5) if meeting else range(16, 21)):
            signal[position] = 0.4
        signal[30] = 0.01 * index
        features[f'case-{index}#0'] = [a + b for a, b in zip(shared, signal)]
        rows[f'case-{index}#0'] = (1.0 if meeting else 0.0, 1.0)
    check = {'integration-a#0': features['case-1#0'], 'integration-b#0': features['case-0#0']}
    check_rows = {'integration-a#0': (1.0, 1.0), 'integration-b#0': (1.0, 1.0)}
    gate, report = router.fit(features, rows, check, check_rows, 1e-6)
    assert gate['rule'] == 'centroid' and report['gate_sha256'] == identity(gate)
    assert report['held_out_accuracy'] == 1.0 and report['integration_errors'] == ['integration-b#0']
    assert report['integration_accuracy'] == 0.5 and report['scheduling_turns'] == 16
    assert selector.choose(gate, features['case-1#0']) and not selector.choose(gate, features['case-0#0'])


def test_the_router_stays_within_the_raised_ceiling_on_one_cpu_host():
    budget = DECLARATION['budget']
    reports = [read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')
               for name in ('stage1', 'round2', 'round3', 'round4')]
    spent = sum(r['resources_finished']['conservative_compute_usd'] for r in reports) + sum(
        read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['conservative_compute_usd']
        for name in ('development', 'development4', 'development5'))
    assert budget['spent_usd'] == round(spent, 2)
    total = budget['spent_usd'] + budget['router_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= budget['ceiling_usd'] == 110 and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    assert router.PROFILE not in cloud.GPU_PROFILES and router.PROFILE not in cloud.UPLOAD_PROFILES
    assert cloud.LONG_CPU_PROFILES[router.PROFILE] == (3.5, budget['router_usd'])
    assert cloud.GRANITE_PROFILES[router.PROFILE][0] == 'assistant_growth_router'
    assert cloud.remote_command(router.PROFILE)[1].endswith(router.SCRIPT)


def test_router_inventory_pins_contracts_sources_and_the_canonical_runtime():
    from neuroshard.evolution.modular_reference_execution import sha256

    execution = read(ROOT / router.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and router.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_router; assert "torch" not in sys.modules; '
             'import neuroshard.evolution.assistant_selector, neuroshard.evolution.assistant_experience_run, '
             'neuroshard.evolution.assistant_experience_train, neuroshard.evolution.assistant_calendar; '
             'root = os.path.abspath("src"); print("\\n".join(sorted(os.path.relpath(x.__file__) '
             'for x in list(sys.modules.values()) if getattr(x, "__file__", None) and '
             'os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(execution['sources'])
    canonical = read(ROOT / 'config/experiments/assistant-workflow-canonical-execution.json')
    assert all(execution[key] == canonical[key] for key in ('packages', 'python', 'required_cpu_flags', 'environment'))
    resources = cloud_module().resources(router.PROFILE)
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge' and 'upload' not in resources
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)


def test_importing_the_router_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_router; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
