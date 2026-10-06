import random
import subprocess
import sys

import pytest

from neuroshard.evolution import assistant_growth_router4 as router4
from neuroshard.evolution import assistant_selector as selector
from neuroshard.evolution.modular_reference_execution import ROOT, identity, read, sha256

from test_assistant_growth_baseline import cloud_module

DECLARATION = read(ROOT / router4.DECLARATION)


def turns(count=40, seed=7):
    """Meeting and drafting turns that share most of their feature, with noise; every drafting turn is listed."""
    rng = random.Random(seed)
    features, rows, drafting = {}, {}, []
    for index in range(count):
        meeting = index % 2 == 0
        vector = [3.0 + rng.uniform(-0.3, 0.3) for _ in range(16)]
        for position in (range(4) if meeting else range(8, 12)):
            vector[position] += 0.5
        key = f'case-{index:02d}#0'
        features[key], rows[key] = vector, (1.0 if meeting else 0.0, 1.0)
        if not meeting:
            drafting.append(key)
    return features, rows, drafting


def test_the_shift_is_the_largest_that_lets_at_most_the_declared_drafting_turns_cross():
    features, rows, drafting = turns()
    shift, margins, allowed = router4.calibrate(features, rows, drafting, 1e-6, 0.1)
    assert allowed == 2 and shift > 0 and set(margins) == set(rows)
    assert sum(margins[key] > -shift for key in drafting) <= allowed
    assert sum(margins[key] > -(shift + 1e-9) for key in drafting) > allowed
    flipped = {**rows, 'case-00#0': (0.0, 1.0)}
    assert router4.calibrate(features, flipped, drafting, 1e-6, 0.1)[1]['case-00#0'] == margins['case-00#0']
    features['case-01#0'] = list(features['case-00#0'])
    assert router4.calibrate(features, rows, drafting, 1e-6, 0.0)[0] == 0.0


def test_a_shift_moves_only_turns_whose_margin_lies_within_it():
    features, rows, _ = turns()
    gate = selector.fit_centroid(features, rows, 1e-6, 'message-mean')
    meeting, draft = features['case-00#0'], features['case-01#0']
    near = [(m + d) / 2 for m, d in zip(meeting, draft)]
    while selector.margin(gate, near) >= 0:
        near = [n + 0.01 * (d - m) for n, m, d in zip(near, meeting, draft)]
    distance = -selector.margin(gate, near)
    assert not selector.choose(gate, near) and not selector.choose(selector.shifted(gate, distance), near)
    assert selector.choose(selector.shifted(gate, distance + 1e-6), near)
    unshifted = selector.shifted(gate, 0.0)
    assert all(selector.choose(unshifted, value) == selector.choose(gate, value) for value in features.values())
    moved = selector.shifted(gate, 0.25)
    assert moved['shift'] == 0.25 and moved['feature'] == 'message-mean' and moved['sha256'] != gate['sha256']
    assert moved['sha256'] == identity({key: value for key, value in moved.items() if key != 'sha256'})
    with pytest.raises(ValueError):
        selector.shifted(gate, -0.1)
    with pytest.raises(ValueError):
        selector.shifted({'rule': 'constant-arm'}, 0.1)


def test_development_runs_only_if_the_shift_recovers_turns_that_need_the_calendar(monkeypatch):
    features, rows, drafting = turns()
    gate = selector.fit_centroid(features, rows, 1e-6, 'message-mean')
    meeting, draft = features['case-00#0'], features['case-01#0']
    near = [(m + d) / 2 for m, d in zip(meeting, draft)]
    while selector.margin(gate, near) >= 0:
        near = [n + 0.01 * (d - m) for n, m, d in zip(near, meeting, draft)]
    check = {'integration-a#0': near, 'integration-b#0': draft}
    check_rows = {'integration-a#0': (1.0, 1.0), 'integration-b#0': (0.0, 1.0)}
    final, report, _ = router4.calibrated(features, rows, check, check_rows, drafting, 1e-6, 0.1)
    held = report['calibration']['held_out']
    assert held['unshifted']['accuracy'] == report['held_out_accuracy']
    assert held['shifted']['drafting_to_scheduling'] <= report['calibration']['allowed_crossings'] == 2
    assert final['shift'] == report['calibration']['shift'] and report['gate_sha256'] == identity(final)
    assert report['unshifted_gate_sha256'] == identity(gate)
    distance = -selector.margin(gate, near)
    for shift, runs in ((distance + 1e-6, True), (0.0, False)):
        monkeypatch.setattr(router4, 'calibrate', lambda *args, shift=shift: (shift, {key: 0.0 for key in rows}, 2))
        _, report, _ = router4.calibrated(features, rows, check, check_rows, drafting, 1e-6, 0.1)
        integration = report['calibration']['integration']
        assert integration['unshifted']['calendar_to_drafting'] == ['integration-a#0']
        assert integration['shifted']['calendar_to_drafting'] == ([] if runs else ['integration-a#0'])
        assert integration['shifted']['other_to_scheduling'] == [] and report['development_runs'] is runs


def test_the_calibrated_refit_stays_within_the_ceiling_on_one_cpu_host():
    budget = DECLARATION['budget']
    reports = [read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')
               for name in ('stage1', 'round2', 'round3', 'round4', 'router2', 'router3')]
    spent = sum(r['resources_finished']['conservative_compute_usd'] for r in reports) + sum(
        read(ROOT / f'config/experiments/assistant-growth-{name}-report.json')['conservative_compute_usd']
        for name in ('development', 'development4', 'development5'))
    assert budget['spent_usd'] == round(spent, 2)
    total = budget['spent_usd'] + budget['router_usd'] + budget['development_usd'] + budget['confirmation_usd']
    assert total <= budget['ceiling_usd'] == 110 and f'${total:.2f}' in budget['allowances']
    cloud = cloud_module()
    assert router4.PROFILE not in cloud.GPU_PROFILES and router4.PROFILE not in cloud.UPLOAD_PROFILES
    assert cloud.LONG_CPU_PROFILES[router4.PROFILE] == (2, budget['router_usd'])
    assert cloud.GRANITE_PROFILES[router4.PROFILE][0] == 'assistant_growth_router4'
    assert cloud.remote_command(router4.PROFILE)[1].endswith(router4.SCRIPT)
    assert DECLARATION['router']['calibration']['rate'] == 0.01


def test_calibrated_refit_inventory_pins_contracts_sources_and_the_canonical_runtime():
    execution = read(ROOT / router4.EXECUTION)
    for name, digest in execution['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(execution['contracts']) <= set(execution['sources']) and router4.SCRIPT in execution['sources']
    probe = ('import os, sys; import neuroshard.evolution.assistant_growth_router4; assert "torch" not in sys.modules; '
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
    resources = cloud_module().resources(router4.PROFILE)
    assert not resources['gpu'] and resources['instance_type'] == 'r7i.4xlarge' and 'upload' not in resources
    assert (execution['prepare_seconds'] + execution['worker_seconds'] + resources['setup_seconds']
            + resources['copy_seconds'] + 600 <= resources['hours'] * 3600)


def test_importing_the_calibrated_refit_does_not_load_torch():
    probe = 'import sys, neuroshard.evolution.assistant_growth_router4; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', probe], check=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'src')})
