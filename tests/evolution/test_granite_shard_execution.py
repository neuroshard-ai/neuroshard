import copy
import importlib.util
import subprocess
import sys

import pytest

from neuroshard.evolution import granite_shard_execution as execution
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

PLAN = read(ROOT / execution.PLAN)


def cloud_module():
    spec = importlib.util.spec_from_file_location('shard_cloud', ROOT / 'scripts/modular_reference_cloud.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workload_is_every_recorded_canonical_generation():
    requests = execution.workload(PLAN)
    assert len(requests) == 230 and len({r['id'] for r in requests}) == 230
    assert all(r['expected'][-1] == 100257 and r['max_new_tokens'] == 192 for r in requests)
    outage = next(r for r in requests if r['id'] == PLAN['outage']['request'])
    assert len(outage['expected']) > PLAN['outage']['at_step']
    with pytest.raises(ValueError, match='canonical result changed'):
        execution.workload({**PLAN, 'workload': {**PLAN['workload'], 'sha256': '0' * 64}})


def test_inventory_is_pinned_to_the_granite_checkpoint_and_covers_every_tensor():
    tensors, artifacts = execution.inventory(PLAN)
    assert len(tensors['tensors']) == 362
    assert sum(s['end'] - s['begin'] for s in tensors['tensors'].values()) == 2 * artifacts['parameters']
    with pytest.raises(ValueError, match='inventory changed'):
        execution.inventory({**PLAN, 'model': {**PLAN['model'], 'inventory_sha256': '0' * 64}})


def test_phase_jobs_hide_expected_tokens_and_inject_the_declared_outage():
    agreement = execution.job(PLAN, 'agreement')
    assert len(agreement['requests']) == 230 and agreement['fail'] is None and agreement['threads'] == 8
    assert not any('expected' in r for r in agreement['requests'])
    outage = execution.job(PLAN, 'outage')
    assert [r['id'] for r in outage['requests']] == [PLAN['outage']['request']]
    assert outage['fail'] == {'rank': 2, 'at': 40}
    resume = execution.job(PLAN, 'resume', [1, 2, 3])
    assert resume['requests'][0]['committed'] == [1, 2, 3] and resume['fail'] is None


def passing_evidence():
    from neuroshard.evolution.sharded import granite

    tensors, _ = execution.inventory(PLAN)
    requests = execution.workload(PLAN)
    fetches = []
    for rank in range(3):
        owned = sorted(n for n in tensors['tensors'] if granite.owner(n, PLAN['boundaries']) == rank)
        fetches.append({'tensors': owned, 'requests': 40, 'fetch_seconds': 30.0,
                        'fetched_bytes': sum(tensors['tensors'][n]['end'] - tensors['tensors'][n]['begin'] for n in owned)})
    owner = {'completed': True, 'peak_rss_bytes': 4 * 1024 ** 3, 'sent_bytes': 1000}
    outage_request = next(r for r in requests if r['id'] == PLAN['outage']['request'])
    phases = {
        'agreement': [{**owner, 'outputs': [{'id': r['id'], 'token_ids': r['expected'], 'seconds': 1.0}
                                            for r in requests]}, owner, owner],
        'outage': [{**owner, 'completed': False, 'inflight': {'token_ids': outage_request['expected'][:39]}},
                   {**owner, 'completed': False}, None],
        'resume': [{**owner, 'outputs': [{'id': outage_request['id'], 'token_ids': outage_request['expected'],
                                          'seconds': 1.0}]}, owner, owner],
    }
    return fetches, phases


def test_assessment_passes_only_with_exact_tokens_ownership_memory_and_recovery():
    fetches, phases = passing_evidence()
    report = execution.assess(PLAN, fetches, phases)
    assert report['passed'] and report['generations'] == 230 and report['outage']['committed_before_loss'] == 39
    wrong = copy.deepcopy(phases)
    wrong['agreement'][0]['outputs'][5]['token_ids'][2] += 1
    report = execution.assess(PLAN, fetches, wrong)
    assert not report['checks']['agreement'] and report['mismatches'][0]['first_difference'] == 2
    heavy = copy.deepcopy(phases)
    heavy['resume'][1]['peak_rss_bytes'] = PLAN['canonical_peak_rss_bytes']
    assert not execution.assess(PLAN, fetches, heavy)['checks']['memory']
    unrecovered = copy.deepcopy(phases)
    unrecovered['resume'][0]['outputs'][0]['token_ids'] = unrecovered['resume'][0]['outputs'][0]['token_ids'][:-2]
    assert not execution.assess(PLAN, fetches, unrecovered)['checks']['recovery']
    greedy = copy.deepcopy(fetches)
    greedy[1]['tensors'] = greedy[1]['tensors'] + ['model.embed_tokens.weight']
    assert not execution.assess(PLAN, greedy, phases)['checks']['ownership']
    survived = copy.deepcopy(phases)
    survived['outage'][2] = {'completed': True}
    assert not execution.assess(PLAN, fetches, survived)['checks']['outage_injected']


def test_the_ring_binds_to_the_default_route_interface(tmp_path):
    routes = tmp_path / 'route'
    routes.write_text('Iface\tDestination\tGateway\n docker0\t000011AC\t00000000\nens5\t00000000\t01401FAC\n')
    assert execution.default_interface(routes) == 'ens5'
    routes.write_text('Iface\tDestination\tGateway\nlo\t0000007F\t00000000\n')
    with pytest.raises(ValueError, match='default route'):
        execution.default_interface(routes)


def test_owner_profile_is_bounded_and_the_freeze_covers_every_imported_source():
    plan = PLAN
    for name, digest in plan['contracts'].items():
        assert sha256(ROOT / name) == digest
    assert set(plan['contracts']) <= set(plan['sources'])
    cloud = cloud_module()
    resources = cloud.resources(execution.PROFILE)
    assert resources['instance_type'] == 'r7i.4xlarge' and not resources['gpu']
    assert 3 * resources['planning_cap_usd'] <= 24
    assert sum(plan['phase_seconds'].values()) + resources['setup_seconds'] + resources['copy_seconds'] <= resources['hours'] * 3600
    assert cloud.GRANITE_PROFILES[execution.PROFILE][0] == 'granite_shard_execution'
    probe = ('import os, sys; import neuroshard.evolution.granite_shard_execution, '
             'neuroshard.evolution.sharded.granite, neuroshard.evolution.sharded.granite_pipeline; '
             'root = os.path.abspath("src"); '
             'print("\\n".join(sorted(os.path.relpath(x.__file__) for x in list(sys.modules.values()) '
             'if getattr(x, "__file__", None) and os.path.abspath(x.__file__).startswith(root))))')
    imported = subprocess.check_output([sys.executable, '-c', probe], cwd=ROOT, text=True,
                                       env={'PYTHONPATH': str(ROOT / 'src')}).split()
    assert set(imported) <= set(plan['sources']), set(imported) - set(plan['sources'])
    assert not plan['training_authorized'] and not plan['gpu_launch_authorized'] and plan['attempts'] == 1


def test_prepare_fetches_the_config_and_only_owned_ranges(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import hashlib
    import json
    from transformers import GraniteForCausalLM
    from neuroshard.evolution.sharded import granite

    from test_granite_partition import BOUNDARIES, range_server, tiny_checkpoint

    split = tmp_path / 'split'
    GraniteForCausalLM.from_pretrained(tiny_checkpoint(tmp_path / 'one'), dtype=torch.bfloat16).save_pretrained(
        split, max_shard_size='100KB')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in split.glob('*.safetensors')}
    tensors = granite.tensor_inventory(split, files)
    raw = (split / 'config.json').read_bytes()
    artifacts = {'files': {'config.json': {'digest': hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()}}}
    monkeypatch.setattr(execution, 'inventory', lambda plan: (tensors, artifacts))
    server, requested = range_server(split)

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return raw

    import urllib.request
    original = urllib.request.urlopen

    def urlopen(request, timeout=None):
        if isinstance(request, str) and request.endswith('/config.json'):
            return Response(request)
        return original(request, timeout=timeout)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    try:
        receipt = execution.prepare({**PLAN, 'boundaries': list(BOUNDARIES)}, 2, tmp_path / 'store',
                                    base=f'http://127.0.0.1:{server.server_address[1]}')
    finally:
        server.shutdown()
    owned = [n for n in tensors['tensors'] if granite.owner(n, BOUNDARIES) == 2]
    assert receipt['tensors'] == sorted(owned)
    assert json.loads((tmp_path / 'store/config/config.json').read_text())['model_type'] == 'granite'
    assert all(granite.owner(n, BOUNDARIES) == 2 for n, spec in tensors['tensors'].items()
               if any(path == f"/{spec['file']}" and b <= spec['begin'] and spec['end'] <= e
                      for path, b, e in requested))
