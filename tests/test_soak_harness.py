"""The soak harness records what actually happened (pilot-1 soak review, October 19, 2026)."""
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('soak', ROOT / 'scripts' / 'soak.py')
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


def events(out):
    return [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]


def test_a_failed_client_is_a_failure_and_keeps_its_output(tmp_path):
    process = subprocess.CompletedProcess(['client'], 1, 'Job 1234567890abcdef:\nassistant (12 s): (no reply)\n'
                                          '  stopped: serving failed\n', 'NeuroShard: settlement failed\n')
    pending = soak.Pending(tmp_path / 'pending.json')
    with patch.object(soak.subprocess, 'run', return_value=process):
        record = soak.conversation(tmp_path, 1, 'q', ('number', '9'), ['client'], tmp_path / 'c', pending)
    assert record['transport_ok'] is False and record['correct'] is False
    assert 'settlement failed' in Path(record['output_file']).read_text()
    assert any(e['kind'] == 'incident' and e['what'] == 'conversation failed' for e in events(tmp_path))


def test_a_good_reply_is_checked_for_the_known_fact_and_followed_to_settlement(tmp_path):
    job = 'ab' * 32
    text = (f'Job {job[:16]}: shards served by a, b.\nyou> assistant (40 s, documents): The baseline quantity is 58 '
            f'units.\nyou> Check it with: neuroshard assistant status --job {job}\n')
    pending = soak.Pending(tmp_path / 'pending.json')
    with patch.object(soak.subprocess, 'run', return_value=subprocess.CompletedProcess(['c'], 0, text, '')):
        record = soak.conversation(tmp_path, 2, 'q', ('number', '61'), ['client'], tmp_path / 'c', pending)
    assert record['transport_ok'] is True and record['correct'] is False and record['job'] == job
    assert record['fact'] == 'mismatch' and pending.items[0]['job'] == job

    class Network:
        @staticmethod
        def chain(*_):
            class Chain:
                def state(self):
                    return {'jobs': {}, 'results': {job: {'status': 'settled', 'paid': [1, 1], 'refunded': 8}}}
            return Chain()
    soak.settle(tmp_path, {}, Network, pending, wait=0)
    settled = [e for e in events(tmp_path) if e['kind'] == 'settlement']
    assert settled[0]['status'] == 'settled' and settled[0]['paid'] == [1, 1] and not pending
    assert soak.Pending(tmp_path / 'pending.json').items == []  # persisted


def test_a_failed_fault_command_is_an_incident_and_restoration_is_verified(tmp_path):
    with patch.object(soak, 'ssh', return_value=(255, 'connection refused')), patch.object(soak.time, 'sleep'), \
            patch.object(soak, 'service_state', return_value='active'):
        soak.fault(tmp_path, 'stop-owner-2b', 300)
    seen = events(tmp_path)
    assert any(e['kind'] == 'incident' and e['what'] == 'fault injection not observed' for e in seen)
    assert [e['phase'] for e in seen if e['kind'] == 'fault'] == ['inject', 'backup-cancelled', 'recovered']


def test_an_injection_timeout_still_restores(tmp_path):
    calls, states = [], iter(['active', 'inactive', 'active', 'active', 'active'])

    def ssh(host, command, timeout=60):
        calls.append(command)
        if 'systemd-run' in command:
            raise subprocess.TimeoutExpired(['ssh'], timeout)
        return 0, ''
    with patch.object(soak, 'ssh', side_effect=ssh), patch.object(soak.time, 'sleep'), \
            patch.object(soak, 'service_state', side_effect=lambda *a: next(states)):
        try:
            soak.fault(tmp_path, 'stop-owner-2b', 300)
        except subprocess.TimeoutExpired:
            pass
    assert 'sudo systemctl start neuroshard-owner' in calls
    assert [e['phase'] for e in events(tmp_path) if e['kind'] == 'fault'] == [
        'restore-attempt', 'backup-cancelled', 'recovered']


def test_a_failed_restoration_keeps_the_backup_timer_and_retries(tmp_path):
    calls = []

    def ssh(host, command, timeout=60):
        calls.append(command)
        return (1, 'Job for neuroshard-owner.service failed') if command.endswith('start neuroshard-owner') else (0, '')
    clock = iter(range(0, 100_000, 50))
    with patch.object(soak, 'ssh', side_effect=ssh), patch.object(soak.time, 'sleep'), \
            patch.object(soak.time, 'time', side_effect=lambda: next(clock)), \
            patch.object(soak, 'service_state', return_value='inactive'):
        soak.fault(tmp_path, 'stop-owner-2b', 300)
    seen = events(tmp_path)
    attempts = [e for e in seen if e.get('phase') == 'restore-attempt']
    assert len(attempts) >= 2 and all(e['exit'] == 1 for e in attempts)  # failures kept, and retried
    assert not any('stop soak-restore' in c for c in calls)  # the backup timer is never cancelled
    assert any(e['kind'] == 'incident' and e['what'] == 'fault not recovered' for e in seen)


def test_a_restarted_validator_counts_as_recovered_only_when_signing_again(tmp_path):
    status = json.dumps({'result': {'validator_info': {'address': 'AB'},
                                    'sync_info': {'catching_up': False, 'latest_block_height': '100'}}})
    with patch.object(soak, 'ssh', return_value=(0, status)), \
            patch.object(soak, 'rpc', return_value={'signed_header': {'commit': {'signatures': [
                {'validator_address': 'CD', 'block_id_flag': 2}]}}}):
        assert soak.validator_recovered('h', 'http://rpc')[0] is False
    with patch.object(soak, 'ssh', return_value=(0, status)), \
            patch.object(soak, 'rpc', return_value={'signed_header': {'commit': {'signatures': [
                {'validator_address': 'AB', 'block_id_flag': 2}]}}}):
        assert soak.validator_recovered('h', 'http://rpc')[0] is True


def test_a_fault_that_never_recovers_is_an_incident(tmp_path):
    clock = iter(range(0, 100_000, 100))
    with patch.object(soak, 'ssh', return_value=(0, '')), patch.object(soak.time, 'sleep'), \
            patch.object(soak.time, 'time', side_effect=lambda: next(clock)), \
            patch.object(soak, 'service_state', return_value='inactive'):
        soak.fault(tmp_path, 'stop-validator', 300)
    assert any(e['kind'] == 'incident' and e['what'] == 'fault not recovered' for e in events(tmp_path))


def test_probes_compare_block_ids_and_app_hashes_from_every_rpc(tmp_path):
    def rpc(address, method, timeout=8):
        if method == 'status':
            return {'sync_info': {'latest_block_height': '102'}}
        app = 'BAD' if 'liar' in address else 'GOOD'
        return {'signed_header': {'commit': {'block_id': {'hash': 'H'}, 'signatures': [{'block_id_flag': 2}] * 5},
                                  'header': {'app_hash': app}}}

    class Network:
        @staticmethod
        def chain(*_):
            raise OSError('offline')
    with patch.object(soak, 'rpc', side_effect=rpc):
        record = soak.probe(tmp_path, {'rpc': 'http://good:26657', 'rpcs': ['http://liar:26657']}, Network)
    assert record['agree'] is False and record['agree_all'] is False and record['height'] == 100
    assert record['rpcs_compared'] == 2 and record['app_hashes'] == ['BAD', 'GOOD']
    assert any(e['kind'] == 'incident' and e['what'] == 'rpcs disagree' for e in events(tmp_path))


def test_summary_reports_runs_separately(tmp_path):
    lines = [{'unix': 1, 'kind': 'start'}, {'unix': 2, 'kind': 'conversation', 'ok': True, 'seconds': 10},
             {'unix': 3, 'kind': 'start'}, {'unix': 4, 'kind': 'conversation', 'ok': False, 'seconds': 10},
             {'unix': 5, 'kind': 'start', 'run': 'r2', 'measurement': 2},
             {'unix': 6, 'kind': 'conversation', 'run': 'r2', 'transport_ok': True, 'correct': True, 'seconds': 9}]
    (tmp_path / 'events.jsonl').write_text('\n'.join(json.dumps(l) for l in lines) + '\n')
    with patch('sys.stdout') as stdout:
        soak.summary(type('A', (), {'out': str(tmp_path)}))
    report = json.loads(''.join(c.args[0] for c in stdout.write.call_args_list))
    assert len(report) == 3 and report['r2']['measurement'] == 2 and report['r2']['correct'] == 1
    assert sorted(r['transport_ok'] for r in report.values()) == [0, 1, 1]


def test_one_answering_rpc_is_not_all_agreeing(tmp_path):
    def rpc(address, method, timeout=8):
        if 'down' in address:
            raise OSError('refused')
        if method == 'status':
            return {'sync_info': {'latest_block_height': '102'}}
        return {'signed_header': {'commit': {'block_id': {'hash': 'H'}, 'signatures': [{'block_id_flag': 2}] * 5},
                                  'header': {'app_hash': 'A'}}}

    class Network:
        @staticmethod
        def chain(*_):
            raise OSError('offline')
    descriptor = {'rpc': 'http://up:1', 'rpcs': [f'http://down{i}:1' for i in range(4)]}
    with patch.object(soak, 'rpc', side_effect=rpc):
        record = soak.probe(tmp_path, descriptor, Network)
    assert record['agree'] is True and record['agree_all'] is False
    assert record['rpcs_compared'] == 1 and record['rpcs_configured'] == 5


def test_contradicting_or_partial_answers_are_not_correct():
    assert soak.score('The baseline is 58 units, not 61.', 'number', '61') == 'ambiguous'
    assert soak.score('The baseline quantity is 61 units. (Sources: doc-88b36697f8)', 'number', '61') == 'match'
    assert soak.score('Revision 2 of Cedar 6581 has 61 units.', 'number', '61') == 'match'
    assert soak.score('It is 58 units.', 'number', '61') == 'mismatch'
    assert soak.score('It starts on May 14, 2026.', 'date', '2027-05-14') == 'mismatch'  # wrong year
    assert soak.score('It starts on **2027-05-14**.', 'date', '2027-05-14') == 'match'
    assert soak.score('I could not find that.', 'number', '61') == 'none'


def test_a_rejected_connection_is_reconciled_through_its_account(tmp_path):
    """The CLI never prints a job ID when the owner closes the link; the chain still names the funded job."""
    key, job = '02' + 'aa' * 32, 'cd' * 32
    states = iter([{'jobs': {}, 'results': {}}, {'jobs': {job: {'user': key}}, 'results': {}}])

    class Chain:
        def state(self):
            return next(states)
    failed = subprocess.CompletedProcess(['c'], 1, 'Requesting test NEURO from the faucet\n',
                                         'NeuroShard: the peer closed the serving link\n')
    pending = soak.Pending(tmp_path / 'pending.json')
    with patch.object(soak.subprocess, 'run', return_value=failed):
        record = soak.conversation(tmp_path, 7, 'q', ('number', '9'), ['c'], tmp_path / 'h', pending, key, Chain())
    assert record['job'] is None and record['opened'] == [job]
    assert pending.items[0]['job'] == job and pending.items[0]['transport_ok'] is False
    assert soak.Pending(tmp_path / 'pending.json').items[0]['job'] == job  # survives a restart


def test_a_long_status_reply_is_parsed_in_full():
    status = json.dumps({'result': {'node_info': {'other': 'x' * 2000}, 'validator_info': {'address': 'AB'},
                                    'sync_info': {'catching_up': False, 'latest_block_height': '100'}}})
    with patch.object(soak.subprocess, 'run', return_value=subprocess.CompletedProcess(['ssh'], 0, status, '')), \
            patch.object(soak, 'rpc', return_value={'signed_header': {'commit': {'signatures': [
                {'validator_address': 'AB', 'block_id_flag': 2}]}}}):
        assert soak.validator_recovered('h', 'http://rpc') == (True, 'signing again at 98')
