"""An owner's per-job timing file: written beside, never inside, the signed audit log, and never fatal."""
import json
from types import SimpleNamespace

from neuroshard.assistant import network


def owner(tmp_path):
    instance = network.Owner.__new__(network.Owner)
    instance.home, instance.rank, instance.threads = tmp_path, 2, 8
    return instance


def test_timing_is_written_beside_the_logs(tmp_path):
    served = {'steps': 40, 'busy_seconds': 3.0, 'timing': {'partition': 3.0, 'passes': 40, 'receiving_tensor': 1.5}}
    owner(tmp_path).save_timing('ab' * 32, 7.25, served, None)
    record = json.loads((tmp_path / 'timing' / f"{'ab' * 32}.json").read_text())
    assert record['serving_seconds'] == 7.25 and record['timing']['partition'] == 3.0 and record['rank'] == 2
    assert record['ended_early'] is None and record['threads'] == 8
    assert not (tmp_path / 'logs').exists(), 'the signed audit log directory is untouched'


def test_a_job_that_ended_early_keeps_its_total(tmp_path):
    owner(tmp_path).save_timing('cd' * 32, 2.0, None, 'ConnectionError: the peer closed the serving link')
    record = json.loads((tmp_path / 'timing' / f"{'cd' * 32}.json").read_text())
    assert record['serving_seconds'] == 2.0 and record['timing'] is None and 'peer closed' in record['ended_early']


def test_a_timing_write_failure_never_affects_serving(tmp_path):
    instance = owner(tmp_path)
    (tmp_path / 'timing').write_text('not a directory')  # makes the write fail
    instance.save_timing('ef' * 32, 1.0, SimpleNamespace(), None)  # must not raise
