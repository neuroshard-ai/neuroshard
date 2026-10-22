"""The live admission burst harness (scripts/admission_burst.py): its outcome classes and its guarantee checks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import admission_burst as burst  # noqa: E402

JOB = 'ab' * 32
SERVED = f'Job {JOB[:16]}: shards served by a, b.\nassistant (u1): 9\nCheck it with: neuroshard assistant status --job {JOB}\n'
BUSY = 'Shards 1, 2 busy; nothing funded, waiting\nNeuroShard: the network stayed busy until the deadline; nothing was funded\n'


def test_outcomes_are_classified_from_output_and_chain_together():
    assert burst.classify(SERVED, 0, [JOB], JOB) == 'served'
    assert burst.classify(BUSY, 1, [], None) == 'busy-unfunded'
    assert burst.classify(BUSY, 1, [JOB], None) == 'violation', 'busy reported while the chain shows a job'
    assert burst.classify(SERVED, 0, [JOB, 'cd' * 32], JOB) == 'violation', 'two jobs for one attempt'
    failed = f'Job {JOB[:16]}: x\nNeuroShard: the peer closed the serving link\n'
    assert burst.classify(failed, 1, [JOB], None) == 'funded-failed'
    assert burst.classify('NeuroShard: no route\n', 1, [], None) == 'unfunded-error'


def test_verify_reports_each_guarantee_from_the_run_events(tmp_path, capsys):
    events = [
        {'run': 'r', 'kind': 'start', 'release': 'pilot-2-rc1', 'measurement': 1},
        {'run': 'r', 'kind': 'attempt', 'phase': 'burst-2', 'opened': [JOB], 'outcome': 'served', 'seconds': 80},
        {'run': 'r', 'kind': 'attempt', 'phase': 'burst-2', 'opened': [], 'outcome': 'busy-unfunded', 'seconds': 600},
        {'run': 'r', 'kind': 'settlement', 'job': JOB, 'status': 'settled', 'paid': 10, 'refunded': 0},
    ]
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    burst.verify(type('A', (), {'out': str(tmp_path)}))
    report = json.loads(capsys.readouterr().out)['r']
    assert report['at_most_one_job_per_attempt'] and report['busy_funded_nothing']
    assert report['every_funded_job_terminal'] and report['settled'] == 1
    assert report['outcomes']['served'] == 1 and report['outcomes']['busy-unfunded'] == 1

    events.pop()  # the funded job never reached a terminal record
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    burst.verify(type('A', (), {'out': str(tmp_path)}))
    assert not json.loads(capsys.readouterr().out)['r']['every_funded_job_terminal']
