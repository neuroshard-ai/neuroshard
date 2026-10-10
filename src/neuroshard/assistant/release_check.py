"""Reproduce the already-opened A3 confirmation through paid, separately hosted shards.

This checks deployment equivalence. It does not train, open a new evaluation set,
or give independent-operation credit. The contract and every execution source must
be committed before any role starts.
"""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

CONTRACT = 'config/experiments/assistant-network-release-check.json'
GENERATION_FIELDS = ('input_token_ids', 'token_ids', 'text', 'terminated', 'executed', 'request_sha256')
ROUND_FIELDS = ('route', 'completed', 'final_text', 'failure', 'snapshot', 'call_count', 'generation_count')


def projection(row):
    """Ignore timing and memory; preserve every generated token and observable decision."""
    return {'id': row['id'], 'a2_selected': row['a2_selected'], 'calls': row['calls'],
            'rounds': [{k: item[k] for k in ROUND_FIELDS} for item in row['rounds']],
            'generations': [{k: item.get(k, True if k == 'executed' else None) for k in GENERATION_FIELDS}
                            for item in row['generations']]}


def checked_workload(plan):
    from neuroshard.evolution import assistant_growth_cohort3_confirm as confirmation
    from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

    for name, digest in plan['contracts'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'release contract bytes changed: {name}')
    execution = read(ROOT / confirmation.EXECUTION)
    cases = {**confirmation.opened(execution, 'drafting'), **confirmation.opened(execution, 'calendar')}
    pairs = {}
    for name, spec in plan['sets'].items():
        rows = read(ROOT / spec['result'])['reply']['episodes'][name]
        if len(rows) != spec['cases'] or [r['id'] for r in rows] != [c['id'] for c in cases[name]]:
            raise ValueError(f'changed {name} workload')
        pairs[name] = list(zip(cases[name], rows))
    # All routes are exercised before the long reproduction starts.
    smoke = [(name, *pairs[name][index]) for index in range(2) for name in pairs]
    return smoke + [(name, *pair) for name in pairs for pair in pairs[name][2:]]


def freeze(plan):
    from neuroshard.evolution import assistant_growth_cohort3_eval as runtime
    from neuroshard.evolution.modular_reference_execution import ROOT, sha256

    for name, digest in plan['sources'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'release execution source changed: {name}')
    names = [CONTRACT, *plan['sources'], *plan['contracts']]
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *names], cwd=ROOT,
                   stdout=subprocess.DEVNULL, check=True)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    if any(name not in tracked for name in names):
        raise ValueError('release execution requires a committed contract and sources')
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'contract_sha256': sha256(ROOT / CONTRACT), **runtime.runtime(plan)}


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def client(plan, descriptor, home, binding, stop=None):
    from neuroshard.assistant import network
    from neuroshard.evolution.modular_reference_execution import identity

    workload = checked_workload(plan)
    if (home / 'result.json').exists():
        raise ValueError('this attempt already ended; it may not be resumed')
    account = network.Account(home / 'account.key')
    chain = network.Chain(descriptor['rpc'], descriptor['chain_id'])
    served, stage = network.fetch_stage(descriptor, 0, home / 'holdings')
    started, results, failed = time.monotonic(), [], None
    atomic(home / 'binding.json', binding)
    try:
        for name, case, expected in workload:
            if time.monotonic() - started >= plan['worker_seconds']:
                raise TimeoutError('release reproduction worker budget exhausted')
            if stop is not None and stop.is_set():
                raise InterruptedError('release reproduction stopped by its supervisor')
            atomic(home / 'progress.json', {'completed': len(results), 'total': len(workload), 'set': name,
                                           'case': case['id'], 'unix': time.time()})
            conversation = network.Conversation(served, stage, chain, account, case['world'],
                                                budget=plan['positions'], price=plan['price'],
                                                faucet_url=descriptor['faucet'], threads=plan['threads'])
            row = {'id': case['id'], 'calls': [], 'rounds': [], 'generations': []}
            try:
                for turn in case['turns']:
                    reply = conversation.say(turn['user'])
                    row['generations'].extend(reply['responses'])
                    row['rounds'].append({**reply, 'call_count': len(conversation.session.calls),
                                          'generation_count': len(row['generations'])})
                    if reply['failure']:
                        break
                row.update(calls=conversation.session.calls, a2_selected='arm' if conversation.arm else 'parent')
            finally:
                conversation.close()
            actual, gold = projection(row), projection(expected)
            matched = actual == gold
            outcome = {'set': name, 'id': case['id'], 'matched': matched, 'job_id': conversation.job_id,
                       'actual_sha256': identity(actual), 'expected_sha256': identity(gold)}
            # Keep the complete measured transcript locally for audit and diagnosis.
            atomic(home / f"{case['id']}.json", {'comparison': outcome, 'actual': actual})
            results.append(outcome)
            atomic(home / 'progress.json', {'completed': len(results), 'total': len(workload),
                                           'matched': sum(r['matched'] for r in results), 'unix': time.time()})
            if not matched:
                raise ValueError(f"deployment differs from accepted conversation: {name}/{case['id']}")
    except Exception as error:
        failed = f'{type(error).__name__}: {error}'
    final = {'binding': binding, 'attempt': plan.get('attempt', 1), 'cases': len(workload), 'completed': len(results),
             'results': results,
             'passed': failed is None and len(results) == len(workload), 'error': failed,
             'wall_seconds': time.monotonic() - started, 'admission_evidence': False,
             'training': False, 'new_final_opened': False, 'independent_operation': False}
    atomic(home / 'result.json', final)
    return 0 if final['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('seed', 'owner', 'client'))
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    from neuroshard.evolution import granite_shard_execution
    granite_shard_execution.configure()
    from neuroshard.assistant import network
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    plan = read(ROOT / CONTRACT)
    binding = freeze(plan)
    descriptor = network.descriptor(args.network)
    if network.model_root(descriptor) != plan['model_root']:
        raise ValueError('the release workload serves another model')
    args.home.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    if args.role == 'client':
        raise SystemExit(client(plan, descriptor, args.home, binding, stopped))
    if args.role == 'seed':
        config, _, server = network.seed(descriptor, args.home, args.host,
                                         params={**network.PARAMS, 'challenge_blocks': 5, 'proof_blocks': 5},
                                         p2p_port=plan['ports']['p2p'], faucet_port=plan['ports']['faucet'])
        try:
            stopped.wait(plan['worker_seconds'])
        finally:
            server.shutdown()
            network.chain_network.stop(config)
    else:
        served, stage = network.fetch_stage(descriptor, args.shard, args.home / 'holdings')
        owner = network.Owner(served, stage, args.shard, network.Chain(descriptor['rpc'], descriptor['chain_id']),
                              network.Account(args.home / 'account.key'),
                              network.signing_key(args.home / f'log-{args.shard}.key'), args.home, plan['threads'])
        owner.register(f"{args.host}:{plan['ports']['owner']}", descriptor['faucet'])
        owner.run('0.0.0.0', plan['ports']['owner'], stopped)


if __name__ == '__main__':
    main()
