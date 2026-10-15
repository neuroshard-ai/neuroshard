"""Reproduce the general-route version's accepted conversations through paid, separately hosted shards.

The candidate (``a4-general``) serves A3's drafting and scheduling units beside the general route's units:
documents (L9) and chat (C1), both additions on the unchanged parent, chosen per turn by a router.
This check is deployment equivalence for that whole system:

- every confirmation7 conversation (132 documents, 96 chat) must reproduce its accepted run token for
  token. That covers each generation's input and output tokens, every tool call and result, the final
  reply, and for documents the workspace snapshot. The router must also choose the route each
  conversation was accepted under.
- an A3 regression set (all 48 cross-capability conversations, the first 24 drafting and the first 24
  scheduling) must reproduce A3's accepted conversations exactly, as in A3's own release check. That
  shows the added routes and router leave A3's behaviour unchanged.

Nothing is trained or tuned and no new evaluation set opens. The first mismatch stops the attempt.
"""

import argparse
import gzip
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

from neuroshard.assistant import release_check as a3

CONTRACT = 'config/experiments/assistant-network-release-check-general.json'
GENERATION_FIELDS = ('input_token_ids', 'token_ids', 'text', 'terminated')


def general_projection(route, final_text, calls, generations, snapshot=None):
    """Every generated token, tool call and result, the reply and (documents) the workspace; no timing."""
    return {'route': route, 'final_text': final_text, 'calls': calls, 'snapshot': snapshot,
            'generations': [{k: g.get(k) for k in GENERATION_FIELDS} for g in generations if g.get('executed', True)]}


def accepted(row, route):
    """The projection of a recorded confirmation7 episode (``assistant_general`` rows)."""
    if route == 'documents':
        last = row['rounds'][-1]
        return general_projection(route, last['final_text'], row['calls'], row['generations'], last['snapshot'])
    return general_projection(route, row['final_text'] if row['terminated'] else '', row['calls'], row['generations'])


def checked_workload(plan):
    from neuroshard.evolution import assistant_chat_data as chat
    from neuroshard.evolution import assistant_documents_data as docs
    from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

    for name, digest in plan['contracts'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'release contract bytes changed: {name}')
    general = plan['general']
    episodes = json.loads(gzip.decompress((ROOT / general['episodes']).read_bytes()))['episodes']
    work = []
    for route, key, module in (('documents', general['documents'], docs), ('chat', general['chat'], chat)):
        cases, rows = module.cases(general['split']), episodes[key]
        if [c['id'] for c in cases] != [r['id'] for r in rows] or len(cases) != general['cases'][route]:
            raise ValueError(f'changed {route} workload')
        work += [(route, case, accepted(row, route)) for case, row in zip(cases, rows)]
    regression = a3.checked_workload(read(ROOT / a3.CONTRACT))
    keep = plan['a3_regression']
    counts, chosen = {name: 0 for name in keep}, []
    for name, case, expected in regression:
        if counts.get(name, keep.get(name, 0)) < keep.get(name, 0):
            counts[name] += 1
            chosen.append(('a3-' + name, case, a3.projection(expected)))
    # Every route is exercised before the long reproduction: two of each first.
    groups = {}
    for item in work + chosen:
        groups.setdefault(item[0], []).append(item)
    smoke = [groups[name][index] for index in range(2) for name in sorted(groups)]
    rest = [item for name in sorted(groups) for item in groups[name][2:]]
    return smoke + rest


def freeze(plan):
    from neuroshard.evolution import assistant_growth_cohort3_eval as runtime
    from neuroshard.evolution.modular_reference_execution import ROOT, sha256

    for name, digest in plan['sources'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'release execution source changed: {name}')
    names = [CONTRACT, *plan['sources'], *plan['contracts']]
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *names], cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    if any(name not in tracked for name in names):
        raise ValueError('release execution requires a committed contract and sources')
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'contract_sha256': sha256(ROOT / CONTRACT), **runtime.runtime(plan)}


def converse(network, served, stage, chain, account, plan, descriptor, set_name, case):
    """One paid conversation through the network; returns its projection in the set's format."""
    conversation = network.Conversation(served, stage, chain, account, case['world'] if 'world' in case else
                                        {'documents': []}, budget=plan['positions'], price=plan['price'],
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
    if set_name.startswith('a3-'):
        return conversation.job_id, a3.projection(row)
    last = row['rounds'][-1]
    return conversation.job_id, general_projection(last['route'], last['final_text'], row['calls'], row['generations'],
                                                   last['snapshot'] if set_name == 'documents' else None)


def client(plan, descriptor, home, binding, stop=None):
    from neuroshard.assistant import network
    from neuroshard.evolution.modular_reference_execution import identity

    workload = checked_workload(plan)
    if (home / 'result.json').exists():
        raise ValueError('this attempt already ended; it may not be resumed')
    account = network.Account(home / 'account.key')
    chain = network.Chain(network.endpoints(descriptor), descriptor['chain_id'])
    served, stage = network.fetch_stage(descriptor, 0, home / 'holdings')
    started, results, failed = time.monotonic(), [], None
    a3.atomic(home / 'binding.json', binding)
    try:
        for name, case, gold in workload:
            if time.monotonic() - started >= plan['worker_seconds']:
                raise TimeoutError('release reproduction worker budget exhausted')
            if stop is not None and stop.is_set():
                raise InterruptedError('release reproduction stopped by its supervisor')
            a3.atomic(home / 'progress.json', {'completed': len(results), 'total': len(workload), 'set': name,
                                              'case': case['id'], 'unix': time.time()})
            job_id, actual = converse(network, served, stage, chain, account, plan, descriptor, name, case)
            matched = actual == gold
            outcome = {'set': name, 'id': case['id'], 'matched': matched, 'job_id': job_id,
                       'actual_sha256': identity(actual), 'expected_sha256': identity(gold)}
            a3.atomic(home / f"{case['id']}.json", {'comparison': outcome, 'actual': actual, 'expected': gold})
            results.append(outcome)
            a3.atomic(home / 'progress.json', {'completed': len(results), 'total': len(workload),
                                              'matched': sum(r['matched'] for r in results), 'unix': time.time()})
            if not matched:
                raise ValueError(f"deployment differs from accepted conversation: {name}/{case['id']}")
    except Exception as error:
        failed = f'{type(error).__name__}: {error}'
    final = {'binding': binding, 'attempt': plan.get('attempt', 1), 'cases': len(workload), 'completed': len(results),
             'results': results, 'passed': failed is None and len(results) == len(workload), 'error': failed,
             'wall_seconds': time.monotonic() - started, 'admission_evidence': False, 'training': False,
             'new_final_opened': False, 'independent_operation': False}
    a3.atomic(home / 'result.json', final)
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
        owner = network.Owner(served, stage, args.shard, network.Chain(network.endpoints(descriptor),
                                                                     descriptor['chain_id']),
                              network.Account(args.home / 'account.key'),
                              network.signing_key(args.home / f'log-{args.shard}.key'), args.home, plan['threads'])
        owner.register(f"{args.host}:{plan['ports']['owner']}", descriptor['faucet'])
        owner.run('0.0.0.0', plan['ports']['owner'], stopped)


if __name__ == '__main__':
    main()
