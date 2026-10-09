"""``neuroshard assistant``: host a shard of the grown assistant, chat through the network, or run the seed."""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import signal
import threading


def register(subcommands, root):
    parser = subcommands.add_parser('assistant', help='Host a shard of the grown assistant, or chat through the network')
    actions = parser.add_subparsers(dest='assistant_action', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--network', type=Path, help='Network descriptor; defaults to the bundled assistant testnet')
    common.add_argument('--home', type=lambda p: Path(p).expanduser(), default=root / 'assistant',
                        help='Keys, fetched shards and logs')
    status = actions.add_parser('status', parents=[common], help='The served version, its owners and a job outcome')
    status.add_argument('--job', help='Show the outcome of this job')
    host = actions.add_parser('host', parents=[common],
                              help='Host one shard: fetch it, bond, publish your endpoint and serve paid jobs')
    host.add_argument('--shard', type=int, required=True)
    host.add_argument('--endpoint', required=True, help='host:port where users reach this machine')
    host.add_argument('--listen', default='0.0.0.0', help='Local address to listen on')
    host.add_argument('--threads', type=int)
    chat = actions.add_parser('chat', parents=[common],
                              help='Chat with the assistant through the network; stage 0 runs on this machine')
    chat.add_argument('--sample', type=int, default=0, help='Sample workspace number')
    chat.add_argument('--world', type=Path, help='Your own workspace JSON: documents')
    chat.add_argument('--budget', type=int, default=16384, help='Token positions to buy for this conversation')
    chat.add_argument('--price', default='1', help='NEURO escrowed for the budget; the unused part is refunded')
    chat.add_argument('--threads', type=int)
    seed = actions.add_parser('seed', parents=[common], help="Run the network's seed: first validator, RPC and faucet")
    seed.add_argument('--public-host', required=True)


def atoms(value):
    from neuroshard.assistant.network import NEURO

    try:
        amount = Decimal(value) * NEURO
    except InvalidOperation:
        raise ValueError('price must be a number of NEURO') from None
    if amount <= 0 or amount != amount.to_integral_value():
        raise ValueError('price must be a positive amount with at most six decimals')
    return int(amount)


def workspace(sample, path):
    from neuroshard.evolution import assistant_workflow_data as data

    if path is not None:
        return json.loads(Path(path).read_text()), []
    cases = data.cases('development')
    if not 0 <= sample < len(cases):
        raise ValueError(f'sample must be 0 to {len(cases) - 1}')
    return cases[sample]['world'], [turn['user'] for turn in cases[sample]['turns']]


def show(world, suggestions):
    print(f"Workspace: {len(world['documents'])} documents, on this machine only.")
    for document in sorted(world['documents'], key=lambda d: (d['project'], d['revision'])):
        print(f"  {document['id']}  {document['title']}, revision {document['revision']}, {document['status']}")
    for index, text in enumerate(suggestions):
        print(('Try: ' if index == 0 else 'Then: ') + text)


def render(result, before):
    print(f"assistant ({result['seconds']:.0f} s): {result['final_text'] or '(no reply)'}")
    for project, draft in result['snapshot']['drafts'].items():
        if before.get(project) != draft:
            fields = '; '.join(f"{key} {', '.join(value) if isinstance(value, list) else value}"
                               for key, value in sorted(draft.items()) if key not in ('project', 'revision'))
            print(f'  draft for {project}: {fields}')
    if result['failure']:
        print(f"  stopped: {result['failure']}")


def run(args):
    from neuroshard.assistant import network

    descriptor = network.descriptor(args.network)
    if args.assistant_action == 'seed':
        config, _, _ = network.seed(descriptor, args.home, args.public_host)
        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        try:
            stopped.wait()
        except KeyboardInterrupt:
            pass
        finally:
            from neuroshard.inference import optimistic_network

            optimistic_network.stop(config)
        return
    chain = network.Chain(descriptor['rpc'], descriptor['chain_id'])
    if args.assistant_action == 'status':
        state = chain.state()
        served = network.version(descriptor)[0]
        print(f"{state['chain_id']} at block {state['height']}, serving {served.name}"
              f"{'' if state['model_root'] == served.root else ' (the ledger serves another version)'}")
        for shard in range(1, served.world):
            hosts = [o for o in state['owners'].values() if o['shard'] == shard and o['status'] == 'active']
            print(f"  shard {shard}: {len(hosts)} active owner(s) "
                  f"{', '.join(o.get('endpoint') or 'no endpoint' for o in hosts)}")
        print(f"  open jobs: {len(state['jobs'])}")
        if args.job:
            print(json.dumps(state['results'].get(args.job) or state['jobs'].get(args.job) or 'unknown job', indent=2))
        return
    if args.assistant_action == 'host':
        served, stage = network.fetch_stage(descriptor, args.shard, args.home)
        owner = network.Owner(served, stage, args.shard, chain, network.Account(args.home / 'account.key'),
                              network.signing_key(args.home / f'log-{args.shard}.key'), args.home, args.threads)
        owner.register(args.endpoint, descriptor.get('faucet'))
        owner.run(args.listen, int(args.endpoint.rsplit(':', 1)[1]))
        return
    if args.budget < 1:
        raise ValueError('budget must be positive')
    price = atoms(args.price)
    served, stage = network.fetch_stage(descriptor, 0, args.home)
    world, suggestions = workspace(args.sample, args.world)
    show(world, suggestions)
    conversation = network.Conversation(served, stage, chain, network.Account(args.home / 'account.key'), world,
                                        args.budget, price, descriptor.get('faucet'), threads=args.threads)
    print(f"Job {conversation.job_id[:16]}: shards served by "
          f"{', '.join(endpoint for _, endpoint in conversation.owners)}. Type a request; /quit ends the job.")
    drafts = {}
    try:
        while True:
            try:
                text = input('you> ').strip()
            except EOFError:
                break
            if text in ('/quit', '/exit'):
                break
            if text:
                result = conversation.say(text)
                render(result, drafts)
                drafts = dict(result['snapshot']['drafts'])
                if result['failure']:
                    break
    finally:
        conversation.close()
    print(f'The owners commit their logs now; the job settles after its challenge window. '
          f'Check it with: neuroshard assistant status --job {conversation.job_id}')
