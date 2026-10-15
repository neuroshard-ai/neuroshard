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
    host.add_argument('--mirror-port', type=int,
                      help='Also serve your verified copies of the released files to peers on this port')
    audit = actions.add_parser('audit', parents=[common], help='Replay committed owner logs and prove faults on chain')
    audit.add_argument('--shard', type=int, action='append', required=True, help='Shard to audit; may be repeated')
    audit.add_argument('--once', action='store_true', help='Audit the currently open jobs and exit')
    audit.add_argument('--interval', type=float, default=10)
    chat = actions.add_parser('chat', parents=[common],
                              help='Chat with the assistant through the network; stage 0 runs on this machine')
    chat.add_argument('--sample', type=int, default=0, help='Sample workspace number')
    chat.add_argument('--world', type=Path, help='Your own documents as JSON: {"documents": [{"id", "title", "content"}]}')
    chat.add_argument('--budget', type=int, default=16384, help='Token positions to buy for this conversation')
    chat.add_argument('--price', default='1', help='NEURO escrowed for the budget; the unused part is refunded')
    chat.add_argument('--threads', type=int)
    govern = actions.add_parser('govern', parents=[common],
                                help='Owner governance of the served version: propose, vote, enact, or list proposals')
    govern.add_argument('govern_action', choices=('list', 'propose', 'propose-validator', 'vote', 'enact', 'handover'))
    govern.add_argument('--consensus-key', help='propose-validator: the CometBFT ed25519 public key, hex')
    govern.add_argument('--power', type=int, help='propose-validator: voting power, 0 removes')
    govern.add_argument('--model-root', help='propose: the version to serve, as its model root')
    govern.add_argument('--quality-root', help='propose: digest of the published release-check evidence')
    govern.add_argument('--proposal', help='vote/enact: the proposal ID')
    govern.add_argument('--shard', type=int, help='vote: the shard whose owner log key votes')
    govern.add_argument('--reject', action='store_true', help='vote: vote no')
    govern.add_argument('--approval-ppm', type=int, default=666_667, help='handover: approving bond share needed')
    govern.add_argument('--vote-blocks', type=int, default=600, help='handover: blocks a proposal stays open')
    grow = actions.add_parser('grow', parents=[common],
                              help='Grow the assistant on the network: open, train, challenge or prove a growth job')
    grow.add_argument('grow_action', choices=('list', 'open', 'train', 'challenge', 'prove'))
    grow.add_argument('--recipe', type=Path, help='open/train/prove: the growth recipe JSON (arm, spec, data, base)')
    grow.add_argument('--job', help='train/challenge/prove: the growth job ID')
    grow.add_argument('--segment', type=int, help='train/challenge: the segment index')
    grow.add_argument('--step', type=int, help='challenge: the step within the segment')
    grow.add_argument('--steps', type=int, help='open: optimizer steps')
    grow.add_argument('--segment-steps', type=int, default=8, help='open: steps per segment')
    grow.add_argument('--reward', default='10', help='open: NEURO escrowed for contributors')
    grow.add_argument('--challenge', help='prove: the challenge ID')
    seed = actions.add_parser('seed', parents=[common], help="Run the network's seed: first validator, RPC and faucet")
    seed.add_argument('--public-host', required=True)
    seed.add_argument('--audit-shards', type=int, nargs='+', default=[1, 2],
                      help='Shard holdings used to judge fraud proofs; defaults to both network owner shards')


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
    for document in sorted(world['documents'], key=lambda d: (d.get('project', ''), d.get('revision', 0), d['id'])):
        if 'revision' in document:
            print(f"  {document['id']}  {document['title']}, revision {document['revision']}, {document['status']}")
        else:
            print(f"  {document['id']}  {document.get('title', '')}")
    for index, text in enumerate(suggestions):
        print(('Try: ' if index == 0 else 'Then: ') + text)


def render(result, before):
    route = f", {result['route']}" if result.get('route') else ''
    print(f"assistant ({result['seconds']:.0f} s{route}): {result['final_text'] or '(no reply)'}")
    for project, draft in result['snapshot'].get('drafts', {}).items():
        if before.get(project) != draft:
            fields = '; '.join(f"{key} {', '.join(value) if isinstance(value, list) else value}"
                               for key, value in sorted(draft.items()) if key not in ('project', 'revision'))
            print(f'  draft for {project}: {fields}')
    if result['failure']:
        print(f"  stopped: {result['failure']}")


def run(args):
    if args.assistant_action in ('host', 'audit', 'seed', 'chat'):
        from neuroshard.evolution import granite_shard_execution as runtime

        # Audits replay an owner's log bit for bit only in the pinned numerical environment, set before torch loads.
        runtime.configure()
        if args.assistant_action in ('host', 'audit', 'seed'):
            from neuroshard.evolution import assistant_growth_cohort3_eval
            from neuroshard.evolution.modular_reference_execution import ROOT, read
            assistant_growth_cohort3_eval.runtime(read(ROOT / runtime.RUNTIME))
            if getattr(args, 'threads', None) not in (None, 8):
                raise ValueError('auditable owners require the pinned eight-thread execution class')
    from neuroshard.assistant import network

    descriptor = network.descriptor(args.network)
    if args.assistant_action == 'seed':
        config, _, _ = network.seed(descriptor, args.home, args.public_host, audit_shards=args.audit_shards)
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
    chain = network.Chain(network.endpoints(descriptor), descriptor['chain_id'])
    if args.assistant_action == 'audit':
        import torch
        from neuroshard.assistant.audit import Auditor
        from neuroshard.evolution.sharded import granite

        if args.interval < 1:
            raise ValueError('audit interval must be at least one second')
        account = network.Account(args.home / 'account.key')
        state = chain.state()
        network.ensure_balance(chain, account, 2 * state['params']['challenge_deposit'], descriptor.get('faucet'))
        torch.set_num_threads(8)
        holdings = {}
        for rank in sorted(set(args.shard)):
            served, stage = network.fetch_stage(descriptor, rank, args.home)
            if rank == 0:
                raise ValueError('stage 0 is not a bonded owner shard')
            partition, _ = granite.load_partition(granite.load_config(stage['config']), stage['shard'], rank)
            adapter = network.load_adapter(partition, served, stage)
            network.warm(partition, adapter)
            holdings[rank] = partition, adapter
        auditor = Auditor(chain, account, descriptor, args.home, holdings)
        if args.once:
            auditor.once()
        else:
            auditor.run(args.interval)
        return
    if args.assistant_action == 'govern':
        govern(args, chain, network)
        return
    if args.assistant_action == 'grow':
        grow(args, chain, network)
        return
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
                              network.signing_key(args.home / f'log-{args.shard}.key'), args.home, args.threads or 8)
        owner.register(args.endpoint, descriptor.get('faucet'))
        if args.mirror_port:
            _, offered = network.serve_artifacts(descriptor, args.home, args.listen, args.mirror_port)
            print(f'Mirroring {len(offered)} pinned file(s) on port {args.mirror_port}')
        owner.run(args.listen, int(args.endpoint.rsplit(':', 1)[1]))
        return
    if not 1 <= args.budget <= network.MAX_PAID_POSITIONS:
        raise ValueError(f'budget must be between 1 and {network.MAX_PAID_POSITIONS} positions')
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
                drafts = dict(result['snapshot'].get('drafts', {}))
                if result['failure']:
                    break
    finally:
        conversation.close()
    print(f'The owners commit their logs now; the job settles after its challenge window. '
          f'Check it with: neuroshard assistant status --job {conversation.job_id}')


def govern(args, chain, network):
    """Every governance step is a signed transaction from this machine's own keys; no operator is involved."""
    from neuroshard.inference import optimistic as ledger

    state = chain.state()
    governance = state.get('governance')
    if args.govern_action == 'list':
        if not ledger.governed(state):
            print('The serving model is stewarded (one promotion key); owners do not govern it yet.')
            return
        print(f"Owner governance since block {governance['handed_over']['height']}: a proposal passes with "
              f"{governance['approval_ppm'] / 10_000:.4g}% of the bonded weight snapshotted when it opened.")
        for pid, proposal in sorted(governance['proposals'].items()):
            approving, total = ledger.tally(proposal)
            what = (f"validator {proposal['consensus_key'][:16]} at power {proposal['power']}"
                    if proposal.get('kind') == 'validator' else
                    f"model {proposal['model_root'][:16]}, quality {proposal['quality_root'][:16]}")
            print(f"  {pid}: {what}, {approving}/{total} approving, open until block {proposal['deadline']}")
        for key, power in sorted(state.get('validators', {}).items()):
            print(f"  admitted validator {key[:16]}: power {power}")
        return
    account = network.Account(args.home / 'account.key')
    if args.govern_action == 'handover':
        chain.submit(account, 'governance_handover', approval_ppm=args.approval_ppm, vote_blocks=args.vote_blocks)
    elif args.govern_action == 'propose':
        if not (args.model_root and args.quality_root):
            raise ValueError('propose needs --model-root and --quality-root')
        envelope = chain.submit(account, 'model_propose', model_root=args.model_root, quality_root=args.quality_root)
        print(f'proposal {ledger.transaction_id(envelope)}')
        return
    elif args.govern_action == 'propose-validator':
        if not args.consensus_key or args.power is None:
            raise ValueError('propose-validator needs --consensus-key and --power')
        envelope = chain.submit(account, 'validator_propose', consensus_key=args.consensus_key, power=args.power)
        print(f'proposal {ledger.transaction_id(envelope)}')
        return
    elif args.govern_action == 'vote':
        if not (args.proposal and args.shard):
            raise ValueError('vote needs --proposal and --shard')
        key = network.public_hex(network.signing_key(args.home / f'log-{args.shard}.key'))
        chain.submit(account, 'model_vote', proposal_id=args.proposal, log_key=key, approve=not args.reject)
    else:
        if not args.proposal:
            raise ValueError('enact needs --proposal')
        chain.submit(account, 'model_enact', proposal_id=args.proposal)
    print('committed')


def grow(args, chain, network):
    """Contributors train growth jobs segment by segment from their own keys; anyone may check a step.

    Training state is kept under ``--home/growth/<job>``: ``segment-<i>.state`` after each committed
    segment, so the next contributor (or a challenger building a proof) starts from committed bytes.
    """
    import hashlib

    from neuroshard.evolution import growth_replay
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.inference import growth as ledger_growth
    from neuroshard.inference import optimistic as ledger

    state = chain.state()
    if args.grow_action == 'list':
        for job_id, job in sorted(state.get('growth', {}).items()):
            print(f"{job_id}: recipe {job['recipe_root'][:16]}, {job['steps']} steps in "
                  f"{ledger_growth.segments(job)} segments, {len(job['commits'])} committed, "
                  f"{len(job['challenges'])} challenged, reward {job['reward']}")
        return
    account = network.Account(args.home / 'account.key')
    recipe = json.loads(args.recipe.read_text()) if args.recipe else None
    if args.grow_action == 'open':
        if not (recipe and args.steps):
            raise ValueError('open needs --recipe and --steps')
        make = model_factory(recipe)
        model = make()
        trainable, optimizer = growth_replay.start(model, recipe)
        envelope = chain.submit(account, 'growth_open', recipe_root=growth_replay.recipe_root(recipe, recipe['model_root']),
                                start_root=growth_replay.state_root(trainable, optimizer), steps=args.steps,
                                segment_steps=args.segment_steps, reward=atoms(args.reward))
        print(f'growth job {ledger.transaction_id(envelope)}')
        return
    job = state.get('growth', {}).get(args.job)
    if job is None:
        raise ValueError('no such open growth job')
    store = args.home / 'growth' / args.job
    store.mkdir(parents=True, exist_ok=True)
    if args.grow_action == 'train':
        if recipe is None or args.segment is None:
            raise ValueError('train needs --recipe and --segment')
        if growth_replay.recipe_root(recipe, recipe['model_root']) != job['recipe_root']:
            raise ValueError('this recipe is not the job\'s recipe')
        model = model_factory(recipe)()
        if args.segment == 0:
            trainable, optimizer = growth_replay.start(model, recipe)
        else:
            trainable, optimizer = growth_replay.load(model, recipe, (store / f'segment-{args.segment - 1}.state').read_bytes())
        if growth_replay.state_root(trainable, optimizer) != ledger_growth.previous_root(job, args.segment):
            raise ValueError('local state differs from the committed root this segment starts from')
        first = args.segment * job['segment_steps']
        roots = growth_replay.train_segment(model, trainable, optimizer, recipe, first,
                                            first + ledger_growth.segment_length(job, args.segment))
        (store / f'segment-{args.segment}.state').write_bytes(growth_replay.save(trainable, optimizer))
        chain.submit(account, 'growth_commit', job_id=args.job, segment=args.segment, step_roots=roots)
    elif args.grow_action == 'challenge':
        # Rebuild the state before the disputed step from committed bytes, check it, and publish the bundle
        # first, so validators hold it before the proof is submitted.
        if recipe is None or args.segment is None or args.step is None:
            raise ValueError('challenge needs --recipe, --segment and --step')
        model = model_factory(recipe)()
        segment, step = args.segment, args.step
        trainable, optimizer = (growth_replay.start(model, recipe) if segment == 0 else growth_replay.load(
            model, recipe, (store / f'segment-{segment - 1}.state').read_bytes()))
        first = segment * job['segment_steps']
        if step:
            growth_replay.train_segment(model, trainable, optimizer, recipe, first, first + step)
        payload = growth_replay.bundle(growth_replay.save(trainable, optimizer), first + step)
        proof_root = hashlib.sha256(payload).hexdigest()
        (store / f'proof-{proof_root}').write_bytes(payload)
        chain.submit(account, 'growth_challenge', job_id=args.job, segment=segment, step=step, proof_root=proof_root)
        from neuroshard.assistant import audit
        for endpoint in network.get('proof_endpoints', []):
            audit.post_proof(endpoint, proof_root, payload)
        print(f'challenge posted with proof {proof_root}; prove it after the next block')
        return
    else:
        if not args.challenge:
            raise ValueError('prove needs --challenge')
        chain.submit(account, 'growth_prove', job_id=args.job, challenge_id=args.challenge)
    print('committed')
