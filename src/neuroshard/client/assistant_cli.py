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
    govern.add_argument('govern_action', choices=('list', 'propose', 'propose-validator', 'propose-levy', 'vote',
                                                  'enact', 'handover'))
    govern.add_argument('--levy-ppm', type=int, help='propose-levy: share of owner payments that funds growth (ppm)')
    govern.add_argument('--consensus-key', help='propose-validator: the CometBFT ed25519 public key, hex')
    govern.add_argument('--power', type=int, help='propose-validator: voting power, 0 removes')
    govern.add_argument('--model-root', help='propose: the version to serve, as its model root')
    govern.add_argument('--quality-root', help='propose: digest of the published release-check evidence')
    govern.add_argument('--proposal', nargs='+', help='vote/enact: the proposal ID (enact takes several, one block)')
    govern.add_argument('--shard', type=int, help='vote: the shard whose owner log key votes')
    govern.add_argument('--reject', action='store_true', help='vote: vote no')
    govern.add_argument('--approval-ppm', type=int, default=666_667, help='handover: approving bond share needed')
    govern.add_argument('--vote-blocks', type=int, default=600, help='handover: blocks a proposal stays open')
    grow = actions.add_parser('grow', parents=[common],
                              help='Grow the assistant on the network: open, train, challenge or prove a growth job')
    grow.add_argument('grow_action', choices=('list', 'open', 'train', 'challenge', 'prove', 'agent', 'propose-corpus'))
    grow.add_argument('--corpus', type=Path, help='propose-corpus: a realistic training corpus (realistic_training)')
    grow.add_argument('--recipes', type=Path, help='agent: directory of published recipe JSON files to contribute to')
    grow.add_argument('--state-stores', nargs='*', default=[], help='agent: HTTP stores of committed training states')
    grow.add_argument('--recipe-stores', nargs='*', default=[],
                      help='agent: HTTP stores of proposed recipes (default: the network proof stores)')
    grow.add_argument('--plan', action='store_true',
                      help='agent: probe the served routes on fresh beacon cases and propose recipes for weak families')
    grow.add_argument('--evaluate', action='store_true',
                      help='agent: run sealed evaluations of grown units and propose the ones that earn it')
    grow.add_argument('--vote-shards', type=int, nargs='*', default=[],
                      help='agent: vote as the owners of these shards (their log keys in --home), after re-verifying')
    grow.add_argument('--probe-every', type=float, default=6 * 3600, help='agent --plan: seconds between probes')
    grow.add_argument('--growth-steps', type=int, default=32, help='agent --plan: optimizer steps per recipe')
    grow.add_argument('--reward-per-step', type=int, default=40_000,
                      help='agent --plan: atoms a recipe pays per step (owners vote on it with the recipe)')
    grow.add_argument('--minimum', type=float, default=0.9, help='sealed pass rate a grown unit must reach')
    grow.add_argument('--no-contribute', action='store_true', help='agent: check and judge, never train segments')
    grow.add_argument('--per-family', type=int, default=48, help='agent --plan: fresh cases per weak family')
    grow.add_argument('--interval', type=float, default=30, help='agent: seconds between ledger passes')
    grow.add_argument('--once', action='store_true', help='agent: one pass, then exit')
    grow.add_argument('--recipe', type=Path, help='open/train/prove: the growth recipe JSON (arm, spec, data, base)')
    grow.add_argument('--job', help='train/challenge/prove: the growth job ID')
    grow.add_argument('--segment', type=int, help='train/challenge: the segment index')
    grow.add_argument('--step', type=int, help='challenge: the step within the segment')
    grow.add_argument('--steps', type=int, help='open: optimizer steps')
    grow.add_argument('--segment-steps', type=int, default=8, help='open: steps per segment')
    grow.add_argument('--reward', default='10', help='open: NEURO escrowed for contributors')
    grow.add_argument('--challenge', help='prove: the challenge ID')
    validator = actions.add_parser('validator', parents=[common],
                                   help='Join the chain as a node owners can vote in as a validator; keys stay here')
    validator.add_argument('--public-host', required=True, help='Address peers reach this machine at')
    validator.add_argument('--audit-shards', type=int, nargs='*', default=[1, 2],
                           help='Shards to hold so this validator can judge fraud claims itself')
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
    if args.assistant_action in ('host', 'audit', 'seed', 'validator', 'chat', 'grow'):
        from neuroshard.evolution import granite_shard_execution as runtime

        # Audits replay an owner's log bit for bit only in the pinned numerical environment, set before torch loads.
        runtime.configure()
        if args.assistant_action in ('host', 'audit', 'seed', 'validator', 'grow'):  # growth steps replay too
            from neuroshard.evolution import assistant_growth_cohort3_eval
            from neuroshard.evolution.modular_reference_execution import ROOT, read
            assistant_growth_cohort3_eval.runtime(read(ROOT / runtime.RUNTIME))
            if getattr(args, 'threads', None) not in (None, 8):
                raise ValueError('auditable owners require the pinned eight-thread execution class')
    from neuroshard.assistant import network

    descriptor = network.descriptor(args.network)
    if args.assistant_action in ('seed', 'validator'):
        from neuroshard.evolution import growth_runtime
        from neuroshard.inference import optimistic_network

        if args.assistant_action == 'seed':
            config, chain, _ = network.seed(descriptor, args.home, args.public_host, audit_shards=args.audit_shards)
        else:
            chain = network.Chain(network.endpoints(descriptor), descriptor['chain_id'])
            descriptor = following(descriptor, chain, args.home, network)
            config = network.join_validator(descriptor, args.home, args.public_host, args.audit_shards)
        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())

        def rebuild(current):
            return growth_runtime.follow(current, chain.state(), args.home, current.get('proof_endpoints', []))

        def restart(config):
            optimistic_network.stop(config)
            optimistic_network.start(config)

        try:
            while not stopped.wait(30):
                try:  # the validator follows owner-enacted versions too, so it judges each job with its own weights
                    descriptor = network.follow_validator(config, descriptor, chain, args.home, rebuild, restart)
                    if args.assistant_action == 'validator' and network.fell_behind(config, chain):
                        # CometBFT leaves fast sync if its peers drop out for a while (the seed reloading its app),
                        # then catches up at consensus speed; a restart resumes fast sync.
                        print('Behind the network outside fast sync; restarting the node', flush=True)
                        restart(config)
                except Exception as error:  # noqa: BLE001 - keep validating; retry on the next pass
                    print(f'Validator version check: {error}', flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            optimistic_network.stop(config)
        return
    chain = network.Chain(network.endpoints(descriptor), descriptor['chain_id'])
    descriptor = following(descriptor, chain, args.home, network)
    if args.assistant_action == 'audit':
        import torch
        from neuroshard.assistant.audit import Auditor
        from neuroshard.evolution.sharded import granite

        if args.interval < 1:
            raise ValueError('audit interval must be at least one second')
        account = network.Account(args.home / 'account.key')
        state = chain.state()
        network.ensure_balance(chain, account, 2 * state['params']['challenge_deposit'], descriptor.get('faucet'))
        from neuroshard.assistant.audit import VersionChanged

        torch.set_num_threads(8)
        while True:
            holdings = {}
            for rank in sorted(set(args.shard)):
                served, stage = network.fetch_stage(descriptor, rank, args.home)
                if rank == 0:
                    raise ValueError('stage 0 is not a bonded owner shard')
                partition, _ = granite.load_partition(granite.load_config(stage['config']), stage['shard'], rank)
                adapter = network.load_adapter(partition, served, stage)
                network.warm(partition, adapter)
                holdings[rank] = partition, adapter
            auditor = Auditor(chain, account, descriptor, args.home, holdings,
                              model_root=network.model_root(descriptor))
            try:
                if args.once:
                    auditor.once()
                    return
                auditor.run(args.interval)
            except VersionChanged as changed:
                # The owners enacted another version: audit it with its own weights, rebuilt from the chain.
                upgraded = following(descriptor, chain, args.home, network)
                if network.model_root(upgraded) != str(changed):
                    print(f'Waiting to rebuild version {str(changed)[:16]} before auditing', flush=True)
                    __import__('time').sleep(60)
                descriptor = upgraded
                print(f'Auditing version {network.model_root(descriptor)[:16]}', flush=True)
        return
    if args.assistant_action == 'govern':
        govern(args, chain, network)
        return
    if args.assistant_action == 'grow':
        grow(args, chain, network, descriptor)
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
        host_following(args, chain, network, descriptor)
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


def following(descriptor, chain, home, network, progress=print):
    """The descriptor of the version the chain serves; rebuilt locally when owners enacted a grown unit."""
    from neuroshard.evolution import growth_runtime

    try:
        return growth_runtime.follow(descriptor, chain.state(), home, descriptor.get('proof_endpoints', []), progress)
    except Exception as error:  # noqa: BLE001 - keep the pinned version; the action reports any real mismatch
        progress(f'Serving version check: {error}')
        return descriptor


def host_following(args, chain, network, descriptor, poll=30.0):
    """Host one shard, and move to each version the owners enact, without anyone publishing a release.

    The served version is checked every ``poll`` seconds; when the chain names a new one, this owner finishes
    its open work, rebuilds the version from committed growth, loads its stage again and keeps serving."""
    import threading
    import time as clock

    mirrored = False
    while True:
        served, stage = network.fetch_stage(descriptor, args.shard, args.home)
        owner = network.Owner(served, stage, args.shard, chain, network.Account(args.home / 'account.key'),
                              network.signing_key(args.home / f'log-{args.shard}.key'), args.home, args.threads or 8)
        owner.register(args.endpoint, descriptor.get('faucet'))
        if args.mirror_port and not mirrored:
            _, offered = network.serve_artifacts(descriptor, args.home, args.listen, args.mirror_port)
            print(f'Mirroring {len(offered)} pinned file(s) on port {args.mirror_port}')
            mirrored = True
        stop = threading.Event()
        thread = threading.Thread(target=owner.run, args=(args.listen, int(args.endpoint.rsplit(':', 1)[1]), stop),
                                  daemon=True)
        thread.start()
        while thread.is_alive():
            clock.sleep(poll)
            try:
                root = chain.state()['model_root']
            except Exception:  # noqa: BLE001 - an unreachable RPC is retried; serving continues meanwhile
                continue
            if root != served.root:
                upgraded = following(descriptor, chain, args.home, network)
                if network.model_root(upgraded) == root:
                    print(f'The owners enacted {root[:16]}; switching after open work commits')
                    stop.set()
                    thread.join()
                    if getattr(owner, 'pending_jobs', None):
                        owner.commit_pending()
                    descriptor = upgraded
                    break
        else:
            return


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
            kind = proposal.get('kind')
            what = (f"validator {proposal['consensus_key'][:16]} at power {proposal['power']}" if kind == 'validator'
                    else f"growth levy {proposal['levy_ppm'] / 10_000:.4g}%" if kind == 'levy'
                    else f"recipe {proposal['recipe_root'][:16]}: {proposal['steps']} steps for {proposal['reward']}"
                    if kind == 'recipe' else
                    f"model {proposal['model_root'][:16]}, quality {proposal['quality_root'][:16]}")
            print(f"  {pid}: {what}, {approving}/{total} approving, open until block {proposal['deadline']}")
        print(f"Growth: levy {governance.get('growth_levy_ppm', 0) / 10_000:.4g}% of owner payments, pool "
              f"{state.get('growth_pool', 0)}, {len(state.get('growth_queue', []))} recipe(s) queued, "
              f"{len(state.get('growth', {}))} job(s) open.")
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
    elif args.govern_action == 'propose-levy':
        if args.levy_ppm is None:
            raise ValueError('propose-levy needs --levy-ppm')
        envelope = chain.submit(account, 'growth_levy', levy_ppm=args.levy_ppm)
        print(f'proposal {ledger.transaction_id(envelope)}')
        return
    elif args.govern_action == 'vote':
        if not (args.proposal and args.shard):
            raise ValueError('vote needs --proposal and --shard')
        key = network.public_hex(network.signing_key(args.home / f'log-{args.shard}.key'))
        for proposal in args.proposal:
            chain.submit(account, 'model_vote', proposal_id=proposal, log_key=key, approve=not args.reject)
    else:
        if not args.proposal:
            raise ValueError('enact needs --proposal')
        # Several enactments are signed with consecutive nonces and broadcast together, so they usually commit in
        # one block: a validator set grows from one to four at once instead of passing through two of two (which
        # would need both to sign every block).
        # The node admits one transaction per account per block (nonces are checked against committed state), so
        # these are enacted one block apart: a validator set grows by one validator per block.
        for proposal in args.proposal:
            chain.submit(account, 'model_enact', proposal_id=proposal)
    print('committed')


def grow(args, chain, network, descriptor=None):
    """Contributors train growth jobs segment by segment from their own keys; anyone may check a step.

    Training state is kept under ``--home/growth/<job>``: ``segment-<i>.state`` after each committed
    segment, so the next contributor (or a challenger building a proof) starts from committed bytes.
    """
    import hashlib

    from neuroshard.evolution import growth_replay
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.inference import growth as ledger_growth
    from neuroshard.inference import optimistic as ledger

    if args.grow_action == 'agent':
        return grow_agent(args, chain, network, descriptor or {})
    if args.grow_action == 'propose-corpus':
        return propose_corpus(args, chain, network, descriptor or {})
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
        for endpoint in (descriptor or {}).get('proof_endpoints', []):
            audit.post_proof(endpoint, proof_root, payload)
        print(f'challenge posted with proof {proof_root}; prove it after the next block')
        return
    else:
        if not args.challenge:
            raise ValueError('prove needs --challenge')
        chain.submit(account, 'growth_prove', job_id=args.job, challenge_id=args.challenge)
    print('committed')


def propose_corpus(args, chain, network, descriptor):
    """Propose a documents recipe trained on a published realistic corpus, continuing the served unit.

    The recipe carries the corpus in full, so every owner verifies it by replay before voting
    (``realistic_training.verify``); it is published to the network's stores by its root."""
    from neuroshard.assistant import audit
    from neuroshard.evolution import growth_replay, growth_runtime, realistic_training

    if not args.corpus or not args.steps:
        raise ValueError('propose-corpus needs --corpus and --steps')
    spec = growth_runtime.growth_spec(args.steps)
    served = growth_runtime.Served(descriptor, args.home, spec)
    root = chain.state()['model_root']
    planner = growth_runtime.RoutePlanner(served, served.make_parent, served.tokenizer, served.policies, spec, root,
                                          served.units)
    encode, policy = planner.encoder('documents'), served.policies['documents']
    from neuroshard.evolution import growth_planner

    families = sorted(growth_planner.sealed.templates('documents'))
    replay = growth_planner.recipe_from(families, 'f0' * 32, policy, encode, 16, 'replay')
    recipe = realistic_training.recipe(json.loads(args.corpus.read_text()), policy, spec, root, encode, replay,
                                       served.units['documents'])
    if not planner.check({**recipe, 'steps': args.steps}):
        raise ValueError('the recipe does not pass the owners\' own check here')
    start = growth_replay.state_root(*growth_replay.start(served.make_parent(), recipe))
    recipe.update(start_root=start, steps=args.steps, segment_steps=args.segment_steps or 8,
                  reward=args.steps * args.reward_per_step)
    recipe_root = growth_replay.recipe_root(recipe, recipe['model_root'])
    payload = json.dumps(recipe, sort_keys=True).encode()
    (args.home / 'growth-agent' / 'recipes').mkdir(parents=True, exist_ok=True)
    (args.home / 'growth-agent' / 'recipes' / f'{recipe_root}.json').write_bytes(payload)
    account = network.Account(args.home / 'account.key')
    envelope = chain.submit(account, 'recipe_propose', recipe_root=recipe_root, start_root=start, steps=args.steps,
                            segment_steps=recipe['segment_steps'], reward=recipe['reward'])
    for endpoint in descriptor.get('proof_endpoints', []):
        audit.post_proof(endpoint, recipe_root, payload, timeout=300)
    print(json.dumps({'proposal_id': __import__('neuroshard.inference.optimistic', fromlist=['transaction_id'])
                      .transaction_id(envelope), 'recipe_root': recipe_root, 'start_root': start,
                      'steps': args.steps, 'reward': recipe['reward'], 'bytes': len(payload)}))


def grow_agent(args, chain, network, descriptor):
    """Run the autonomous growth loop with this machine's keys: contribute, check, challenge, seal, evaluate.

    It acts only through ledger transactions it is allowed to sign, and only on work it can reproduce."""
    import time as clock

    from neuroshard.evolution import growth_agent, growth_replay
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.evolution.growth_runtime import StateTrainer

    # Committed training states travel through the same stores as recipes and proofs, unless told otherwise.
    args.state_stores = list(args.state_stores) or list(descriptor.get('proof_endpoints', []))
    recipes = {}
    for path in sorted(Path(args.recipes).glob('*.json')) if args.recipes else []:
        recipe = json.loads(path.read_text())
        recipes[growth_replay.recipe_root(recipe, recipe['model_root'])] = recipe
    trainers = {root: StateTrainer(args.home / 'growth-states' / root, model_factory(recipe), args.state_stores)
                for root, recipe in recipes.items()}

    from neuroshard.assistant import audit

    def deliver(root, payload):
        for endpoint in descriptor.get('proof_endpoints', []):
            try:
                audit.post_proof(endpoint, root, payload)
            except Exception as error:  # noqa: BLE001 - one store being down does not stop the others
                print(f'proof store {endpoint}: {error}', flush=True)

    class ByRecipe:
        """One protocol trainer per recipe, created on first use, so recipes the network proposes later are trained too."""
        def trainer(self, recipe):
            root = growth_replay.recipe_root(recipe, recipe['model_root'])
            if root not in trainers:
                trainers[root] = StateTrainer(args.home / 'growth-states' / root, model_factory(recipe), args.state_stores)
            return trainers[root]

        def segment(self, recipe, job_id, start, first, last):
            return self.trainer(recipe).segment(recipe, job_id, start, first, last)

        def replay(self, recipe, job, index, step):
            return self.trainer(recipe).replay(recipe, job, index, step)

        def state_bytes(self, recipe, root):
            return self.trainer(recipe).fetch(root)

    account = network.Account(args.home / 'account.key')
    by_recipe = ByRecipe()
    planner = evaluator = None
    owner_keys = {network.public_hex(network.signing_key(args.home / f'log-{shard}.key')): account
                  for shard in args.vote_shards}
    if args.plan or args.evaluate or owner_keys:
        from neuroshard.evolution import growth_runtime

        spec = growth_runtime.growth_spec(args.growth_steps)
        served = growth_runtime.Served(descriptor, args.home, spec)
        root = chain.state()['model_root']
        planner = growth_runtime.RoutePlanner(served, served.make_parent, served.tokenizer, served.policies, spec,
                                              root, served.units, reward_per_step=args.reward_per_step,
                                              per_family=args.per_family, segment_steps=args.segment_steps or 8)
        served.keep = not owner_keys
        evaluator = growth_runtime.Evaluators(recipes, served, by_recipe, served.policies, descriptor, args.minimum)
    agent = growth_agent.Agent(chain, account, args.home / 'growth-agent', recipes=recipes, trainer=by_recipe,
                               deliver=deliver, evaluator=evaluator if (args.evaluate or owner_keys) else None,
                               owner_keys=owner_keys or None, planner=planner if args.plan else None,
                               probe_every=args.probe_every)
    if planner is not None:
        agent.recipe_check = planner.check
    if evaluator is not None:
        evaluator.recipes = agent.recipes  # recipes fetched from the network are evaluated too
    agent.contributing = not args.no_contribute
    agent.recipe_stores = list(args.recipe_stores) or list(descriptor.get('proof_endpoints', []))
    while True:
        try:
            # Keep enough to bond the next segment and pay fees; commit bonds return when a job settles.
            state = chain.state()
            from neuroshard.inference import growth as ledger_growth

            needed = 2 * ledger_growth.params(state)['growth_bond'] + 8 * state['params']['fee']
            network.ensure_balance(chain, account, needed, descriptor.get('faucet'), lambda *_: None)
            for action in agent.tick():
                print(json.dumps(action, sort_keys=True), flush=True)
        except Exception as error:  # noqa: BLE001 - a refused transaction or a dropped RPC: report, retry next pass
            print(json.dumps({'kind': 'error', 'error': f'{type(error).__name__}: {error}'}), flush=True)
            if args.once:
                raise
        if args.once:
            return
        clock.sleep(args.interval)
