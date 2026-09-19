"""Run an assigned model partition using a local validator and a local key.

This is the opt-in provider research profile. The running public 0.4.0 network
does not enable these transactions. No network input installs executable code.
"""
import argparse
import fcntl
import json
from pathlib import Path
import threading
import time

from neuroshard.client.local_node import LocalNode as PinnedNode
from neuroshard.demo import protocol
from . import expert_lifecycle, provider_assets, provider_transport as transport
from .reference_data import identity, save
from .transactions import Outbox


class LocalNode(PinnedNode):
    def lookup(self, job_id):
        snapshot = self.snapshot(job_id)
        state = {**snapshot, 'hosting': {'leases': {job_id: snapshot['lease']}}}
        return transport.context(state, job_id)


def response_claim(job, result):
    """Construct the same complete statement the native transition will audit."""
    from .sharded.graph_service import inference_transcript
    graph = job['graph']
    if result['graph'] != identity(graph) or result['request'] != job['request']:
        raise ValueError('Execution changed the reserved graph or request')
    claim = {'kind': 'expert_inference', 'job_id': job['id'], 'graph': graph,
        'model_root': identity(graph), 'executor_root': graph['executor_root'],
        'request': job['request'], 'outputs': result['outputs'], 'text': result['text'],
        'stages': sum(len(row['token_ids']) for row in result['outputs']), 'record_root': '0'*64}
    if 'answering' in graph:
        claim['response'] = result['answering']
    transcript = inference_transcript(claim, result)
    claim['record_root'] = identity(transcript)
    return claim, transcript


def execute(job, network, peer, *, on_text=None):
    """Compute locally, then authenticate every owner's identical result."""
    request = job['request']
    result = network.answer(request.get('messages', request.get('question')), request['max_tokens'], on_text=on_text)
    claim, transcript = response_claim(job, result)
    rank = str(peer.rank)
    receipt = (expert_lifecycle.answering_receipt(peer.routing['chain_id'], job, claim['response'],
               claim['record_root'], rank) if 'response' in claim else
               expert_lifecycle.inference_receipt(peer.routing['chain_id'], job, claim['outputs'],
               claim['text'], claim['record_root'], rank))
    signed = peer.identity.sign(receipt)
    gathered = network.all_owners.exchange(signed)
    receipts = {}
    for index, envelope in enumerate(gathered):
        body, signer = protocol.verify(envelope)
        if body != {**receipt, 'rank': str(index)} or signer != job['workers'][str(index)]:
            raise ValueError('Providers disagree on their complete execution receipts')
        receipts[str(index)] = envelope
    payload = {'job_id': job['id'], 'workers': receipts, 'transcript_root': claim['record_root']}
    if 'response' in claim:
        payload.update(response=claim['response'], kind='respond_answering')
    else:
        payload.update(outputs=claim['outputs'], text=claim['text'], kind='respond_expert')
    return {'claim': claim, 'transcript': transcript, 'submission': payload}


def run_job(config, node, owner, box, outbox, job_id, cache=None, events=None):
    """One capacity slot; restart requires native replacement of its epoch."""
    from .sharded.graph_execution import GraphNetwork
    from .sharded.peer_wire import ServingMesh
    snapshot = node.snapshot(job_id, refresh=True)
    job, lease, rank = snapshot['job'], snapshot['lease'], config['rank']
    if (not job or not lease or job['workers'].get(str(rank)) != owner.public_key
            or job['hosting'] != lease['assignment_root']):
        raise ValueError('This provider does not own the requested assignment')
    assigned = lease['providers'][str(rank)]
    _, certificate = transport.certificate(config['home'], owner)
    if assigned['certificate'] != certificate or assigned['endpoint'] != config['advertise']:
        raise ValueError('The native assignment pins a different local endpoint or certificate')
    epoch = lease['assignment_root']
    if events is not None and rank == 0:
        events.begin(job, snapshot['chain_id'], epoch)
    home = Path(config['home'])/'jobs'/job_id/epoch
    home.mkdir(parents=True, exist_ok=True)
    # A restarted transport cannot know which frame acknowledgements remote
    # owners observed. It must never reset sequences within the same epoch.
    if (home/'started.json').exists():
        raise transport.Unavailable('This interrupted execution needs a fresh native assignment epoch')
    profile = json.loads(Path(config['profile']).read_bytes())
    inventory = json.loads(Path(config['inventory']).read_bytes())
    assets = Path(config['home'])/'models'/identity(job['graph'])
    prepared = provider_assets.prepare(job['graph'], profile, rank, assets, config['source_home'],
        inventory, config['mirrors'], max_bytes=config['max_bytes'], max_seconds=config['prepare_seconds'])
    save(home/'assets.json', prepared)
    current = node.snapshot(job_id, refresh=True)
    if current['lease'] is None or current['lease']['assignment_root'] != epoch:
        raise transport.Unavailable('The assignment changed while restoring model bytes')
    if owner.public_key not in current['lease']['accepted']:
        outbox.send('accept-'+epoch, 'accept_hosted_job', job_id=job_id, assignment_root=epoch)
    deadline = time.monotonic() + config['prepare_seconds']
    while time.monotonic() < deadline:
        current = node.snapshot(job_id, refresh=True)
        node.lookup(job_id)
        if current['lease']['assignment_root'] != epoch:
            raise transport.Unavailable('The assignment was replaced before all providers accepted')
        if current['lease']['status'] == 'ready':
            break
        time.sleep(.25)
    else:
        raise transport.Unavailable('Not every assigned provider accepted within the local time bound')
    routing = node.lookup(job_id)
    save(home/'started.json', {'job_id': job_id, 'assignment_root': epoch, 'rank': rank})
    peer = transport.Peer(owner, routing, rank, box, timeout=config['frame_seconds'],
                          allow_private=config.get('allow_private', False))
    try:
        model_key = (identity(job['graph']), rank)
        network = cache.get(model_key) if cache is not None else None
        if network is None:
            if cache is not None:
                # One advertised capacity slot keeps one complete local model
                # partition resident. Old policy objects contain cycles.
                cache.clear()
                import gc
                gc.collect()
            network = GraphNetwork(job['graph'], profile, objects=assets/'objects', interpreter=assets/'interpreter',
                seed=assets/'seed', source_home=config['source_home'], rank=rank, mesh=ServingMesh(peer))
            if cache is not None:
                cache[model_key] = network
        else:
            network.rebind(ServingMesh(peer))
        callback = (lambda text: events.emit(job_id, epoch, text=text)) if events is not None and rank == 0 else None
        result = execute(job, network, peer, on_text=callback)
        if events is not None and rank == 0:
            events.emit(job_id, epoch, text=result['claim']['text'], status='generated')
        save(home/'result.json', result)
        if rank == 0:
            while node.query('/candidate') is not None:
                current = node.lookup(job_id)
                if current['assignment_root'] != epoch:
                    raise transport.Unavailable('The completed response belongs to a replaced assignment')
                time.sleep(.25)
            submission = {**result['submission'], 'audit_budget': lease['audit_budget']}
            kind = submission.pop('kind')
            receipt = outbox.send('response-'+epoch, kind, **submission)
            save(home/'submitted.json', receipt)
            if events is not None:
                events.emit(job_id, epoch, status='submitted')
        return result
    finally:
        peer.close()
        box.retire(job_id, epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--job', help='Serve this native job, or discover owned assignments when omitted')
    parser.add_argument('--identity', action='store_true', help='Create local keys and print public registration fields')
    parser.add_argument('--publish-offer', action='store_true', help='Register configured collateral and advertise one capacity slot')
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    home = Path(config['home'])
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (home/'provider.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = protocol.Identity.load_or_create(home/'identity')
        tls, fingerprint = transport.certificate(home, owner)
        if args.identity:
            print(json.dumps({'owner': owner.public_key, 'endpoint': config['advertise'], 'certificate': fingerprint}))
            return
        node = LocalNode(config['node_rpc'], config['chain_id'], config['manifest_root'])
        box = transport.Mailbox(owner.public_key, node.lookup, timeout=config['frame_seconds'])
        from .provider_stream import StreamLog
        events = StreamLog(owner, node.snapshot)
        server = transport.Server((config['bind'], config['port']), tls, box, events=events)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        outbox = Outbox(home/'transactions.sqlite', config['node_rpc'], config['chain_id'], owner)
        control_lock = threading.RLock()
        maintenance_stop, maintenance_thread = threading.Event(), None
        if config.get('maintain') and not args.publish_offer:
            from .provider_control import LockedOutbox, maintain_background
            outbox = LockedOutbox(outbox, control_lock)
            maintenance_thread = threading.Thread(target=maintain_background,
                args=(node, config, owner, control_lock, maintenance_stop), daemon=True)
            maintenance_thread.start()
        deadline = time.monotonic() + config['run_seconds']
        attempted = set()
        cache = {}
        try:
            if args.publish_offer:
                graph = node.query('/expert_lifecycle')['serving_graph']
                profile = json.loads(Path(config['profile']).read_bytes())
                if identity(profile) != graph['executor_root']:
                    raise ValueError('The local executor differs from the currently accepted graph')
                registration = {'endpoint': config['advertise'], 'certificate': fingerprint,
                                'collateral': config['collateral']}
                market = node.query('/hosting')
                if owner.public_key not in market['providers']:
                    outbox.send('register-'+identity(registration), 'register_provider', **registration)
                else:
                    registered = market['providers'][owner.public_key]
                    if registered['endpoint'] != config['advertise'] or registered['certificate'] != fingerprint:
                        raise ValueError('Update the native provider endpoint before publishing a new offer')
                offer = {'graph': identity(graph), 'rank': config['rank'], 'fee': config['fee'],
                         'capacity': 1, 'expires_in': config['offer_blocks']}
                operation = 'offer-'+identity(offer)
                outbox.send(operation, 'offer_expert', **offer)
                print(json.dumps({'owner': owner.public_key, 'offer_id': outbox.logical_id(operation), **offer}))
                return
            while time.monotonic() < deadline:
                with control_lock:
                    if outbox.pending():
                        body = outbox.pending_body()
                        if body.get('job_id'):
                            outbox.retire_hosted(node.snapshot(body['job_id'], refresh=True))
                        if outbox.pending():
                            try:
                                outbox.confirm(outbox.pending(), timeout=5)
                            except (OSError, ValueError, TimeoutError):
                                time.sleep(.5)
                                continue
                market = node.query('/hosting')
                if not market:
                    raise ValueError('The pinned genesis does not enable provider hosting')
                jobs = [args.job] if args.job else sorted(market['leases'])
                for job_id in jobs:
                    lease = market['leases'].get(job_id)
                    if not lease or lease['status'] not in ('preparing', 'ready'):
                        continue
                    assigned = lease['providers'].get(str(config['rank']))
                    if not assigned or assigned['owner'] != owner.public_key or lease['assignment_root'] in attempted:
                        continue
                    attempted.add(lease['assignment_root'])
                    try:
                        run_job(config, node, owner, box, outbox, job_id, cache, events)
                    except (OSError, ValueError, RuntimeError) as error:
                        # Keep failures local and bounded; don't print conversations,
                        # keys, signed frames or object-store credentials.
                        save(home/'last-failure.json', {'job_id': job_id,
                            'assignment_root': lease['assignment_root'], 'error': type(error).__name__})
                        if config['rank'] == 0 and not (outbox.pending_body() or {}).get('kind', '').startswith('respond_'):
                            events.emit(job_id, lease['assignment_root'], text='', status='failed')
                    if args.job:
                        return
                time.sleep(.5)
        finally:
            maintenance_stop.set()
            if maintenance_thread is not None:
                maintenance_thread.join(timeout=30)
            box.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            outbox.close()


if __name__ == '__main__':
    main()
