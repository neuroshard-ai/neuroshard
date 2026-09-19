"""Maintain paid provider availability and recover timed assignments locally.

Only a pinned local full node supplies authority. This controller never creates
cloud machines, changes model weights, or mistakes a heartbeat for useful work.
"""
import argparse
import fcntl
import json
from pathlib import Path
import threading
import time

from neuroshard.client.local_node import LocalNode
from neuroshard.demo import protocol
from .reference_data import identity, save
from .transactions import Outbox


class LockedOutbox:
    """Serialize access with a maintainer using its own SQLite connection."""
    def __init__(self, outbox, lock, node):
        self.outbox, self.lock, self.node = outbox, lock, node

    def send(self, operation, kind, **fields):
        # A maintainer can leave its exact signed heartbeat pending after a
        # delayed acknowledgement. Acquiring the mutex alone does not resolve
        # that durable operation. Finish or retire it from committed state
        # before signing the next acceptance or completed model response.
        deadline = time.monotonic() + fields.get('timeout', 120)
        with self.lock:
            while self.outbox.pending() not in (None, operation):
                try:
                    reconcile(self.node, self.outbox, self.node.query('/hosting/control'))
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.25)
            return self.outbox.send(operation, kind, **fields)

    def __getattr__(self, name):
        target = getattr(self.outbox, name)
        if not callable(target):
            return target
        def call(*args, **kwargs):
            with self.lock:
                return target(*args, **kwargs)
        return call


def reconcile(node, outbox, current):
    if outbox.pending() is None:
        return False
    body = outbox.pending_body()
    outbox.retire_control(current)
    if outbox.pending() and body.get('job_id'):
        outbox.retire_hosted(node.snapshot(body['job_id'], refresh=True))
    if outbox.pending():
        outbox.confirm(outbox.pending(), timeout=5)
    return True


def maintain(node, outbox, config, owner):
    """One bounded control action; no owner key or nonce is shared unlocked."""
    current = node.query('/hosting/control')
    profile, height = current['profile'], current['height']
    if not profile:
        raise ValueError('Provider maintenance requires the operated admission profile')
    if reconcile(node, outbox, current):
        return 'reconciled'
    market = current['hosting']
    provider = market['providers'].get(owner)
    if provider is None:
        return 'registration_required'
    if provider['endpoint'] != config['advertise']:
        raise ValueError('The provider registration no longer matches this runtime')
    valid_until = height + 64
    # Availability must cover a complete new request, not merely the next
    # heartbeat. Renew before a long job's quote loses its replacement runway.
    runway = current.get('request_blocks', 0) + profile['provider_heartbeat_blocks']
    if provider['registration_expires'] - height < max(profile['provider_blocks'] // 2, runway):
        outbox.send('registration-renewal-'+str(provider['registration_expires'])+'-'+str(valid_until),
                    'renew_provider', valid_until=valid_until, timeout=5)
        return 'registration_renewed'
    if height - provider['last_seen'] >= profile['provider_heartbeat_blocks'] // 2:
        outbox.send('heartbeat-'+str(height), 'heartbeat_provider', valid_until=valid_until, timeout=5)
        return 'heartbeat'
    executor = identity(json.loads(Path(config['profile']).read_bytes()))
    if current['executor_root'] != executor:
        return 'executor_update_required'
    owned = [(key, row) for key, row in market['offers'].items()
             if row['owner'] == owner and row['rank'] == config['rank'] and row['graph'] == current['graph']]
    if not owned:
        # Initial publishing is explicit. A dropped or expired advertisement
        # must not cause an unbounded series of automatically purchased offers.
        return 'offer_required'
    key, offer = min(owned, key=lambda row: row[0])
    duration = min(config['offer_blocks'], provider['registration_expires'] - height)
    if config['offer_blocks'] <= runway:
        raise ValueError('Offer lifetime cannot cover a complete request and renewal runway')
    if offer['expires'] - height < max(config['offer_blocks'] // 2, runway):
        outbox.send('offer-renewal-'+key+'-'+str(offer['expires'])+'-'+str(valid_until), 'renew_expert_offer',
            offer_id=key, expires_in=duration, valid_until=valid_until, timeout=5)
        return 'offer_renewed'
    return 'available'


def recover(node, outbox):
    current = node.query('/hosting/control')
    if not current['profile']:
        raise ValueError('Automatic recovery requires standing admission')
    if reconcile(node, outbox, current):
        return 'reconciled'
    for key, lease in sorted(current['hosting']['leases'].items()):
        if lease['status'] not in ('preparing', 'ready'):
            continue
        deadline = lease['prepare_deadline'] if lease['status'] == 'preparing' else lease['work_deadline']
        if current['height'] <= deadline:
            continue
        try:
            quote = node.query('/hosting/replacement', {'job_id': key})
        except ValueError:
            continue
        # Epoch and absolute deadline make this intent permanently retireable.
        # Admission rechecks capacity; a read-only quote is not a reservation.
        operation = 'recover-'+key+'-'+lease['assignment_root']+'-'+str(quote['valid_until'])
        outbox.send(operation, 'recover_hosted_job', job_id=key,
            assignment_root=quote['assignment_root'], valid_until=quote['valid_until'], timeout=5)
        return 'replacement_submitted'
    return 'idle'


def maintain_background(node, config, owner, lock, stop):
    """Heartbeat during downloads and model execution, not only between jobs."""
    home = Path(config['home'])
    outbox = Outbox(home/'transactions.sqlite', config['node_rpc'], config['chain_id'], owner)
    try:
        while not stop.is_set():
            try:
                with lock:
                    phase = maintain(node, outbox, config, owner.public_key)
                save(home/'maintenance.json', {'phase': phase, 'time': time.time()})
            except (OSError, ValueError, RuntimeError) as error:
                save(home/'maintenance.json', {'phase': 'retry', 'error': type(error).__name__, 'time': time.time()})
            stop.wait(2)
    finally:
        outbox.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    home = Path(config['home'])
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (home/'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = protocol.Identity.load_or_create(home/'identity')
        node = LocalNode(config['node_rpc'], config['chain_id'], config['manifest_root'])
        outbox = Outbox(home/'transactions.sqlite', config['node_rpc'], config['chain_id'], owner)
        deadline = time.monotonic() + config['run_seconds']
        try:
            while time.monotonic() < deadline:
                try:
                    result = {'phase': recover(node, outbox)}
                except (OSError, ValueError, RuntimeError) as error:
                    result = {'phase': 'retry', 'error': type(error).__name__}
                save(home/'recovery.json', {**result, 'time': time.time()})
                time.sleep(2)
        finally:
            outbox.close()


if __name__ == '__main__':
    main()
