#!/usr/bin/env python3
"""Frozen accepted-model gate; the persistent controller performs every audit.

Loss cases stop one provider service (preventing its automatic restart), then
observe native replacement with a preexisting replica on another machine.
No replacement transaction or audit verdict is submitted by this driver.
"""
import argparse
import json
from pathlib import Path
import time

from neuroshard.client import provider_wire, wire
from neuroshard.client.hosted import Customer
from neuroshard.client.local_node import LocalNode
from neuroshard.evolution.reference_data import identity, save
from neuroshard.evolution.transactions import Outbox
from ordinary_cloud import Cloud
from portable_native_trial import Network
from operate_alpha import read

ROOT = Path(__file__).resolve().parents[1]

def batch(cloud, network, providers, customers, cases, maximum, *, report_home=None):
    report_home = report_home or cloud.home
    began = time.monotonic()
    rows, streams, measurements, assigned = [], {}, {}, {}
    for customer, case in zip(customers, cases):
        row = customer.prepare(case['messages'], maximum, 3_000_000_000)
        rows.append(row)
        measurements[row['id']] = {'case': case['id'], 'events': [], 'first_visible_seconds': None,
                                  'generated_seconds': None, 'settled_seconds': None}
    for customer, row in zip(customers, rows):
        assigned[row['id']] = time.monotonic()
        update = customer.tick(row)
        if update['status'] != 'serving':
            raise ValueError('A paid customer did not acquire an available native provider replica')
        measurements[row['id']]['job_id'] = row['job_id']
    # Generation latency starts before the single atomic admission submission.
    # Customer-observed end-to-end time is reported separately, without hiding it.
    faults = {}
    deadline = began + (1600 if any('fault_rank' in c for c in cases) else 1250)
    try:
        while time.monotonic() < deadline:
            for index, (customer, row, case) in enumerate(zip(customers, rows, cases)):
                update = customer.tick(row)
                result = measurements[row['id']]
                if update['status'] == 'finished':
                    if update['result']['status'] != 'completed':
                        raise ValueError('The frozen request did not complete: '+case['id'])
                    if result['settled_seconds'] is None:
                        result.update(settled_seconds=time.monotonic()-began, response=update['result'])
                    continue
                snapshot = update['snapshot']
                lease, epoch = snapshot['lease'], snapshot['lease']['assignment_root']
                if 'fault_rank' in case and row['id'] not in faults and lease['status'] == 'ready':
                    lost = next(p for p in providers if p['owner'] == lease['providers'][str(case['fault_rank'])]['owner'])
                    cloud.command(lost['physical'], ['sudo', 'systemctl', 'stop', lost['unit']])
                    faults[row['id']] = {'lost': lost, 'epoch': epoch, 'at_seconds': time.monotonic()-began,
                                         'deadline': lease['work_deadline'], 'replacement': None}
                fault = faults.get(row['id'])
                if fault and fault['replacement'] is None and epoch != fault['epoch']:
                    replacement = lease['providers'][str(case['fault_rank'])]
                    if replacement['owner'] == fault['lost']['owner']:
                        raise ValueError('Native recovery selected the lost owner again')
                    if identity(snapshot['job']['request']) != row['quote']['request_root']:
                        raise ValueError('Automatic recovery changed the paid request')
                    fault['replacement'] = replacement
                    fault['new_assignment'] = epoch
                    save(report_home/'cases'/case['id']/'fault.json', fault)
                coordinator = lease['providers']['0']
                stream = streams.get(row['id'])
                if stream is None or stream['epoch'] != epoch:
                    if stream:
                        stream['connection'].close()
                    stream = {'epoch': epoch, 'after': 0, 'connection': provider_wire.PinnedConnection(
                        coordinator['endpoint'], coordinator['certificate'], timeout=1)}
                    streams[row['id']] = stream
                try:
                    event = provider_wire.poll(stream['connection'], customer.wallet, coordinator['owner'],
                        network.genesis['chain_id'], row['job_id'], epoch, stream['after'])
                    if event is not None:
                        if any(event[key] != row['quote'][key] for key in ('graph', 'tokenizer', 'request_root')):
                            raise RuntimeError('The streamed response changed its native commitment')
                        stream['after'] = event['sequence']
                        elapsed = time.monotonic()-assigned[row['id']]
                        result['events'].append({'seconds': elapsed, **event})
                        if event['text'] and result['first_visible_seconds'] is None:
                            result['first_visible_seconds'] = elapsed
                        if event['status'] in ('generated', 'submitted') and result['generated_seconds'] is None:
                            result['generated_seconds'] = elapsed
                except (OSError, ValueError):
                    stream['connection'].close()
            if all(row['phase'] == 'finished' for row in rows):
                break
            time.sleep(.1)
        else:
            raise TimeoutError('Frozen provider request exceeded its settlement bound')
    finally:
        for stream in streams.values():
            stream['connection'].close()
        for fault in faults.values():
            lost = fault['lost']
            cloud.command(lost['physical'], ['sudo', 'systemctl', 'start', lost['unit']])
        for row, case in zip(rows, cases):
            save(report_home/'cases'/case['id']/'measurement.json', measurements[row['id']])
    for customer, row, case in zip(customers, rows, cases):
        # A different validator may still be one block behind the customer's
        # completed receipt. Read subsequent history from that same pinned
        # full node; lag elsewhere is not a missing or duplicate settlement.
        history = customer.node.query('/hosting')['history']
        audits = customer.node.query('/auditing')['history']
        matches = [value for value in history if value['job_id'] == row['job_id']]
        if len(matches) != 1:
            raise ValueError('A hosted request did not settle exactly once')
        response = measurements[row['id']]['response']
        if response['paid_atoms'] + response['refunded_atoms'] != row['quote']['execution_atoms']:
            raise ValueError('Execution settlement lost or duplicated prepaid funds')
        audited = [value for value in audits if value['id'] == row['budget_id']]
        if len(audited) != 1 or audited[0]['paid_atoms'] + audited[0]['refunded_atoms'] != row['quote']['verification_atoms']:
            raise ValueError('Complete verification failed to refund every unused atom')
        save(report_home/'cases'/case['id']/'hosting.json', matches[0])
        save(report_home/'cases'/case['id']/'audit-payment.json', audited[0])
        if 'fault_rank' in case and (row['id'] not in faults or faults[row['id']]['replacement'] is None):
            raise ValueError('The declared owner-loss recovery was not actually exercised')
    return list(measurements.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', required=True, type=Path)
    parser.add_argument('--report-home', type=Path,
                        help='Separate evidence directory for an explicitly frozen deployment repair')
    args = parser.parse_args()
    home = args.home.resolve()
    report_home = args.report_home.resolve() if args.report_home else home
    if (report_home/'release-gate-started.json').exists():
        raise ValueError('A gate already started; inspect its evidence, do not silently rerun it')
    cloud = Cloud(home)
    network = Network(home/'native')  # RPC tunnels belong to the running controller.
    original = read(ROOT/'config/experiments/provider-llm-service.json')
    freeze = read(home/'freeze.json')
    save(report_home/'release-gate-started.json', {'time': time.time(), 'freeze': identity(freeze),
        'deployment': str(home), 'resource_amendment': read(home/'resource-amendment.json')
        if (home/'resource-amendment.json').exists() else None})
    providers = [read(home/'providers'/f'{i}.json') for i in range(18)]
    boxes, customers = [], []
    node = LocalNode(network.urls[-1], network.genesis['chain_id'], identity(network.genesis['app_state']['manifest']))
    for i in range(2):
        folder = home/f'customer-{i}'
        wallet = wire.Wallet(folder/'account.key')
        box = Outbox(folder/'hosted-chat.sqlite', node.url, node.chain_id, wallet)
        boxes.append(box)
        customers.append(Customer(folder/'hosted-requests', node, wallet, box))
    result = {'passed': False, 'warmup': [], 'ordinary': [], 'faults': []}
    try:
        warmup = [{'id': 'warmup-'+str(i), 'messages': original['warmup']['messages']} for i in range(2)]
        result['warmup'] = batch(cloud, network, providers, customers, warmup, 4, report_home=report_home)
        for start in range(0, 6, 2):
            result['ordinary'].extend(batch(cloud, network, providers, customers,
                                           original['requests'][start:start+2], 64, report_home=report_home))
        for row in result['ordinary']:
            if (row['first_visible_seconds'] is None or row['generated_seconds'] is None
                    or row['first_visible_seconds'] > freeze['release_gate']['warm_first_visible_p95_seconds']
                    or row['generated_seconds'] > freeze['release_gate']['warm_generation_p95_seconds']
                    or row['settled_seconds'] > freeze['release_gate']['settlement_seconds_per_request']):
                raise ValueError('The unchanged six-case service gate failed')
        # A known ordinary-latency failure cannot be rescued by lengthy fault
        # trials. Keep its measurements and stop before reserving more work.
        for case in original['requests'][6:]:
            result['faults'].extend(batch(cloud, network, providers, customers[:1], [case], 64,
                                         report_home=report_home))
        if any(r['settled_seconds'] > freeze['release_gate']['recovered_settlement_seconds'] for r in result['faults']):
            raise ValueError('Automatic recovery exceeded the declared settlement deadline')
        if network.query()['issued'] != 0:
            raise ValueError('Serving unexpectedly issued tokens')
        result.update(passed=True, issued_atoms=0, finished_at=time.time(), height=network.query()['height'])
    finally:
        save(report_home/'release-gate.json', result)
        for box in boxes:
            box.close()
        network.close()
    print(json.dumps({'passed': True, 'ordinary': 6, 'automatic_recoveries': 2, 'issued_atoms': 0}))


if __name__ == '__main__':
    main()
