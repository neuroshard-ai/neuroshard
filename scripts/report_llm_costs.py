"""Reconcile finite learning/serving sponsorship from public cost evidence.

This is offline accounting, not a billing API or a NEURO exchange-rate oracle.
Input files are allowlisted research records; never point it at private node or
wallet backups. Allocation compute already includes all numerical replays.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def report(home, expected_provider_attempts):
    plan, prices = read(home/'sponsorship.json'), read(home/'prices.json')['rates']
    learning = read(home/'learning-allocations.json')
    if set(learning) != set(plan['scope']['learning_allocations']):
        raise ValueError('Include every declared learning allocation')
    allocated, learning_costs = set(), {}
    for name, row in learning.items():
        allocation = read(home/name/'allocation.json')
        owners = {host['InstanceId'] for host in allocation['instances']}
        if allocated & owners or len(owners) != 7:
            raise ValueError('Do not count a reused allocation twice')
        allocated |= owners
        if not row['all_metrics_present'] or row['interior_gaps']:
            raise ValueError('Complete learning network measurements are required')
        metrics = row['metrics']
        if len(metrics) != 2*len(owners) or {(value['instance_id'], value['metric']) for value in metrics} != {
                (owner, metric) for owner in owners for metric in ('NetworkIn', 'NetworkOut')}:
            raise ValueError('Missing or duplicated learning traffic series')
        for value in metrics:
            raw = read(home/name/(value['instance_id']+'-'+value['metric']+'.json'))
            if sum(point['Sum'] for point in raw['raw']['Datapoints']) != value['bytes']:
                raise ValueError('Learning traffic differs from raw CloudWatch samples')
        if sum(value['bytes'] for value in metrics) != row['traffic_bytes']:
            raise ValueError('Learning traffic totals differ from measurements')
        finished = read(home/name/'finished.json')
        seconds = (datetime.fromisoformat(finished.get('finished', finished.get('checked_at')))
                   - datetime.fromisoformat(allocation['created'])).total_seconds()
        if seconds != row['seconds'] or allocation['hourly_rate_usd'] != prices['g5.xlarge_hour']:
            raise ValueError('Allocation duration or GPU price changed')
        month, hours = seconds/(28*24*3600), seconds/3600
        # All three frozen learning allocations declare 12,000 IOPS and
        # 500 MiB/s gp3; capacity is recorded in their allocation manifest.
        computed = len(owners)*(hours*(prices['g5.xlarge_hour']+prices['public_ipv4_hour'])
            + month*(allocation['resources']['disk_gib']*prices['gp3_gib_month']
                     + 9000*prices['gp3_iops_month_above_3000']
                     + 375*prices['gp3_mib_per_second_month_above_125']))
        computed += row['traffic_bytes']/1e9*prices['every_network_in_and_out_decimal_gb_upper']
        if not math.isclose(computed, sum(row['usd_upper'].values()), abs_tol=1e-8):
            raise ValueError('Learning prices do not reproduce the original report')
        learning_costs[name] = computed

    serving, separate_caps = {}, {}
    for path in sorted((home/'provider-costs').glob('*.json')):
        row = read(path)
        if not row['all_host_counters_present']:
            raise ValueError('Complete provider network measurements are required')
        parts = row['usd_upper_estimates']
        # A prior-attempt reserve is not an additional bill. Retention is priced
        # once below against the whole public pool, including failed research.
        serving[path.stem] = sum(value for name, value in parts.items()
            if name not in ('prior_failed_allocation_allowance', 'ninety_day_retention'))
        separate_caps[path.stem] = {'reported_with_prior_reserve': row['total_usd_upper'],
            'original_cap': row['sponsor_cap_usd'],
            'within_original_cap': row['total_usd_upper'] <= row['sponsor_cap_usd']}
    if len(serving) != expected_provider_attempts or expected_provider_attempts < 1:
        raise ValueError('Include every declared provider attempt, including failures')

    controller = read(home/'controllers.json')
    raw_controller = read(home/'controller-network.json')
    controller_owners = {host['instance_id'] for host in controller['hosts']}
    if (len(controller_owners) != len(controller['hosts'])
            or len(raw_controller) != 2*len(controller_owners)
            or {(row['instance_id'], row['metric']) for row in raw_controller} != {
                (owner, metric) for owner in controller_owners for metric in ('NetworkIn', 'NetworkOut')}):
        raise ValueError('Missing or duplicated controller traffic series')
    for row in raw_controller:
        if (not row['raw']['Datapoints']
                or row['accounted_through'] != controller['end']
                or sum(point['Sum'] for point in row['raw']['Datapoints']) != row['bytes']):
            raise ValueError('Controller traffic differs from raw CloudWatch samples')
    if sum(row['bytes'] for row in raw_controller) != controller['traffic_bytes']:
        raise ValueError('Controller traffic totals differ from measurements')
    seconds = (datetime.fromisoformat(controller['end'])
               - datetime.fromisoformat(plan['measurement_start_utc'])).total_seconds()
    if seconds <= 0 or not controller['complete_series']:
        raise ValueError('Complete controller measurements are required')
    month, hours = seconds/(28*24*3600), seconds/3600
    controller_parts = {'compute': 0, 'disks': 0, 'ipv4': 0,
        'network': controller['traffic_bytes']/1e9*prices['every_network_in_and_out_decimal_gb_upper']}
    for host in controller['hosts']:
        if host['instance_id'] in allocated or host['type'] != 'm5.xlarge':
            raise ValueError('Unexpected or duplicated controller')
        controller_parts['compute'] += hours*prices['m5.xlarge_hour']
        controller_parts['ipv4'] += hours*prices['public_ipv4_hour']*host['public_ipv4']
        for volume in host['volumes']:
            if volume['VolumeType'] != 'gp3':
                raise ValueError('Unknown controller volume pricing')
            controller_parts['disks'] += month*(volume['Size']*prices['gp3_gib_month']
                + max(0, volume['Iops']-3000)*prices['gp3_iops_month_above_3000']
                + max(0, volume['Throughput']-125)*prices['gp3_mib_per_second_month_above_125'])

    retained = read(home/'retention.json')
    if (retained['count'] != len(retained['objects']) or retained['total_bytes']
            != sum(row['bytes'] for row in retained['objects'].values())
            or any(row['storage_class'] != 'STANDARD' for row in retained['objects'].values())):
        raise ValueError('Require a consistent Standard-storage inventory')
    # Use the same conservative 28-day billing-month denominator as disks.
    retained_bytes = retained['total_bytes'] + plan['scope']['archive_size_allowance_bytes']
    retention = retained_bytes/1e9*prices['s3_standard_decimal_gb_month']*93/28
    parts = {'learning_allocations': sum(learning_costs.values()),
        'provider_allocations_without_duplicate_retention': sum(serving.values()),
        'controllers': sum(controller_parts.values()), 'public_retention_93_days': retention,
        'additional_request_allowance': plan['scope']['requests_allowance_usd']}
    total = sum(parts.values())
    return {'format': 'neuroshard-complete-finite-cost-report-v1',
        'start': plan['measurement_start_utc'], 'end': controller['end'],
        'learning': learning_costs, 'serving': serving, 'controller_parts': controller_parts,
        'parts': parts, 'total_usd_upper': total, 'sponsor_cap_usd': plan['usd_cap'],
        'within_aggregate_cap': total <= plan['usd_cap'],
        'provider_attempts': len(serving),
        'provider_original_caps': separate_caps, 'retained_bytes_with_archive_allowance': retained_bytes,
        'limitations': ['Price-based pre-tax estimate, not billed usage or remaining credit.',
            'All measured incoming and outgoing traffic is charged, including free traffic.',
            'Controller footprint is assumed constant at its recorded provisioned size; shared older services are included.',
            'S3 request charges use a declared allowance, not billing measurements.',
            'Inherited checkpoints, future work, unlimited downloads and perpetual retention are outside this finite scope.',
            'An aggregate pass cannot change an earlier experiment cap or result.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-provider-attempts', type=int, required=True)
    args = parser.parse_args()
    value = report(args.evidence, args.expected_provider_attempts)
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({'total_usd_upper': value['total_usd_upper'],
        'within_aggregate_cap': value['within_aggregate_cap']}))
