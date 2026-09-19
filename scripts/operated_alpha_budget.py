"""Conservative cost watch for only the alpha's recorded AWS resources."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import math

import boto3
from neuroshard.evolution.reference_data import save
from ordinary_cloud import PROTECTED


def observe(home):
    home = Path(home)
    freeze = json.loads((home/'freeze.json').read_bytes())
    path = home/'cost-watch.json'
    previous = json.loads(path.read_bytes()) if path.exists() else None
    if previous and time.time()-previous['observed_at'] < 300:
        return previous
    gpu = json.loads((home/'allocation.json').read_bytes())
    ledger = json.loads((home/'ledger-hosts/allocation.json').read_bytes())
    rows = [*gpu['instances'], *ledger['instances']]
    if len(rows) != 11 or any(row['InstanceId'] in PROTECTED for row in rows):
        raise ValueError('Cost watch accepts only the eleven recorded alpha hosts')
    cloudwatch = boto3.client('cloudwatch', region_name=freeze['gpu_resources']['region'])
    end = datetime.now(timezone.utc)
    total_bytes = 0
    observations = {}
    for row in rows:
        start = datetime.fromisoformat(row['LaunchTime'])
        points = cloudwatch.get_metric_statistics(Namespace='AWS/EC2', MetricName='NetworkOut',
            Dimensions=[{'Name': 'InstanceId', 'Value': row['InstanceId']}], StartTime=start,
            EndTime=end, Period=300*max(1, math.ceil((end-start).total_seconds()/(300*1400))),
            Statistics=['Sum'])['Datapoints']
        amount = sum(point['Sum'] for point in points)
        # Never lower the cumulative estimate after a temporarily sparse AWS reply.
        amount = max(amount, (previous or {}).get('network_out', {}).get(row['InstanceId'], 0))
        observations[row['InstanceId']] = amount
        total_bytes += amount
    gpu_hours, ledger_hours = freeze['gpu_resources']['max_hours'], freeze['ledger_resources']['hours']
    fixed = {
        'prior_failed_deployment_reserve': freeze['funding'].get('prior_deployment_full_upper_reserve_usd', 0),
        'complete_gpu_window': 7*gpu_hours*freeze['gpu_resources']['verified_instance_hourly_usd'],
        'complete_ledger_window': 4*ledger_hours*freeze['ledger_resources']['verified_instance_hourly_usd'],
        'gpu_gp3_with_provisioned_performance': 7*(200*.08 + 9000*.005 + 375*.04)*gpu_hours/730,
        'ledger_gp3': 4*40*.08*ledger_hours/730,
        'all_public_ipv4': (7*gpu_hours+4*ledger_hours)*.005,
        'whole_shared_controller': .192*gpu_hours,
        'initial_model_delivery_metadata_requests_and_retention_reserve': 50.0,
    }
    # Charge all observed egress at a conservative public-transfer rate, even
    # though much of it is cheaper inter-zone or free same-zone traffic.
    transfer = total_bytes/1_000_000_000*.10
    total = sum(fixed.values())+transfer
    cap = freeze['funding']['aggregate_new_allocation_ceiling_usd']
    result = {'observed_at': time.time(), 'network_out': observations, 'fixed_complete_window': fixed,
        'observed_network_out_bytes': total_bytes, 'transfer_upper_estimate_usd': transfer,
        'complete_window_plus_observed_transfers_usd': total, 'planning_ceiling_usd': cap,
        'close_future_capacity': total >= cap-50,
        'scope': 'Conservative planning estimate, not an AWS invoice. Includes complete declared runtimes in advance. '
                 'Metrics may arrive late; retain a USD 50 drain reserve. Existing retained model objects are unchanged.'}
    save(path, result)
    return result
