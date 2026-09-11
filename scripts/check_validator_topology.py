#!/usr/bin/env python3
"""Check native quorum against declared host/operator failure domains.

Inventory labels are operator declarations, not cryptographic independence
proofs. This check covers quorum topology, not overall production readiness.
"""
import argparse
import base64
import json
import re
from pathlib import Path

import requests


def assess(validators, inventory):
    if not isinstance(inventory,dict) or set(inventory) != {'validators'} or not isinstance(inventory['validators'],list):
        raise ValueError('Inventory requires a validators list')
    declared = {}
    for row in inventory['validators']:
        if not isinstance(row,dict) or set(row) != {'consensus_key','host','operator'}:
            raise ValueError('Each validator needs a consensus key, host and operator')
        key = row['consensus_key']
        if not isinstance(key,str) or re.fullmatch('[0-9a-f]{64}',key) is None or key in declared:
            raise ValueError('Invalid or duplicate inventory consensus key')
        if any(not isinstance(row[field],str) or not 1 <= len(row[field]) <= 128 for field in ('host','operator')):
            raise ValueError('Failure-domain labels must be bounded nonempty strings')
        declared[key] = row
    if not isinstance(validators,dict) or not validators or set(validators) != set(declared):
        raise ValueError('Inventory must cover exactly the current voting validator keys')
    if any(type(power) is not int or power <= 0 for power in validators.values()):
        raise ValueError('Voting powers must be positive integers')
    total = sum(validators.values())
    failures = []
    for field in ('host','operator'):
        groups = {}
        for key,power in validators.items():
            label = declared[key][field]
            groups[label] = groups.get(label,0)+power
        for label,power in sorted(groups.items()):
            remaining = total-power
            failures.append({'domain':field,'label':label,'lost_power':power,'remaining_power':remaining,
                             'can_finalize':3*remaining > 2*total})
    return {'validators':len(validators),'total_voting_power':total,
            'tolerates_each_single_declared_domain_failure':all(row['can_finalize'] for row in failures),
            'failure_cases':failures,
            'scope':'declared host/operator quorum only; labels do not prove independent ownership or production readiness'}


def powers(rpc):
    if not isinstance(rpc,str) or not rpc.startswith(('http://','https://')):
        raise ValueError('Use a trusted native node HTTP(S) RPC endpoint')
    session = requests.Session()
    session.trust_env = False
    result, height, total, page = {}, None, None, 1
    try:
        while total is None or len(result) < total:
            params = {'page':str(page),'per_page':'100'}
            if height is not None:params['height'] = height
            response = session.post(rpc,json={'jsonrpc':'2.0','id':1,'method':'validators','params':params},timeout=15)
            response.raise_for_status()
            data = response.json()
            if 'error' in data:raise ValueError('Native validator query failed')
            data = data['result']
            returned_height = str(data['block_height'])
            if height is not None and returned_height != height:
                raise ValueError('Validator query changed heights during pagination')
            height = returned_height
            count = int(data['total'])
            if not 1 <= count <= 4096 or (total is not None and count != total):
                raise ValueError('Invalid or changing validator count')
            total = count
            if not data['validators']:raise ValueError('Incomplete validator pagination')
            for validator in data['validators']:
                if validator['pub_key']['type'] != 'tendermint/PubKeyEd25519':
                    raise ValueError('Unsupported native validator key type')
                raw = base64.b64decode(validator['pub_key']['value'],validate=True)
                key = raw.hex()
                power = int(validator['voting_power'])
                if len(raw) != 32 or power <= 0 or key in result:
                    raise ValueError('Invalid or repeated native validator')
                result[key] = power
            if len(result) > total:raise ValueError('Native validator count mismatch')
            page += 1
        return result,height
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rpc',required=True,help='Your trusted local/full-node RPC URL')
    parser.add_argument('--inventory',type=Path,required=True)
    args = parser.parse_args()
    validators,height = powers(args.rpc)
    result = assess(validators,json.loads(args.inventory.read_bytes()))
    print(json.dumps({'height':height,**result},indent=2))
    raise SystemExit(0 if result['tolerates_each_single_declared_domain_failure'] else 2)


if __name__ == '__main__':main()
