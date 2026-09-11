#!/usr/bin/env python3
"""Resume funded full-model training, evaluation and serving on an existing chain.

This operated candidate uses keys for the operator's own worker group. Data
admission votes and model growth are separate explicit actions. It never changes
genesis, creates an independent operator identity, or promises model improvement.
"""
import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from neuroshard.dataflow.store import canonical
from neuroshard.demo import protocol, client as wire
from neuroshard.evolution import cohorts, forward
from neuroshard.evolution.app import code_hash
from neuroshard.evolution.audit_worker import required_objects
from neuroshard.evolution.model import place
from neuroshard.evolution.objects import Objects, digest
from neuroshard.evolution.pipeline import Pipeline, LocalEndpoint
from neuroshard.evolution.transactions import Outbox
from neuroshard.evolution.transport import Endpoint
from neuroshard.evolution.verification import bundle
from neuroshard.evolution.worker import Worker


def inference_price_floor(manifest, partitions, auditor_count):
    """Audit and publisher transaction cost for a response ending at one token.

    Each token has one forward trace per partition plus the output head.
    This lower bound excludes compute, storage and a provider margin.
    """
    return ((partitions+1)*auditor_count*manifest['auditing']['price_per_stage']
            + 2*manifest['params']['fee'])


class Operator:
    def __init__(self, config_path):
        self.config = json.loads(config_path.read_bytes())
        base = config_path.resolve().parent
        def path(value):
            return base / Path(value).expanduser()
        c = self.config
        self.home = path(c['home'])
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.home/'operator.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.url = c['rpc']
        genesis = wire.rpc(self.url, 'genesis')['genesis']
        if digest(canonical(genesis)) != c['genesis_sha256']:
            raise ValueError('Operator genesis differs from configured commitment')
        self.manifest = genesis['app_state']['manifest']
        self.chain_id = genesis['chain_id']
        if self.manifest['code_hash'] != code_hash() or 'auditing' not in self.manifest or 'lifecycle' not in self.manifest:
            raise ValueError('Operator requires the matching funded lifecycle source and genesis')
        self.owner = protocol.Identity.load_or_create(path(c['key']))
        self.worker_keys = [protocol.Identity.load_or_create(path(key)) for key in c['worker_keys']]
        self.auditors = sorted(c['auditors'])
        self.store = Objects(path(c['objects']))
        self.capacities = [w['capacity'] for w in c['workers']]
        self.endpoints = []
        for index, worker in enumerate(c['workers']):
            if 'url' in worker:
                endpoint = Endpoint(worker['url'], path(worker['token_file']).read_text().strip(), self.store)
            else:
                endpoint = LocalEndpoint(Worker(self.home/f'worker{index}', self.store, worker['capacity']))
            self.endpoints.append(endpoint)
        if len(self.worker_keys) != len(self.endpoints):
            raise ValueError('Provide exactly one owned signing key for each configured worker')
        self.outbox = Outbox(self.home/'outbox.sqlite', self.url, self.chain_id, self.owner)
        budget = c['budget']
        if (type(budget['training_round_limit']) is not int or budget['training_round_limit'] < 1
                or type(budget['minimum_balance']) is not int or budget['minimum_balance'] < 0):
            raise ValueError('Set a persistent absolute training-round cap and minimum liquid reserve')
        if type(budget.get('allow_inference_subsidy', False)) is not bool:
            raise ValueError('An inference subsidy requires an explicit boolean policy')
        limit = budget.get('inference_token_limit', 1)
        if type(limit) is not int or not 0 <= limit <= 8:
            raise ValueError('Set an inference token limit between zero and eight; zero disables serving')

    def send(self, operation, kind, **fields):
        self.outbox.send(operation, kind, **fields)
        return self.outbox.logical_id(operation)

    def query(self, path='/status', options=None):
        return wire.query(self.url, path, options)

    def budget(self, action, count):
        key = self.send('fund:'+action, 'fund_audit', publisher=self.owner.public_key,
                        auditors=self.auditors, stage_limit=count, expires_in=1024)
        budget = self.query('/auditing')['budgets'].get(key)
        if budget is None:
            raise ValueError('This action exhausted its audit offer; inspect its retained outcome before retrying')
        return key if all(a['bond'] for a in budget['auditors'].values()) else None

    def pipe(self, root, name, step=0):
        count = len(place(self.store.json(root), self.capacities))
        return Pipeline(self.store, root, self.endpoints[:count], self.capacities,
                        self.chain_id+':'+name, learning_rate=self.manifest['learning_rate'],
                        clip_norm=self.manifest['clip_norm'], start_step=step)

    def publish(self, kind, record):
        values = bundle(self.store, record['record_root']) if kind == 'training' else forward.bundle(self.store, record['record_root'])
        claim = {'kind':kind, 'record_root':record['record_root'], 'model_root':record['model_root'], 'metadata':values}
        for key in required_objects(claim):
            self.store.get(key)
        return values

    def tick(self):
        pending = self.outbox.pending()
        if pending:
            self.outbox.confirm(pending)
        status = self.query()
        if status['chain_id'] != self.chain_id:
            raise ValueError('RPC chain changed')
        if status['candidate']:
            return {'phase':'waiting_for_funded_settlement', 'claim':status['candidate']['id']}
        account = self.query('/account', {'public_key':self.owner.public_key})
        if account['balance'] < self.config['budget']['minimum_balance']:
            return {'phase':'waiting_for_funding', 'balance':account['balance']}
        if shutil.disk_usage(self.store.root).free < 2*1024**3:
            raise OSError('Operator artifact store has less than 2 GiB free')
        data, evaluation = self.query('/data'), self.query('/evaluation')
        reservation = status['assignment']
        if reservation:
            if reservation['owner'] != self.owner.public_key:
                return {'phase':'another_publisher_reserved_training'}
            root, step = status['model_root'], status['training_round']
            count = len(place(self.store.json(root), self.capacities))
            if reservation['workers'] != [k.public_key for k in self.worker_keys[:count]]:
                raise ValueError('Reserved worker identities differ from this operator group')
            pipe = self.pipe(root, reservation['id'], step)
            try:
                # Reopening this reservation reuses the workers' durable
                # operation results, including a completed but unclaimed step.
                record = pipe.train(data['batches'][reservation['batch']])
            finally:
                pipe.close()
            values = self.publish('training', record)
            receipts = [key.sign({'domain':'neuroshard/evolution/work/v1', 'chain_id':self.chain_id,
                'assignment':reservation['id'], 'record_root':record['record_root'], 'stage':i,
                'trace_root':record['traces'][i]}) for i,key in enumerate(self.worker_keys[:count])]
            self.send('claim:'+reservation['id'], 'claim', record_root=record['record_root'], metadata=values,
                      workers=receipts, data_root=reservation['data_root'], sequence_index=reservation['sequence_index'])
            return {'phase':'training_claimed', 'round':step, 'record_root':record['record_root']}
        if evaluation:
            for side in ('baseline','candidate'):
                for role in ('retention','fresh'):
                    offset = len(evaluation['measurements'][side][role])
                    if offset >= len(cohorts.evaluation_rows(data, role)):
                        continue
                    action = f'score:{evaluation["id"]}:{side}:{role}:{offset}'
                    root = evaluation[side]
                    count = len(place(self.store.json(root), self.capacities))+1
                    budget = self.budget(action, count)
                    if budget is None:
                        return {'phase':'waiting_for_auditor_acceptance'}
                    _, batch = cohorts.evaluation_batch(data, role, offset)
                    pipe = self.pipe(root, action)
                    try: record = pipe.evaluate_record(batch)
                    finally: pipe.close()
                    values = self.publish('score', record)
                    self.send(action, 'score', evaluation_id=evaluation['id'], side=side, role=role, offset=offset,
                              record_root=record['record_root'], metadata=values, audit_budget=budget)
                    return {'phase':'evaluation_claimed', 'side':side, 'role':role, 'offset':offset}
            self.send('finish:'+evaluation['id'], 'finish_evaluation', evaluation_id=evaluation['id'])
            return {'phase':'evaluation_finished'}
        if data and not data['closed']:
            if data['step'] == len(data['schedule']):
                self.send('evaluate:'+data['root']+':'+status['model_root'], 'open_evaluation')
                return {'phase':'evaluation_opened'}
            if status['training_round'] >= self.config['budget']['training_round_limit']:
                return {'phase':'training_budget_complete', 'round':status['training_round']}
            action = f'train:{data["root"]}:{data["step"]}:{status["training_round"]}:{status["model_root"]}'
            count = len(place(self.store.json(status['model_root']), self.capacities))
            budget = self.budget(action, count)
            if budget is None:
                return {'phase':'waiting_for_auditor_acceptance'}
            self.send('reserve:'+action, 'reserve', parent=status['model_root'], round=status['training_round'],
                      workers=[k.public_key for k in self.worker_keys[:count]], audit_budget=budget)
            return {'phase':'training_reserved', 'round':status['training_round']}
        jobs = self.query('/inference')['jobs']
        underpriced = []
        oversized = []
        for key, job in sorted(jobs.items(), key=lambda item:(item[1]['expires'],item[0])):
            if job['provider'] != self.owner.public_key or job['claim_id'] or status['height'] > job['expires']:
                continue
            limit = self.config['budget'].get('inference_token_limit', 1)
            if job['max_tokens'] > limit:
                oversized.append({'job':key, 'requested_tokens':job['max_tokens'], 'operator_token_limit':limit})
                continue
            root = job['model_root']
            partitions = len(place(self.store.json(root), self.capacities))
            floor = inference_price_floor(self.manifest, partitions, len(self.auditors))
            if job['unit_price'] < floor and not self.config['budget'].get('allow_inference_subsidy', False):
                underpriced.append({'job':key, 'unit_price':job['unit_price'], 'minimum_cost_per_token':floor})
                continue
            count = (partitions+1)*job['max_tokens']
            action = 'respond:'+key
            budget = self.budget(action, count)
            if budget is None:
                return {'phase':'waiting_for_auditor_acceptance'}
            pipe = self.pipe(root, action)
            try: record = pipe.generate_record(job['prompt_ids'], job['max_tokens'], job['eos_ids'])
            finally: pipe.close()
            values = self.publish('inference', record)
            self.send(action, 'respond', job_id=key, record_root=record['record_root'], metadata=values, audit_budget=budget)
            return {'phase':'inference_claimed', 'job':key}
        if underpriced:
            return {'phase':'inference_requires_explicit_subsidy_or_new_price_profile', 'jobs':underpriced}
        if oversized:
            return {'phase':'inference_exceeds_operator_capacity_policy', 'jobs':oversized}
        return {'phase':'waiting_for_admitted_data_or_inference', 'round':status['training_round']}

    def close(self):
        self.outbox.close()
        self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    operator = Operator(args.config)
    try:
        while True:
            result = operator.tick()
            print(json.dumps(result), flush=True)
            if args.once:
                break
            time.sleep(1)
    finally:
        operator.close()


if __name__ == '__main__':
    main()
