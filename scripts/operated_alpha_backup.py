#!/usr/bin/env python3
"""Private alpha recovery copies. Nothing written here is public evidence.

Wallet copies preserve assigned provider identities and consistent transaction
journals. The scheduled final ledger copy stops its four recorded nodes before
copying validator signing state and block stores, then retires those hosts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from neuroshard.evolution.reference_data import save
from ordinary_cloud import Cloud, REMOTE
from operated_alpha_hosts import LedgerHosts, retire


def write_private(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix+'.pending')
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def wallets(home):
    if (home/'resources-finished.json').exists():
        return
    cloud = Cloud(home)
    rows = [json.loads(p.read_bytes()) for p in sorted((home/'providers').glob('*.json'))]
    code = '''import io,pathlib,sqlite3,sys,tarfile,tempfile
p=pathlib.Path(sys.argv[1]); output=io.BytesIO()
with tempfile.TemporaryDirectory() as temporary:
 journal=pathlib.Path(temporary)/'transactions.sqlite'
 original=sqlite3.connect((p/'transactions.sqlite').as_uri()+'?mode=ro',uri=True)
 target=sqlite3.connect(journal); original.backup(target); target.close(); original.close()
 with tarfile.open(fileobj=output,mode='w:gz') as archive:
  for name in ('identity','config.json','transport-cert.pem','transport-key.pem'):
   path=p/name
   if path.exists():
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<1024**2
    archive.add(path,arcname=name)
  assert (p/'identity').is_file() and journal.stat().st_size<128*1024**2
  archive.add(journal,arcname='transactions.sqlite')
sys.stdout.buffer.write(output.getvalue())
'''
    def one(row):
        raw = cloud.command(row['physical'], ['python3', '-c', code, row['home']], timeout=90).stdout
        write_private(home/'private-recovery'/'providers'/f'{row["index"]}.tar.gz', raw)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, rows))
    save(home/'private-recovery/provider-copy.json', {'providers': len(rows),
        'time': datetime.now(timezone.utc).isoformat(), 'private': True})


def final_ledger(home, *, drained=False):
    hosts = LedgerHosts(home/'ledger-hosts')
    if drained:
        # Failed, unpublished deployments may finish their existing native
        # refunds after GPU retirement. Do not destroy an outstanding budget.
        expected = json.loads((home/'access.json').read_bytes())
        code = ('import json; from neuroshard.demo import client; u="http://127.0.0.1:26657"; '
                'print(json.dumps({p:client.query(u,p) for p in '
                '["/status","/hosting","/auditing","/service_admission"]}))')
        view = json.loads(hosts.command(0, ['env', 'PYTHONPATH=/home/ubuntu/neuroshard-study/src',
            '/home/ubuntu/neuroshard-study/.neuroshard/native/bin/python', '-c', code]).stdout)
        if (view['/status']['chain_id'] != expected['chain_id'] or view['/hosting']['leases']
                or view['/auditing']['budgets'] or view['/service_admission']['services']):
            raise ValueError('Keep the ledger available until every old obligation has closed')
    elif hosts.remaining(margin=0) > 3600:
        raise ValueError('Final ledger retirement is limited to the last funded hour')
    # Quiesce signing before backup. Copying live CometBFT databases or stale
    # private-validator state is not a safe consensus recovery procedure.
    def stop(index):
        units = ['neuroshard-alpha-node', 'neuroshard-alpha-app']
        if index == 0:
            units += ['neuroshard-alpha-credits', 'neuroshard-alpha-recovery']
        hosts.command(index, ['sudo', 'systemctl', 'stop', *units], timeout=90)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(stop, range(4)))
    for index in range(4):
        path = home/'private-recovery'/'ledger'/f'{index}.tar.gz'
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix('.pending')
        folders = ['native'] + (['starter-credits', 'recovery'] if index == 0 else [])
        command = shlex.join(['tar', '-czf', '-', '-C', REMOTE, *folders])
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'wb') as output:
            subprocess.run([*hosts.ssh(index), command], stdout=output, stderr=subprocess.PIPE,
                           timeout=min(600, hosts.remaining()), check=True)
            output.flush()
            os.fsync(output.fileno())
        with tarfile.open(temporary) as archive:
            names = set(archive.getnames())
            if not {'native/config/priv_validator_key.json', 'native/data/priv_validator_state.json',
                    'native/config/genesis.json', 'native/evolution.sqlite'} <= names:
                raise ValueError('Private ledger backup lacks signing or application recovery state')
        temporary.replace(path)
    save(home/'private-recovery/ledger-copy.json', {'validators': 4,
        'time': datetime.now(timezone.utc).isoformat(), 'private': True, 'signing_stopped_before_copy': True})
    retire(home/'ledger-hosts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['wallets', 'final-ledger', 'drained-ledger'])
    parser.add_argument('--home', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'wallets':
        wallets(args.home.resolve())
    else:
        final_ledger(args.home.resolve(), drained=args.action == 'drained-ledger')
