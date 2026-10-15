"""Build the general-route candidate version: its released assets and its network descriptor.

    PYTHONPATH=src python scripts/build_general_release.py --a3-assets <dir> --units <dir> --router <file> \
        --out <dir> --chain-id neuroshard-assistant-release-2 --host 172.31.73.85 --assets-url file:///...

The candidate keeps everything A3 serves (U1, L2 drafting-upgrade L3, A3's router, policies and plan) and adds
two routes: documents (route 3, L9 on the unchanged parent, under documents5) and chat (route 4, C1 on the
unchanged parent, under chat-tools/1), with the top router ``general-router.json``. Every file is pinned by
size and SHA-256; the descriptor's model root names the whole version.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from neuroshard.assistant import network
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

A3 = 'src/neuroshard/client/networks/assistant-testnet.json'
VERSION = 'a4-general'
POLICIES = {'documents': 'config/experiments/assistant-general-policy-documents5.json',
            'chat': 'config/experiments/assistant-general-policy-chat3.json'}
UNITS = {'documents': ('L9', 3), 'chat': ('C1', 4)}


def pin(path, asset):
    data = Path(path).read_bytes()
    return {'asset': asset, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def build(a3_assets, units, router, out, chain_id, host, assets_url, ports):
    a3 = read(ROOT / A3)
    out = Path(out)
    (out / 'assets').mkdir(parents=True, exist_ok=True)
    files = dict(a3['files'])
    for name, item in a3['files'].items():
        source = Path(a3_assets) / item['asset']
        if pin(source, item['asset']) != item:
            raise ValueError(f'A3 asset {item["asset"]} differs from its pin')
        shutil.copy2(source, out / 'assets' / item['asset'])
    for route, (unit, _) in UNITS.items():
        for name in ('manifest.json', 'trainable.safetensors'):
            source = Path(units) / unit / name
            asset = f'{unit}-{name}'
            shutil.copy2(source, out / 'assets' / asset)
            files[f'{unit}/{name}'] = pin(source, asset)
        manifest = json.loads((Path(units) / unit / 'manifest.json').read_text())
        if manifest['arm'] != 'addition' or hashlib.sha256((Path(units) / unit / 'trainable.safetensors').read_bytes()
                                                           ).hexdigest() != manifest['trainable_sha256']:
            raise ValueError(f'{unit} differs from its manifest')
    shutil.copy2(router, out / 'assets' / 'general-router.json')
    files['general-router.json'] = pin(router, 'general-router.json')
    descriptor = {**a3, 'version': VERSION, 'chain_id': chain_id, 'rpc': f'http://{host}:{ports["p2p"] + 1}',
                  'faucet': f'http://{host}:{ports["faucet"]}/faucet', 'proof_endpoints': [], 'assets': assets_url,
                  'files': dict(sorted(files.items())),
                  'policies': {**a3['policies'], **{route: {'path': path, 'sha256': sha256(ROOT / path)}
                                                    for route, path in POLICIES.items()}},
                  'routing': {**a3['routing'],
                              **{route: {'route': index, 'base': 'parent', 'addition': unit}
                                 for route, (unit, index) in UNITS.items()},
                              'general_selector': 'general-router.json', 'general_routes': ['a3', 'chat', 'documents']}}
    served, _ = network.version(descriptor)
    if served.routes != {1: ('L3', 'U1'), 2: ('L2', 'U1'), 3: ('L9', 'parent'), 4: ('C1', 'parent')}:
        raise ValueError(f'unexpected route table {served.routes}')
    (out / 'network.json').write_text(json.dumps(descriptor, indent=2, sort_keys=True) + '\n')
    return descriptor, network.model_root(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('--a3-assets', '--units', '--router', '--out', '--chain-id', '--host', '--assets-url'):
        parser.add_argument(name, required=True)
    parser.add_argument('--p2p', type=int, default=28670)
    parser.add_argument('--faucet', type=int, default=28782)
    args = parser.parse_args(argv)
    descriptor, root = build(args.a3_assets, args.units, args.router, args.out, args.chain_id, args.host,
                             args.assets_url, {'p2p': args.p2p, 'faucet': args.faucet})
    print(json.dumps({'version': descriptor['version'], 'model_root': root, 'files': len(descriptor['files'])}))


if __name__ == '__main__':
    main()
