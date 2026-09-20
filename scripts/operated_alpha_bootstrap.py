"""Wait for a provider's own full node to confirm its bootstrap transfer."""
import argparse
import base64
import json
from pathlib import Path
import time

from neuroshard.client import wire
from neuroshard.client.local_node import LocalNode
from neuroshard.client.provider_wire import Unavailable


def wait_funding(config, receipt, *, seconds=180, rpc=wire.rpc,
                 clock=time.monotonic, sleep=time.sleep):
    if receipt['tx_result'].get('code', 0) or int(receipt['height']) < 1:
        raise ValueError('Require a successful committed funding transfer')
    txhash = receipt['hash'].upper()
    params = {'hash': base64.b64encode(bytes.fromhex(txhash)).decode(), 'prove': False}
    deadline = clock() + seconds
    last = None
    while clock() < deadline:
        try:
            node = LocalNode(config['node_rpc'], config['chain_id'], config['manifest_root'], rpc=rpc)
            local = rpc(node.url, 'tx', params, timeout=5)
            if (local['hash'].upper() != txhash or int(local['height']) != int(receipt['height'])
                    or local['tx_result'].get('code', 0)):
                raise RuntimeError('The provider node disagrees with its funding receipt')
            return {'transaction': txhash, 'height': int(local['height']), 'chain_id': node.chain_id}
        except (OSError, ValueError, Unavailable) as error:
            last = error
            sleep(.5)
    raise TimeoutError('The provider full node did not confirm its funding transfer') from last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--funding', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(wait_funding(json.loads(args.config.read_bytes()),
                                  json.loads(args.funding.read_bytes()))))


if __name__ == '__main__':
    main()
