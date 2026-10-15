"""Public randomness nobody controls: drand quicknet beacons, verified here, never trusted from an API.

drand is a threshold network run by many independent organisations (the League of Entropy). Every
three seconds it publishes a BLS signature over the round number; the round's randomness is the
SHA-256 of that signature. No party, and no minority of the network, can know or bias a future round.

This module verifies a beacon against quicknet's fixed group key (``bls-unchained-g1-rfc9380``: G1
signatures, the G2 group key, message SHA-256 of the 8-byte big-endian round) and maps time to rounds,
so the ledger can name a round that did not exist yet when a commitment was made.
"""

import hashlib

CHAIN = '52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971'
PUBLIC_KEY = ('83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8c4b450b6a0a6c3ac6a5776a2d1064510d1f'
              'ec758c921cc22b0e17e63aaf4bcb5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a')
GENESIS, PERIOD = 1692803367, 3
DST = b'BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_'
ENDPOINTS = ('https://api.drand.sh', 'https://api2.drand.sh', 'https://api3.drand.sh', 'https://drand.cloudflare.com')


def round_at(unix):
    """The first round published at or after ``unix`` seconds."""
    if unix <= GENESIS:
        return 1
    return -(-(int(unix) - GENESIS) // PERIOD) + 1


def time_of(round_number):
    return GENESIS + (round_number - 1) * PERIOD


def randomness(signature_hex):
    return hashlib.sha256(bytes.fromhex(signature_hex)).hexdigest()


def _bls():
    """py_ecc's hash-to-curve and point decompression, without its ``bls`` package initialiser.

    That initialiser builds the Ethereum ciphersuites and pulls in eth-utils, pydantic and cytoolz; verifying a
    drand beacon needs none of them, so the two modules it uses are loaded under a bare package entry.
    """
    import importlib
    import sys
    import types

    if 'py_ecc.bls' not in sys.modules:
        import py_ecc

        package = types.ModuleType('py_ecc.bls')
        package.__path__ = [str(__import__('pathlib').Path(py_ecc.__file__).parent / 'bls')]
        sys.modules['py_ecc.bls'] = package
    curve = importlib.import_module('py_ecc.bls.hash_to_curve')
    points = importlib.import_module('py_ecc.bls.point_compression')
    return curve.hash_to_G1, points.decompress_G1, points.decompress_G2


def verify(round_number, signature_hex, randomness_hex=None):
    """True if ``signature_hex`` is quicknet's signature for ``round_number`` (and matches ``randomness_hex``)."""
    hash_to_G1, decompress_G1, decompress_G2 = _bls()
    from py_ecc.optimized_bls12_381 import FQ12, G2, final_exponentiate, neg, pairing

    if type(round_number) is not int or round_number < 1:
        return False
    try:
        signature = bytes.fromhex(signature_hex)
        key = bytes.fromhex(PUBLIC_KEY)
        if len(signature) != 48:
            return False
        point = decompress_G1(int.from_bytes(signature, 'big'))
        group = decompress_G2((int.from_bytes(key[:48], 'big'), int.from_bytes(key[48:], 'big')))
    except (ValueError, AssertionError):
        return False
    if randomness_hex is not None and randomness(signature_hex) != randomness_hex:
        return False
    message = hash_to_G1(hashlib.sha256(round_number.to_bytes(8, 'big')).digest(), DST, hashlib.sha256)
    # e(sig, g2) == e(H(m), pk)  <=>  e(sig, -g2) * e(H(m), pk) == 1
    product = pairing(neg(G2), point, final_exponentiate=False) * pairing(group, message, final_exponentiate=False)
    return final_exponentiate(product) == FQ12.one()


def fetch(round_number, opener=None, timeout=20):
    """One verified beacon from the first drand endpoint that serves a valid one: (round, signature, randomness)."""
    import json
    import urllib.request

    opener = opener or urllib.request.urlopen
    failures = []
    for base in ENDPOINTS:
        try:
            with opener(f'{base}/{CHAIN}/public/{round_number}', timeout=timeout) as response:
                value = json.loads(response.read())
        except Exception as error:  # an endpoint is only a transport; the signature is what counts
            failures.append(f'{base}: {error}')
            continue
        if value.get('round') == round_number and verify(round_number, value['signature'], value.get('randomness')):
            return round_number, value['signature'], randomness(value['signature'])
        failures.append(f'{base}: invalid beacon')
    raise ValueError(f'no verified beacon for round {round_number}: ' + '; '.join(failures))
