"""A minimal CometBFT light client: verify that a ledger state is the one more than two thirds of validators signed.

The ledger's ABCI app hash is its state root (``optimistic.root``), and CometBFT puts the app hash after block H
in the header of block H + 1. So a state read from any RPC is accepted only if:

1. a header at the next height carries its root as ``app_hash``;
2. that header hashes (CometBFT's Merkle hash of its fields) to the block ID its commit signs;
3. the validator set returned for that height hashes to the header's ``validators_hash``;
4. ed25519 precommit signatures from more than two thirds of that set's power sign the block ID; and
5. signers holding at least one third of a *trusted* validator set's power are among them (the standard
   light-client "skipping" trust rule), so a set the RPC invented cannot pass.

The trusted set is pinned in the network descriptor. An RPC can withhold or delay answers, but it cannot make a
client accept a state the validators did not commit. Encoding follows CometBFT v0.38 (``types/block.go``,
``types/vote.go``, ``types/canonical.go``).
"""

import base64
import hashlib
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PRECOMMIT, COMMIT_FLAG = 2, 2


class Unverified(ValueError):
    """The validators' signatures do not establish this state."""


# Protobuf encoding (proto3: zero values are omitted).

def _varint(value):
    out = bytearray()
    value &= (1 << 64) - 1
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(field, wire):
    return _varint(field << 3 | wire)


def _uint(field, value):
    return _key(field, 0) + _varint(value) if value else b''


def _bytes(field, value):
    return _key(field, 2) + _varint(len(value)) + value if value else b''


def _message(field, value):
    return _key(field, 2) + _varint(len(value)) + value  # embedded messages are written even when empty


def _sfixed64(field, value):
    return _key(field, 1) + int(value).to_bytes(8, 'little', signed=True) if value else b''


def _timestamp(text):
    """RFC 3339 with up to nanosecond precision, as CometBFT's JSON writes it."""
    head, _, fraction = text.rstrip('Z').partition('.')
    seconds = int(datetime.fromisoformat(head).replace(tzinfo=timezone.utc).timestamp())
    nanos = int((fraction + '000000000')[:9]) if fraction else 0
    return _uint(1, seconds) + _uint(2, nanos)


def _hex(value):
    return bytes.fromhex(value) if value else b''


def _block_id(block_id):
    parts = block_id.get('parts') or block_id.get('part_set_header') or {}
    return _bytes(1, _hex(block_id.get('hash', ''))) + _message(2, _uint(1, int(parts.get('total', 0)))
                                                                + _bytes(2, _hex(parts.get('hash', ''))))


def merkle_root(leaves):
    """RFC 6962 Merkle root, as CometBFT's ``crypto/merkle.HashFromByteSlices``."""
    if not leaves:
        return hashlib.sha256(b'').digest()
    if len(leaves) == 1:
        return hashlib.sha256(b'\0' + leaves[0]).digest()
    split = 1 << (len(leaves) - 1).bit_length() - 1
    return hashlib.sha256(b'\1' + merkle_root(leaves[:split]) + merkle_root(leaves[split:])).digest()


def header_hash(header):
    """CometBFT's ``Header.Hash``: the Merkle root of each field's protobuf encoding."""
    version = header.get('version', {})
    wrapped = lambda value: _bytes(1, value)  # gogoproto StringValue / BytesValue
    leaves = [
        _uint(1, int(version.get('block', 0))) + _uint(2, int(version.get('app', 0))),
        wrapped(header['chain_id'].encode()),
        _uint(1, int(header['height'])),  # Int64Value
        _timestamp(header['time']),
        _block_id(header['last_block_id']),
        *(wrapped(_hex(header[name])) for name in (
            'last_commit_hash', 'data_hash', 'validators_hash', 'next_validators_hash', 'consensus_hash',
            'app_hash', 'last_results_hash', 'evidence_hash', 'proposer_address')),
    ]
    return merkle_root(leaves).hex().upper()


def _validator_key(validator):
    key = validator['pub_key']
    if key.get('type') != 'tendermint/PubKeyEd25519':
        raise Unverified('only ed25519 validators are supported')
    return base64.b64decode(key['value'])


def validators_hash(validators):
    """CometBFT's ``ValidatorSet.Hash``: Merkle root over each validator's (public key, voting power)."""
    return merkle_root([_message(1, _bytes(1, _validator_key(v))) + _uint(2, int(v['voting_power']))
                        for v in validators]).hex().upper()


def address(public):
    return hashlib.sha256(public).digest()[:20].hex().upper()


def vote_sign_bytes(chain_id, height, round_, block_id, timestamp):
    """Length-prefixed ``CanonicalVote`` for a precommit, what each validator's signature covers."""
    body = (_uint(1, PRECOMMIT) + _sfixed64(2, height) + _sfixed64(3, round_)
            + _message(4, _block_id(block_id)) + _message(5, _timestamp(timestamp)) + _bytes(6, chain_id.encode()))
    return _varint(len(body)) + body


def verify_commit(signed_header, validators, trusted=None):
    """Raise ``Unverified`` unless the commit is signed by more than 2/3 of ``validators`` and at least 1/3 of
    ``trusted`` (a {public key hex: power} map), for a header that hashes to the block ID it signs."""
    header, commit = signed_header['header'], signed_header['commit']
    if header_hash(header) != commit['block_id']['hash'].upper():
        raise Unverified('the header does not hash to the block ID its commit signs')
    if int(commit['height']) != int(header['height']):
        raise Unverified('the commit is for another height')
    if validators_hash(validators) != header['validators_hash'].upper():
        raise Unverified('the validator set does not match the header')
    by_address = {address(_validator_key(v)): v for v in validators}
    total = sum(int(v['voting_power']) for v in validators)
    signed, signers = 0, set()
    for signature in commit['signatures']:
        if signature.get('block_id_flag') != COMMIT_FLAG:
            continue
        validator = by_address.get(signature['validator_address'].upper())
        if validator is None or signature['validator_address'].upper() in signers:
            continue
        public = _validator_key(validator)
        message = vote_sign_bytes(header['chain_id'], int(commit['height']), int(commit['round']),
                                  commit['block_id'], signature['timestamp'])
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(base64.b64decode(signature['signature']), message)
        except (InvalidSignature, ValueError):
            continue
        signers.add(signature['validator_address'].upper())
        signed += int(validator['voting_power'])
    if 3 * signed <= 2 * total:
        raise Unverified(f'only {signed} of {total} voting power signed')
    if trusted:
        trusted_total = sum(trusted.values())
        overlap = sum(power for key, power in trusted.items() if address(bytes.fromhex(key)) in signers)
        if 3 * overlap < trusted_total:
            raise Unverified('too few of the trusted validators signed: update the network descriptor')
    return header
