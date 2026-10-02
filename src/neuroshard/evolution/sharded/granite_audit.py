"""Owner logs and replay audits: optimistic verification of sharded Granite work.

An owner's serving loop is a deterministic state machine: its cache and arm
setting change only by the commands it receives, and each forward pass maps a
received tensor to a sent tensor. The owner logs every command with the digests
of what it received and sent, and retains the received tensors. Anyone holding
only that owner's shard can replay the log from an empty cache and compare
digests; because execution is bit-exact, the first differing digest is a fraud
proof that any other holder of the shard can check the same way.

For settlement a log is bound to one job. Every message on every serving link
carries its sender's signature over that link's running transcript; the user's
device signs under the job's session key. An owner's log keeps the latest
signature it received, so its inputs are what its upstream sender sent for that
job. A log whose entries depart from the transcript it was sent, or from the
transcript its owner signed when passing results on, is provable fraud without
any replay.
"""
import hashlib
import json
import logging
import time
from pathlib import Path

import torch

from .granite_pipeline import FORWARD, RESET
from .granite_serving import ADAPTER, CROP, FEATURE

LOG_FORMAT = 'neuroshard-granite-owner-log/1'
PROOF_FORMAT = 'neuroshard-granite-fraud-proof/1'
BOUND_PROOF_FORMAT = 'neuroshard-granite-fraud-proof/2'
CLAIMS = ('replay', 'unattested', 'equivocation')
PROOF_FILES = ('inputs.safetensors', 'proof.json', 'record.json')
SIGNATURE_BYTES = 64
LOG = logging.getLogger('neuroshard.audit')


def protocol():
    """The ledger's transcript and statement formats, imported only where a bound log is used."""
    from neuroshard.inference import optimistic

    return optimistic


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Link:
    """One owner's two serving links in a job: the link it receives on and the link it sends on.

    Each link's transcript steps once per command and once per message. A receiver checks
    the sender's signature over its own copy of the transcript before using a message and
    keeps the latest one; that signature binds its log to the job.
    """

    def __init__(self, session, rank, world, key, upstream):
        self.ledger = protocol()
        self.session = {name: session[name] for name in ('chain_id', 'job_id', 'request_root')}
        self.key, self.upstream = key, upstream
        self.hops = ((rank - 1) % world, rank)
        self.heads = [self.ledger.link_seed(self.session['chain_id'], self.session['job_id'], hop) for hop in self.hops]
        self.counts = [0, 0]
        self.received = None

    def message(self, side, head, count):
        return self.ledger.link_message(self.session['chain_id'], self.session['job_id'], self.hops[side], count, head)

    def command(self, op, value):
        self.heads = [self.ledger.link_step(head, op, value) for head in self.heads]
        self.counts = [count + 1 for count in self.counts]

    def receive(self, op, value, tensor, signature):
        """Accept a received message only with its sender's signature over the transcript that includes it.

        Returns that signature as an attestation: the evidence of everything sent on this link so far.
        """
        head, count = self.ledger.link_step(self.heads[0], op, value, digest(tensor)), self.counts[0] + 1
        if not self.ledger.ed25519_valid(self.upstream, self.message(0, head, count), signature.hex()):
            raise ValueError('the upstream sender did not sign this message')
        self.heads[0], self.counts[0] = head, count
        self.received = {'key': self.upstream, 'entries': count, 'head': head, 'signature': signature.hex()}
        return self.received

    def send(self, op, value, tensor):
        """This owner's signature over its outgoing transcript, including the message it is about to send."""
        self.heads[1] = self.ledger.link_step(self.heads[1], op, value, digest(tensor))
        self.counts[1] += 1
        return self.key.sign(self.message(1, self.heads[1], self.counts[1]))


class OwnerLog:
    """Commands in order, with input and output digests and the retained inputs.

    ``attested`` is the latest upstream signature covering every logged message, updated only
    once the message it covers has been logged.
    """

    def __init__(self, rank):
        self.rank, self.entries, self.payloads, self.attested = rank, [], {}, None

    def command(self, op, value):
        self.entries.append({'op': op, 'value': value})

    def forward(self, op, value, received, sent, attested=None):
        index = len(self.entries)
        self.entries.append({'op': op, 'value': value, 'input': digest(received), 'output': digest(sent)})
        self.payloads[index] = received.detach().contiguous().view(torch.uint8).clone()
        if attested is not None:
            self.attested = attested

    def save(self, directory, key=None, session=None):
        """Write the log and retained inputs; with an Ed25519 ``key`` the owner signs the log's digest.

        With the job's ``session`` the log is bound to that job and ends at the last entry its
        upstream sender signed; commands after the last message change no output.
        """
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries, payloads = self.entries, self.payloads
        if session is not None:
            signed = (self.attested or {}).get('entries', 0)
            entries, payloads = entries[:signed], {i: p for i, p in payloads.items() if i < signed}
        save_file({f'p{index}': value for index, value in payloads.items()}, str(directory / 'inputs.safetensors'))
        inputs_sha256 = hashlib.sha256((directory / 'inputs.safetensors').read_bytes()).hexdigest()
        if session is None:
            record = {'format': LOG_FORMAT, 'rank': self.rank, 'entries': entries, 'inputs_sha256': inputs_sha256}
            if key is not None:
                record = sign(record, key)
        else:
            record = bind(self.rank, entries, inputs_sha256, key, session, self.attested)
        (directory / 'log.json').write_text(json.dumps(record) + '\n')
        return {'entries': len(entries), 'forwards': len(payloads), 'directory': str(directory),
                'public_key': record.get('public_key')}


def header(record):
    """What the ledger checks when a bound log is committed: owner, job binding and the upstream signature."""
    return {name: record[name] for name in sorted(protocol().HEADER_FIELDS)}


def bound(record):
    return isinstance(record, dict) and record.get('format') == protocol().OWNER_LOG_FORMAT


def statement(record):
    """The signed and committed bytes.

    A bound log commits to its header and the digest of its entries; an unbound log to every
    field except the signature, canonically encoded.
    """
    if bound(record):
        ledger = protocol()
        return bytes.fromhex(ledger.log_statement(header(record), ledger.digest(record['entries'])))
    body = {k: v for k, v in record.items() if k != 'signature'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).digest()


def public_hex(key):
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def sign(record, key):
    record = {**record, 'public_key': public_hex(key)}
    return {**record, 'signature': key.sign(statement(record)).hex()}


def bind(rank, entries, inputs_sha256, key, session, attested):
    """A log bound to its job: the header the ledger checks, the entries auditors replay, the owner's signature."""
    record = {'format': protocol().OWNER_LOG_FORMAT, 'rank': rank, 'public_key': public_hex(key),
              'session': {name: session[name] for name in ('chain_id', 'job_id', 'request_root')},
              'upstream': attested, 'entries': entries, 'inputs_sha256': inputs_sha256}
    return {**record, 'signature': key.sign(statement(record)).hex()}


def commitment(record):
    """What an owner submits to commit a bound log: its header, the digest of its entries and its statement."""
    ledger = protocol()
    head, entries_root = header(record), ledger.digest(record['entries'])
    return {'header': head, 'entries_root': entries_root, 'statement_root': ledger.log_statement(head, entries_root)}


def signed_by(record, public_key):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if record.get('public_key') != public_key or 'signature' not in record:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(bytes.fromhex(record['signature']),
                                                                             statement(record))
        return True
    except (InvalidSignature, ValueError):
        return False


def transcript(record, side, count=None):
    """The link transcript a bound log implies: of its inputs (``'input'``) or of its outputs (``'output'``)."""
    ledger = protocol()
    session = record['session']
    hop = record['rank'] - 1 if side == 'input' else record['rank']
    head = ledger.link_seed(session['chain_id'], session['job_id'], hop)
    for entry in record['entries'][:count]:
        head = ledger.link_step(head, entry['op'], entry['value'], entry.get(side))
    return head


def unattested(record):
    """True when a bound log's entries are not exactly the transcript its upstream sender signed."""
    upstream = record['upstream']
    return not (isinstance(upstream, dict) and upstream['entries'] == len(record['entries'])
                and transcript(record, 'input') == upstream['head'])


def equivocates(record, attestation):
    """True when the log's owner signed ``attestation``, an outgoing transcript head, that its log contradicts."""
    ledger = protocol()
    if not isinstance(attestation, dict) or not {'entries', 'head', 'signature'} <= set(attestation) <= {
            'key', 'entries', 'head', 'signature'} or attestation.get('key', record['public_key']) != record['public_key']:
        return False
    entries, head, signature = attestation['entries'], attestation['head'], attestation['signature']
    if type(entries) is not int or entries < 1 or not isinstance(head, str) or not isinstance(signature, str):
        return False
    session = record['session']
    message = ledger.link_message(session['chain_id'], session['job_id'], record['rank'], entries, head)
    if not ledger.ed25519_valid(record['public_key'], message, signature):
        return False
    return entries > len(record['entries']) or transcript(record, 'output', entries) != head


def fraud_proof(record, payloads, report):
    """The owner's own signed log and the retained inputs up to its first wrong output."""
    if report['valid'] or report['first_mismatch'] is None:
        raise ValueError('no mismatch to prove')
    needed = {index: payloads[index] for index in payloads if index <= report['first_mismatch']}
    return {'claim': 'replay', 'record': record, 'mismatch': report['first_mismatch'], 'inputs': needed}


def claim_proof(record, claim, attestation=None):
    """A proof against a bound log that needs no replay: its entries were not what it was sent
    (``'unattested'``), or not what its owner signed passing results on (``'equivocation'``)."""
    if claim not in ('unattested', 'equivocation') or (claim == 'equivocation') != (attestation is not None):
        raise ValueError('unsupported claim')
    return {'claim': claim, 'record': record, 'inputs': {}, **({'attestation': attestation} if attestation else {})}


def save_proof(proof, directory):
    """The owner's unchanged log, the claim's evidence, and a manifest binding them."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = proof['record']
    save_file({f'p{i}': v.contiguous().view(torch.uint8) for i, v in proof.get('inputs', {}).items()},
              str(directory / 'inputs.safetensors'))
    (directory / 'record.json').write_text(json.dumps(record) + '\n')
    inputs_sha256 = hashlib.sha256((directory / 'inputs.safetensors').read_bytes()).hexdigest()
    if bound(record):
        claim = proof.get('claim', 'replay')
        manifest = {'format': BOUND_PROOF_FORMAT, 'claim': claim, 'rank': record['rank'],
                    'public_key': record['public_key'], 'inputs_sha256': inputs_sha256}
        if claim == 'replay':
            manifest['mismatch'] = proof['mismatch']
        elif claim == 'equivocation':
            manifest['attestation'] = proof['attestation']
    else:
        manifest = {'format': PROOF_FORMAT, 'rank': record['rank'], 'mismatch': proof['mismatch'],
                    'public_key': record.get('public_key'), 'inputs_sha256': inputs_sha256}
    (directory / 'proof.json').write_text(json.dumps(manifest) + '\n')
    return manifest


def parse_bundle(files):
    """The proof held in a bundle's bytes; raises if its contents are malformed."""
    from safetensors.torch import load as load_tensors

    manifest = json.loads(files['proof.json'])
    if not isinstance(manifest, dict) or manifest.get('format') not in (PROOF_FORMAT, BOUND_PROOF_FORMAT):
        raise ValueError('unsupported fraud proof')
    if hashlib.sha256(files['inputs.safetensors']).hexdigest() != manifest['inputs_sha256']:
        raise ValueError('proof inputs differ from the proof manifest')
    stored = load_tensors(files['inputs.safetensors'])
    proof = {'claim': manifest.get('claim', 'replay'), 'record': json.loads(files['record.json']),
             'mismatch': manifest.get('mismatch'), 'attestation': manifest.get('attestation'),
             'inputs': {int(k[1:]): v.view(torch.bfloat16) for k, v in stored.items()}}
    return proof, manifest


def load_proof(directory):
    directory = Path(directory)
    return parse_bundle({name: (directory / name).read_bytes() for name in PROOF_FILES})


def bundle_digest(files):
    listing = {name: hashlib.sha256(files[name]).hexdigest() for name in PROOF_FILES}
    return hashlib.sha256(json.dumps(listing, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bundle_root(directory):
    """Content address of a saved fraud proof: its files by digest."""
    directory = Path(directory)
    return bundle_digest({name: (directory / name).read_bytes() for name in PROOF_FILES})


def read_bundle(store, root):
    """The bytes held at content address ``root``; ProofUnavailable unless every file is here and matches it."""
    ledger = protocol()
    if store is None:
        raise ledger.ProofUnavailable('this validator keeps no bundle store')
    directory = Path(store) / root
    try:
        files = {name: (directory / name).read_bytes() for name in PROOF_FILES}
    except OSError as error:
        raise ledger.ProofUnavailable(f'bundle {root} is not held here ({type(error).__name__})') from error
    if bundle_digest(files) != root:
        raise ledger.ProofUnavailable(f'bundle {root} is held incompletely or differs from its address')
    return files


def challenge_checker(store, partitions, adapters=None):
    """A ledger executor: true only if the bundle at ``proof_root`` proves fraud against the committed log.

    ``store`` holds bundles in directories named by their content address. A bundle this
    validator does not hold raises ``ProofUnavailable``; a replay it cannot run, without the
    challenged shard or out of memory, raises ``NoVerdict``. Neither is a verdict. Every
    verdict is a function of the request and of bytes matching its content address.
    """
    ledger = protocol()

    def check(state, request):
        files = read_bundle(store, request['proof_root'])
        try:
            proof, _ = parse_bundle(files)
            return judge(proof, request, partitions, adapters or {}) is True
        except ledger.NoVerdict:
            raise
        except (MemoryError, torch.OutOfMemoryError) as error:
            raise ledger.NoVerdict('the proof replay ran out of memory') from error
        except RuntimeError as error:
            if 'allocate memory' in str(error):
                raise ledger.NoVerdict('the proof replay ran out of memory') from error
            LOG.warning('bundle %s is malformed: %s', request['proof_root'], error)
            return False
        except Exception as error:
            LOG.warning('bundle %s is malformed: %s: %s', request['proof_root'], type(error).__name__, error)
            return False
    return check


def judge(proof, request, partitions, adapters):
    """The verdict on a parsed bundle against one committed log.

    The record must be the bound log committed for the job: its statement must equal the
    committed root, whose header the ledger checked at commitment. Its own signature is not
    required; the owner's commitment already signs that statement.
    """
    record, claim = proof['record'], proof['claim']
    if (not bound(record) or claim not in CLAIMS or statement(record).hex() != request['statement_root']
            or record['public_key'] != request['log_key'] or record['rank'] != request['shard']
            or record['session']['chain_id'] != request['chain_id'] or record['session']['job_id'] != request['job_id']):
        return False
    if claim == 'unattested':
        return unattested(record)
    if claim == 'equivocation':
        return equivocates(record, proof['attestation'])
    if request['shard'] not in partitions:
        raise protocol().NoVerdict('this validator does not hold the challenged shard')
    mismatch = proof['mismatch']
    if type(mismatch) is not int or not 0 <= mismatch < len(record['entries']):
        return False
    head = {**record, 'entries': record['entries'][:mismatch + 1]}
    report = replay(partitions[request['shard']], head, proof['inputs'], adapters.get(request['shard']))
    return report['input_mismatch'] is None and report['first_mismatch'] == mismatch


def check_fraud_proof(partition, proof, public_key, adapter=None):
    """True when the owner signed the log and its logged output at ``mismatch`` is not what its shard computes."""
    record = proof['record']
    if not signed_by(record, public_key) or record['rank'] != partition.rank:
        return False
    head = {**record, 'entries': record['entries'][:proof['mismatch'] + 1]}
    report = replay(partition, head, proof['inputs'], adapter)
    return report['input_mismatch'] is None and report['first_mismatch'] == proof['mismatch']


def load(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    record = json.loads((directory / 'log.json').read_text())
    if record['format'] not in (LOG_FORMAT, protocol().OWNER_LOG_FORMAT):
        raise ValueError('unsupported owner log')
    if hashlib.sha256((directory / 'inputs.safetensors').read_bytes()).hexdigest() != record['inputs_sha256']:
        raise ValueError('retained inputs differ from the log')
    stored = load_file(str(directory / 'inputs.safetensors'))
    return record, {int(key[1:]): value.view(torch.bfloat16) for key, value in stored.items()}


def outputs(entries):
    return [entry['output'] for entry in entries if 'output' in entry]


def inputs(entries):
    return [entry['input'] for entry in entries if 'input' in entry]


def continuity(upstream, downstream):
    """First forward message whose received digest differs from what the upstream owner logged sending, or None."""
    sent, received = outputs(upstream), inputs(downstream)
    for index, (a, b) in enumerate(zip(sent, received)):
        if a != b:
            return index
    return None if len(sent) == len(received) else min(len(sent), len(received))


def replay(partition, record, payloads, adapter=None):
    """Re-execute one owner's log from an empty cache; returns the first entry whose output differs.

    Every logged command is checked against what serving could have sent before it runs, so
    a malformed log fails the same way on every replaying machine. A retained input that is
    missing, misshapen or different from its logged digest stops the replay as an input
    mismatch. The caller must run the auditor in the owners' pinned runtime, after the
    declared warm-up.
    """
    from transformers import DynamicCache

    last = partition.rank == len(partition.boundaries) - 2
    width, positions = partition.config.hidden_size, partition.config.max_position_embeddings
    cache, checked, started = DynamicCache(), 0, time.monotonic()
    report = {'entries': len(record['entries']), 'input_mismatch': None, 'first_mismatch': None}
    for index, entry in enumerate(record['entries']):
        op, value = entry['op'], entry['value']
        if type(op) is not int or type(value) is not int:
            raise ValueError('malformed logged command')
        if op == RESET:
            cache = DynamicCache()
        elif op == CROP:
            if not 0 <= value <= cache.get_seq_length():
                raise ValueError('logged crop outside the cache')
            cache.crop(value)
        elif op == ADAPTER:
            if value not in (0, 1):
                raise ValueError('malformed logged arm switch')
            if adapter is not None:
                adapter.set(bool(value))
        elif op in (FORWARD, FEATURE):
            past = cache.get_seq_length() if op == FORWARD else 0
            if not 0 < value <= positions - past:
                raise ValueError('logged message outside the model context')
            hidden = payloads.get(index)
            if (hidden is None or hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, value, width)
                    or digest(hidden) != entry['input']):
                report['input_mismatch'] = index
                break
            with torch.inference_mode():
                if op == FORWARD:
                    mask = torch.ones((1, past + value), dtype=torch.long)
                    out = partition(hidden, mask, cache)
                else:
                    out = partition(hidden, None, DynamicCache())
            checked += 1
            if digest(out[:, -1:] if last else out) != entry['output']:
                report['first_mismatch'] = index
                break
        else:
            raise ValueError('unknown logged command')
    return {**report, 'forwards_checked': checked, 'seconds': time.monotonic() - started,
            'valid': report['input_mismatch'] is None and report['first_mismatch'] is None}
