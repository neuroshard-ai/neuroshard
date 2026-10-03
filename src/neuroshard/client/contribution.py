"""Opt-in, signed assistant demonstrations and explicitly reviewed training export.

Packaging is local. A signature identifies a submitter, not truth, independent
ownership or entitlement to tokens. No submitted example executes external code.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from . import wire

DOMAIN = 'neuroshard/assistant-demonstration/v1'
MAX_BYTES = 1024 * 1024
MAX_EXPORT_BYTES = 32 * 1024 * 1024


def policy():
    return wire.parse(Path(__file__).with_name('assistant-policy.json').read_bytes())


def load(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Contribution exceeds 1 MiB')
    try:
        return wire.parse(raw)
    except RecursionError:
        raise ValueError('Contribution nesting is too deep') from None


def _text(value, maximum):
    return isinstance(value, str) and bool(value.strip()) and len(value.encode()) <= maximum


def validate_example(example):
    """Replay human-supplied actions; goals are read by the scorer only."""
    from neuroshard.evolution import assistant_workflow as workflow
    from neuroshard.evolution.assistant_workspace import REGISTRY, Workspace
    from neuroshard.evolution.modular_tools import _check_value

    if not isinstance(example, dict) or set(example) != {'world', 'turns'}:
        raise ValueError('Example requires world and turns')
    if len(wire.canonical(example)) > MAX_BYTES // 2:
        raise ValueError('Example exceeds 512 KiB')
    world = example['world']
    if (not isinstance(world, dict) or set(world) != {'documents'}
            or not isinstance(world['documents'], list)
            or not 1 <= len(world['documents']) <= 32):
        raise ValueError('Require a bounded public document workspace')
    for document in world['documents']:
        if (not isinstance(document, dict)
                or any(not _text(document.get(key), 1024) for key in ('id', 'project', 'title'))):
            raise ValueError('Document metadata must contain text identifiers')
    Workspace(world)
    turns = example['turns']
    if not isinstance(turns, list) or not 1 <= len(turns) <= 4:
        raise ValueError('Supply one to four conversation turns')
    texts = []
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {'user', 'assistant', 'expected'}:
            raise ValueError('Each turn requires user, assistant responses and expected draft')
        if (not _text(turn['user'], 16000) or not isinstance(turn['assistant'], list)
                or not 1 <= len(turn['assistant']) <= 6
                or any(not _text(text, 16384) for text in turn['assistant'])):
            raise ValueError('Malformed or over-budget demonstration turn')
        _check_value(turn['expected'], REGISTRY['save_draft'])
        texts.extend(turn['assistant'])
    root = wire.digest({'policy': policy(), 'example': example})
    case = {'id': root, 'world': example['world'],
            'turns': [{'user': t['user'], 'expected': t['expected']} for t in turns]}
    responses = iter(texts)
    records = []
    def respond(messages, tools):
        try:
            text = next(responses)
        except StopIteration:
            raise ValueError('Missing assistant response in demonstration') from None
        records.append({'messages': copy.deepcopy(messages) + [{'role': 'assistant', 'content': text}],
                        'tools': copy.deepcopy(tools), 'example_root': root})
        return {'text': text, 'terminated': True, 'executed': False,
                'input_token_ids': [], 'token_ids': [], 'prompt_sha256': wire.digest(messages)}
    result = workflow.execute(case, respond, policy())
    if next(responses, None) is not None or not result['score']['passed']:
        raise ValueError('Demonstration does not complete every declared draft outcome')
    previous = 0
    for turn, observed in zip(turns, result['rounds']):
        if observed['generation_count'] - previous != len(turn['assistant']):
            raise ValueError('Assistant responses cross declared conversation turns')
        previous = observed['generation_count']
    # Do not allow a malformed call followed by a lucky final state into SFT.
    if any('error' in row['result'] for row in result['calls']) or any(
            message['role'] == 'tool' and '"error"' in message['content'] for message in result['messages']):
        raise ValueError('Training demonstrations must not contain rejected tool calls')
    return {'example_root': root, 'turns': len(turns), 'assistant_decisions': len(records),
            'tool_calls': result['tool_calls'], 'records': records,
            'transcript_root': wire.digest(result['calls']), 'neural_execution': False}


def create(example, wallet, origin, revision, public_training=False):
    if public_training is not True:
        raise ValueError('Explicit public/training consent and rights attestation are required')
    if not _text(origin, 1024) or not _text(revision, 256):
        raise ValueError('Describe the source and its revision')
    checked = validate_example(example)
    return wallet.sign({'domain': DOMAIN, 'policy_root': wire.digest(policy()),
                        'example_root': checked['example_root'], 'example': example,
                        'source': {'origin': origin, 'revision': revision, 'license': 'Apache-2.0'},
                        'consent': {'public': True, 'training': True, 'rights_attested': True}})


def verify(packet):
    if len(wire.canonical(packet)) > MAX_BYTES:
        raise ValueError('Contribution exceeds 1 MiB')
    body, author = wire.verify(packet)
    if (set(body) != {'domain', 'policy_root', 'example_root', 'example', 'source', 'consent'}
            or body['domain'] != DOMAIN or body['policy_root'] != wire.digest(policy())):
        raise ValueError('Unknown contribution format or tool policy')
    consent = body['consent']
    if (not isinstance(consent, dict) or set(consent) != {'public', 'training', 'rights_attested'}
            or any(value is not True for value in consent.values())):
        raise ValueError('Contribution lacks public training consent')
    source = body['source']
    if (not isinstance(source, dict) or set(source) != {'origin', 'revision', 'license'}
            or source['license'] != 'Apache-2.0' or not _text(source['origin'], 1024)
            or not _text(source['revision'], 256)):
        raise ValueError('Missing source or declared license')
    checked = validate_example(body['example'])
    if checked['example_root'] != body['example_root']:
        raise ValueError('Contribution content root differs')
    return {**checked, 'author': author, 'source': source, 'admitted': False, 'rewarded': False}


def write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def protected_prompts():
    # Static contamination screening is not an evaluation or model call.
    # Values are public commitments; none enter exported training records.
    from neuroshard.evolution.assistant_workflow_data import cases
    return {normalize(turn['user']) for split in ('development', 'confirmation')
            for case in cases(split) for turn in case['turns']}


def normalize(text):
    return ' '.join(text.casefold().split())


def export_reviewed(manifest, directory, output):
    """No folder-wide auto-ingestion. Every packet hash must be named by review."""
    if (not isinstance(manifest, dict) or set(manifest) != {'format', 'purpose', 'entries'}
            or manifest['format'] != 'neuroshard-assistant-review/v1' or manifest['purpose'] != 'training'
            or not isinstance(manifest['entries'], list) or not 1 <= len(manifest['entries']) <= 256):
        raise ValueError('Require a bounded explicit training review manifest')
    seen, lines, blocked = set(), [], protected_prompts()
    total_bytes = 0
    for entry in manifest['entries']:
        if (not isinstance(entry, dict) or set(entry) != {'file', 'sha256', 'example_root', 'accepted_for_training'}
                or entry['accepted_for_training'] is not True
                or not isinstance(entry['file'], str) or not entry['file'] or entry['file'] in ('.', '..')
                or Path(entry['file']).name != entry['file']
                or not isinstance(entry['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])):
            raise ValueError('Require an approved packet filename and exact SHA-256')
        path = Path(directory) / entry['file']
        if path.is_symlink():
            raise ValueError('Contribution packet must not be a symlink')
        with path.open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Reviewed packet bytes changed')
        packet = wire.parse(raw)
        result = verify(packet)
        if result['example_root'] != entry['example_root'] or result['example_root'] in seen:
            raise ValueError('Duplicate or changed reviewed example')
        if any(normalize(turn['user']) in blocked for turn in packet['body']['example']['turns']):
            raise ValueError('Reserved workflow evaluation prompt cannot become training data')
        seen.add(result['example_root'])
        for record in result['records']:
            line = wire.canonical(record) + b'\n'
            total_bytes += len(line)
            if total_bytes > MAX_EXPORT_BYTES:
                raise ValueError('Reviewed training export exceeds 32 MiB')
            lines.append(line)
    # Entire manifest is validated before creating output; no partial dataset.
    payload = b''.join(lines)
    write_new(output, payload)
    return {'examples': len(seen), 'assistant_decisions': len(lines),
            'sha256': hashlib.sha256(payload).hexdigest(), 'review_root': wire.digest(manifest),
            'training_started': False, 'admitted': False, 'rewarded': False}


def register(subcommands, common):
    parser = subcommands.add_parser('contribute', help='Package or verify public assistant corrections locally')
    actions = parser.add_subparsers(dest='contribution_action', required=True)
    pack = actions.add_parser('package', parents=[common], help='Replay and sign an explicitly public training example')
    pack.add_argument('--input', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--origin', required=True)
    pack.add_argument('--revision', required=True)
    pack.add_argument('--public-training', action='store_true',
                      help='Attest rights to publish this entire example and permit Apache-2.0 training reuse')
    check = actions.add_parser('verify', help='Verify signature, source, consent and deterministic tool replay')
    check.add_argument('file', type=Path)
    export = actions.add_parser('export', help='Export only explicitly reviewed packets as assistant-decision JSONL')
    export.add_argument('--review', type=Path, required=True)
    export.add_argument('--directory', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)


def run(args):
    if args.contribution_action == 'package':
        if not args.public_training:
            raise ValueError('Review the example and use --public-training to attest public training rights')
        example = load(args.input)
        validate_example(example)  # Check before creating a local identity.
        wallet = wire.Wallet(args.home / 'account.key', create=True)
        packet = create(example, wallet, args.origin, args.revision, args.public_training)
        write_new(args.output, wire.canonical(packet) + b'\n')
        value = verify(packet)
        value.pop('records')
        print(json.dumps({**value, 'file': str(args.output), 'uploaded': False,
                          'next': 'Review the public packet and submit it for maintainer review; no automatic training or tokens.'}))
    elif args.contribution_action == 'verify':
        value = verify(load(args.file))
        value.pop('records')
        print(json.dumps(value))
    else:
        print(json.dumps(export_reviewed(load(args.review), args.directory, args.output)))
