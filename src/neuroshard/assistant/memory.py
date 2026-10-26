"""Memory across conversations, kept on the user's device (milestone 1, client only).

The user decides what is remembered: an item is saved only by an explicit ``/remember`` that the user confirms.
Nothing is saved from documents, tool results or the assistant's replies. Memory is off until the user enables it.

**Where it lives.** ``<home>/memory/store.bin``, encrypted with AES-256-GCM under ``<home>/memory/memory.key``
(created with mode 0600). It is never written to the chain, the request root, or a training export.

**What the network sees.** On a documents or chat turn, the few saved items relevant to the message (``select``:
BM25 over the message, at most ``MAX_ITEMS`` items and ``MAX_TOKENS`` tokens, each item at most once per
conversation) are put in front of it as a labelled block. Like the rest of a prompt, that block is computed on by the turn's owners, and
the activations they receive and keep in their logs can practically be turned back into text. So **items used in
a turn should be treated as readable by that turn's owners, by auditors during its challenge window, and by anyone
who later obtains those logs.** The client prints which items it used, and how many tokens, on every such turn.

Routing never sees memory: the gate and the router read the user's message alone, so the chosen route and module
are exactly those without memory. Drafting and scheduling, whose policies drive strict workflows, receive no
memory in this milestone.
"""
import json
import math
import os
import re
import secrets
import time
from pathlib import Path

MAX_STORED = 256
MAX_ITEM_BYTES = 280
MAX_STORE_BYTES = 64 * 1024
MAX_ITEMS = 3
MAX_TOKENS = 160
KINDS = ('fact', 'preference', 'context')
ROUTES = ('documents', 'chat')  # routes that receive memory in this milestone
HEADER = '[Saved notes the user chose to share for this conversation; use them only if relevant]'
FOOTER = '[End of notes]'
DISCLOSURE = ('Items used in a turn are visible to the network operators serving that turn, and to auditors '
              'who check it.')
WORD = re.compile(r'[a-z0-9]+')
STOP = frozenset('a an and are as at be by do for from has have how i in is it its me my of on or our so that the '
                 'their them they this to was we what when where which who why will with you your'.split())

SECRETS = [
    ('a private key', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\bssh-(?:ed25519|rsa) AAAA')),
    ('a 64-character hex key', re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])')),
    ('an access token', re.compile(r'\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abpr]-[A-Za-z0-9-]{10,}|'
                                   r'sk-[A-Za-z0-9]{20,})\b')),
    ('a password or PIN', re.compile(r'\b(?:password|passcode|passwd|pin|ssn|social security)\b\s*(?:is|:|=)?\s*\S',
                                     re.I)),
    ('an IBAN', re.compile(r'\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b')),
]
CARD = re.compile(r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)')


class Invalid(ValueError):
    """An item or store operation outside memory's bounds."""


def luhn(digits):
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch) * (2 if double else 1)
        total += d - 9 if d > 9 else d
        double = not double
    return total % 10 == 0


def secret_check(text):
    """Why ``text`` looks like it holds a secret (an empty list if it doesn't). A guard, not a guarantee."""
    found = [name for name, pattern in SECRETS if pattern.search(text)]
    for match in CARD.finditer(text):
        digits = re.sub(r'\D', '', match.group())
        if 13 <= len(digits) <= 19 and luhn(digits):
            found.append('a card number')
            break
    words = text.lower().split()
    if len(words) in (12, 24) and all(w.isalpha() and 3 <= len(w) <= 8 for w in words):
        found.append('a recovery phrase')
    return found


def clean(text):
    """One line of plain text: no brackets, angle brackets or line breaks that could pose as a block or a tool call."""
    text = re.sub(r'[\[\]<>{}]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


class Store:
    """The device's memory: encrypted at rest, bounded, with an on/off setting kept beside it."""

    def __init__(self, home):
        self.directory = Path(home) / 'memory'
        self.path, self.key_path = self.directory / 'store.bin', self.directory / 'memory.key'
        self.settings_path = self.directory / 'settings.json'

    # Settings ------------------------------------------------------------------------------------------------
    @property
    def enabled(self):
        if not self.settings_path.exists():
            return False
        return bool(json.loads(self.settings_path.read_text()).get('enabled'))

    def enable(self, on=True):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps({'enabled': bool(on)}) + '\n')

    # Storage -------------------------------------------------------------------------------------------------
    def _key(self, create):
        if not self.key_path.exists():
            if not create:
                return None
            self.directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(secrets.token_bytes(32))
        return self.key_path.read_bytes()

    def items(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = self._key(create=False)
        if key is None or not self.path.exists():
            return []
        blob = self.path.read_bytes()
        return json.loads(AESGCM(key).decrypt(blob[:12], blob[12:], b'neuroshard-memory/1'))['items']

    def _save(self, items):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        body = json.dumps({'items': items}, ensure_ascii=False, sort_keys=True).encode()
        if len(body) > MAX_STORE_BYTES:
            raise Invalid(f'memory is full ({MAX_STORE_BYTES // 1024} KB); forget some items first')
        nonce = secrets.token_bytes(12)
        blob = nonce + AESGCM(self._key(create=True)).encrypt(nonce, body, b'neuroshard-memory/1')
        partial = self.path.with_suffix('.partial')
        partial.write_bytes(blob)
        os.chmod(partial, 0o600)
        partial.replace(self.path)

    def add(self, text, kind='fact'):
        text = clean(text)
        if not text:
            raise Invalid('nothing to remember')
        if kind not in KINDS:
            raise Invalid(f'kind is one of {", ".join(KINDS)}')
        if len(text.encode()) > MAX_ITEM_BYTES:
            raise Invalid(f'an item holds at most {MAX_ITEM_BYTES} bytes')
        items = self.items()
        if len(items) >= MAX_STORED:
            raise Invalid(f'memory holds at most {MAX_STORED} items; forget some first')
        number = max((int(i['id'][1:]) for i in items), default=0) + 1
        item = {'id': f'm{number}', 'text': text, 'kind': kind, 'created': time.strftime('%Y-%m-%d'),
                'last_used': None}
        self._save(items + [item])
        return item

    def remove(self, item_id):
        items = self.items()
        kept = [i for i in items if i['id'] != item_id]
        if len(kept) == len(items):
            raise Invalid(f'no item {item_id}')
        self._save(kept)

    def used(self, ids):
        if not ids:
            return
        today = time.strftime('%Y-%m-%d')
        self._save([{**i, 'last_used': today} if i['id'] in ids else i for i in self.items()])

    def wipe(self):
        """Delete the items and the key: nothing can be decrypted afterwards."""
        for path in (self.path, self.key_path):
            if path.exists():
                path.unlink()


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def select(items, query, count_tokens, k=MAX_ITEMS, max_tokens=MAX_TOKENS):
    """The items relevant to ``query``, best first: BM25 over the items, only positive scores, ties by ID.

    Deterministic: no clock, no randomness. Items are added while the rendered block stays within ``max_tokens``
    (counted by ``count_tokens``, the served tokenizer), so an irrelevant store sends nothing."""
    docs = [(item, words(item['text'])) for item in items]
    terms = set(words(query))
    if not docs or not terms:
        return []
    average = sum(len(w) for _, w in docs) / len(docs) or 1
    frequency = {t: sum(t in set(w) for _, w in docs) for t in terms}
    scored = []
    for item, w in docs:
        score = 0.0
        for term in terms:
            tf = w.count(term)
            if tf:
                idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(w) / average))
        if score > 0:
            scored.append((-score, int(item['id'][1:]), item))
    chosen = []
    for _, _, item in sorted(scored, key=lambda row: row[:2]):
        if len(chosen) == k:
            break
        if count_tokens(render(chosen + [item])) <= max_tokens:
            chosen.append(item)
    return chosen


def render(items):
    """The block put before the user's message: labelled as the user's own notes, one plain line per item."""
    if not items:
        return ''
    lines = [HEADER] + [f'- ({item["kind"]}) {clean(item["text"])}' for item in items] + [FOOTER]
    return '\n'.join(lines) + '\n'


class Recall:
    """What a conversation uses: the store, the served tokenizer, and the items already sent in it."""

    def __init__(self, store, count_tokens):
        self.store, self.count_tokens, self.sent = store, count_tokens, set()

    def block(self, route, text, skip=False):
        """(block, items) for this turn: the relevant items not yet sent in this conversation, on documents or chat.

        An item sent once stays in the conversation's history, so it is never sent again in the same conversation."""
        if skip or route not in ROUTES or not self.store.enabled:
            return '', []
        fresh = [i for i in self.store.items() if i['id'] not in self.sent and not secret_check(i['text'])]
        chosen = select(fresh, text, self.count_tokens)
        if not chosen:
            return '', []
        self.sent |= {i['id'] for i in chosen}
        self.store.used({i['id'] for i in chosen})
        return render(chosen), chosen
