"""Granite tokenization checked against the canonical parent tokenizer.json.

transformers 5.x resolves Granite's hub class name (GPT2Tokenizer) to a
constructor that replaces the serialized pre-tokenizer with GPT-2's byte regex
(huggingface/transformers#45812). Digit runs and ``_word`` identifiers then split
differently from training. The pinned Switch checkpoint serializes that GPT-2
regex in its own tokenizer.json, so matching a checkpoint's own file is not
enough: every checkpoint uses the parent's splitting, keeps its own added-token
IDs, and every runtime encode is checked against ``tokenizers``.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from neuroshard.evolution.modular_reference_execution import ROOT, identity, read

ARTIFACTS = 'config/experiments/granite-reference-artifacts.json'
# Strings whose pieces differ under the GPT-2 byte regex: identifiers, digit runs,
# ISO dates, document IDs, tool envelopes and non-ASCII punctuation from replies.
FIXTURE = (
    'save_draft', 'read_document', 'list_documents', 'shift_date', 'source_ids', 'due_date',
    '"start_date": "2027-05-14"', 'Cedar 6460', 'Meadow 6540 Annex', 'doc-2798891989',
    'doc-33e94c5aea', '1000000000000', '2027\u201104\u201126',
    '<tool_call>\n{"name": "save_draft", "arguments": {"project": "Harbor 6502", "total": 58}}\n</tool_call>',
    '<|start_of_role|>assistant<|end_of_role|>Done.<|end_of_text|>\n',
    "It's 3.5x faster\n\n  def f(x_1):\r\n\treturn x_1**2",
)
# The post-processor only acts with implicit special tokens, which Checked refuses;
# transformers 5 installs an empty template there.
PIPELINE_KEYS = ('normalizer', 'pre_tokenizer', 'model', 'decoder', 'added_tokens')
SPLITTING_KEYS = ('normalizer', 'pre_tokenizer', 'model', 'decoder')
PARENT_KEYS = ('normalizer', 'pre_tokenizer', 'post_processor', 'decoder')
# Granite's forward has no token types; the generic fast class would add them.
MODEL_INPUTS = ('input_ids', 'attention_mask')


def pipeline(serialized):
    value = json.loads(serialized)
    return {key: value.get(key) for key in PIPELINE_KEYS}


def splitting(serialized):
    value = json.loads(serialized)
    return {key: value.get(key) for key in SPLITTING_KEYS}


def blob_sha1(path):
    raw = Path(path).read_bytes()
    return hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()


def canonical(own, parent):
    """A checkpoint's vocabulary and added tokens on the parent's splitting pipeline."""
    if own['model'] != parent['model']:
        raise ValueError('vocabulary or merges differ from the canonical parent')
    parent_tokens = {token['id']: token for token in parent['added_tokens']}
    own_tokens = {token['id']: token for token in own['added_tokens']}
    if any(own_tokens.get(key) != token for key, token in parent_tokens.items()):
        raise ValueError('parent added-token IDs are not preserved')
    vocabulary = set(own['model']['vocab'].values())
    controls = {token['content']: key for key, token in sorted(own_tokens.items()) if key not in parent_tokens}
    if vocabulary.intersection(controls.values()):
        raise ValueError('a control token reuses a vocabulary ID')
    return {**own, **{key: parent.get(key) for key in PARENT_KEYS}}, controls


class Checked:
    """Delegate to the runtime tokenizer; refuse any encode that differs from the canonical reference."""

    def __init__(self, runtime, reference):
        self.runtime, self.reference = runtime, reference
        self.checked_encodes = 0

    def __getattr__(self, name):
        if name in ('runtime', 'reference'):
            raise AttributeError(name)
        return getattr(self.runtime, name)

    def __call__(self, text, **kwargs):
        if not isinstance(text, str) or kwargs.get('add_special_tokens', True) is not False:
            raise ValueError('checked tokenizer encodes one string without implicit special tokens')
        encoded = self.runtime(text, **kwargs)
        ids = encoded['input_ids']
        if kwargs.get('return_tensors') is not None:
            if len(ids) != 1:
                raise ValueError('checked tokenizer expects a single sequence')
            ids = ids[0].tolist()
        if list(ids) != self.reference.encode(text, add_special_tokens=False).ids:
            raise ValueError('runtime tokenization differs from the canonical tokenizer')
        self.checked_encodes += 1
        return encoded


def conformance(runtime, reference, parent=None, controls=None):
    """Parent splitting, own added tokens, parent fixture and control IDs; raises before any model call."""
    parent = parent or reference
    serialized = runtime.backend_tokenizer.to_str() if hasattr(runtime, 'backend_tokenizer') else None
    if serialized is None or identity(splitting(serialized)) != identity(splitting(parent.to_str())):
        raise ValueError('runtime tokenizer splitting differs from the canonical parent tokenizer.json')
    if identity(pipeline(serialized)) != identity(pipeline(reference.to_str())):
        raise ValueError('runtime tokenizer pipeline differs from its canonical serialization')
    fixture = [parent.encode(text, add_special_tokens=False).ids for text in FIXTURE]
    if [runtime(text, add_special_tokens=False)['input_ids'] for text in FIXTURE] != fixture:
        raise ValueError('runtime tokenizer fixture differs from the canonical parent')
    controls = controls or {}
    if any(runtime(content, add_special_tokens=False)['input_ids'] != [key] for content, key in controls.items()):
        raise ValueError('a control token no longer encodes to its checkpoint ID')
    if tuple(runtime(FIXTURE[0], add_special_tokens=False)) != MODEL_INPUTS:
        raise ValueError('runtime tokenizer returns undeclared model inputs')
    return {'runtime_class': type(runtime).__name__, 'pipeline_sha256': identity(pipeline(reference.to_str())),
            'splitting_sha256': identity(splitting(parent.to_str())), 'control_tokens': controls,
            'fixture_count': len(FIXTURE), 'fixture_sha256': identity(fixture),
            'eos_token_id': runtime.eos_token_id, 'pad_token_id': runtime.pad_token_id,
            'transformers': importlib.metadata.version('transformers'),
            'tokenizers': importlib.metadata.version('tokenizers')}


def agrees(runtime, reference):
    """Diagnostic only: does a tokenizer loaded another way match the canonical pipeline?"""
    try:
        conformance(runtime, reference)
    except ValueError:
        return False
    return True


def load(directory, parent=None, *, parent_digest=None):
    """Tokenizer for a Granite checkpoint directory, checked against the canonical parent.

    ``parent`` holds the parent tokenizer.json (default: ``directory``). Its git-blob
    SHA-1 must equal the pinned parent artifact unless ``parent_digest`` names another.
    """
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    directory = Path(directory)
    parent = Path(parent) if parent is not None else directory
    if parent_digest is None:
        parent_digest = read(ROOT / ARTIFACTS)['models']['baseline']['files']['tokenizer.json']['digest']
    if blob_sha1(parent / 'tokenizer.json') != parent_digest:
        raise ValueError('canonical parent tokenizer.json differs from the pinned Granite parent')
    own = json.loads((directory / 'tokenizer.json').read_text(encoding='utf-8'))
    serialized, controls = canonical(own, json.loads((parent / 'tokenizer.json').read_text(encoding='utf-8')))
    config = json.loads((directory / 'tokenizer_config.json').read_text(encoding='utf-8'))
    template = directory / 'chat_template.jinja'
    if not template.is_file():
        raise ValueError('pinned chat template is missing')
    # Direct construction: hub class resolution would reinstate the checkpoint's splitting.
    runtime = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(serialized)), bos_token=config['bos_token'],
        eos_token=config['eos_token'], pad_token=config['pad_token'], unk_token=config['unk_token'],
        padding_side=config.get('padding_side', 'left'), clean_up_tokenization_spaces=False,
        model_input_names=list(MODEL_INPUTS), chat_template=template.read_text(encoding='utf-8'))
    reference = Tokenizer.from_str(json.dumps(serialized))
    report = conformance(runtime, reference, Tokenizer.from_file(str(parent / 'tokenizer.json')), controls)
    return Checked(runtime, reference), {**report, 'parent_tokenizer_digest': parent_digest}
