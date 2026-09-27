"""Granite tokenization exactly as serialized in the pinned tokenizer.json.

transformers 5.x resolves Granite's hub class name (GPT2Tokenizer) to a
constructor that replaces the serialized pre-tokenizer with GPT-2's byte regex
(huggingface/transformers#45812). Digit runs and ``_word`` identifiers then split
differently from training. Every runtime encode is checked against ``tokenizers``.
"""

import importlib.metadata
import json
from pathlib import Path

from neuroshard.evolution.modular_reference_execution import identity

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
# Granite's forward has no token types; the generic fast class would add them.
MODEL_INPUTS = ('input_ids', 'attention_mask')


def pipeline(serialized):
    value = json.loads(serialized)
    return {key: value.get(key) for key in PIPELINE_KEYS}


class Checked:
    """Delegate to the runtime tokenizer; refuse any encode that differs from tokenizer.json."""

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
            raise ValueError('runtime tokenization differs from pinned tokenizer.json')
        self.checked_encodes += 1
        return encoded


def conformance(runtime, reference):
    """Pipeline and fixture agreement; raises before any model call on a mismatch."""
    serialized = runtime.backend_tokenizer.to_str() if hasattr(runtime, 'backend_tokenizer') else None
    if serialized is None or identity(pipeline(serialized)) != identity(pipeline(reference.to_str())):
        raise ValueError('runtime tokenizer pipeline differs from pinned tokenizer.json')
    fixture = [reference.encode(text, add_special_tokens=False).ids for text in FIXTURE]
    if [runtime(text, add_special_tokens=False)['input_ids'] for text in FIXTURE] != fixture:
        raise ValueError('runtime tokenizer fixture differs from pinned tokenizer.json')
    if tuple(runtime(FIXTURE[0], add_special_tokens=False)) != MODEL_INPUTS:
        raise ValueError('runtime tokenizer returns undeclared model inputs')
    return {'runtime_class': type(runtime).__name__, 'pipeline_sha256': identity(pipeline(reference.to_str())),
            'fixture_count': len(FIXTURE), 'fixture_sha256': identity(fixture),
            'eos_token_id': runtime.eos_token_id, 'pad_token_id': runtime.pad_token_id,
            'transformers': importlib.metadata.version('transformers'),
            'tokenizers': importlib.metadata.version('tokenizers')}


def agrees(runtime, reference):
    """Diagnostic only: does a tokenizer loaded another way match the pinned pipeline?"""
    try:
        conformance(runtime, reference)
    except ValueError:
        return False
    return True


def load(directory):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    directory = Path(directory)
    reference = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    runtime = PreTrainedTokenizerFast.from_pretrained(directory, local_files_only=True,
                                                      model_input_names=list(MODEL_INPUTS))
    if runtime.chat_template is None:
        raise ValueError('pinned chat template was not loaded')
    report = conformance(runtime, reference)
    return Checked(runtime, reference), report
