import json
import os
from pathlib import Path

import pytest

from neuroshard.evolution import granite_tokenizer

SPLIT = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
         "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
SPECIAL = ['<|pad|>', '<|end_of_text|>', '<|start_of_role|>', '<|end_of_role|>']
CONTROLS = ['<|citations|>', '<|answerability|>']
TEMPLATE = ("{% for m in messages %}<|start_of_role|>{{ m.role }}<|end_of_role|>{{ m.content }}<|end_of_text|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>{% endif %}")
GPT2_REGEX = {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True, 'use_regex': True}


def granite_like(directory):
    """Tiny BPE with Granite's serialized pipeline and its misleading hub class name."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT), behavior='removed', invert=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    corpus = [text for text in granite_tokenizer.FIXTURE for _ in range(20)]
    corpus += ['save_draft draft list_documents documents 2027 1234567 revision approved'] * 20
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / 'tokenizer.json'))
    (directory / 'tokenizer_config.json').write_text(json.dumps({
        'tokenizer_class': 'GPT2Tokenizer', 'add_bos_token': False, 'add_prefix_space': False,
        'bos_token': '<|end_of_text|>', 'eos_token': '<|end_of_text|>', 'pad_token': '<|pad|>',
        'unk_token': '<|end_of_text|>', 'model_max_length': 4096}))
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    return directory


def switch_like(parent, directory):
    """Same vocabulary and merges, GPT-2 splitting in its own file, and appended control tokens."""
    serialized = json.loads((parent / 'tokenizer.json').read_text())
    serialized['pre_tokenizer'] = GPT2_REGEX
    first = max(max(serialized['model']['vocab'].values()), max(t['id'] for t in serialized['added_tokens'])) + 1
    serialized['added_tokens'] += [{'id': first + i, 'content': content, 'single_word': False, 'lstrip': False,
                                    'rstrip': False, 'normalized': False, 'special': True}
                                   for i, content in enumerate(CONTROLS)]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'tokenizer.json').write_text(json.dumps(serialized))
    config = json.loads((parent / 'tokenizer_config.json').read_text())
    (directory / 'tokenizer_config.json').write_text(json.dumps({**config, 'extra_special_tokens': CONTROLS}))
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    return directory, {content: first + i for i, content in enumerate(CONTROLS)}


def load_tiny(directory, parent=None):
    """Tests pin their own tiny parent instead of the Granite artifact."""
    reference = Path(parent if parent is not None else directory) / 'tokenizer.json'
    return granite_tokenizer.load(directory, parent, parent_digest=granite_tokenizer.blob_sha1(reference))


def gpt2_regex_runtime(directory):
    """The transformers 5 defect: the serialized split is replaced by GPT-2's byte regex."""
    from tokenizers import Tokenizer, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    broken = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    broken.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    return PreTrainedTokenizerFast(tokenizer_object=broken, eos_token='<|end_of_text|>', pad_token='<|pad|>')


def test_loader_follows_serialized_pipeline_and_checks_every_encode(tmp_path):
    directory = granite_like(tmp_path / 'granite')
    tokenizer, report = load_tiny(directory)
    assert report['fixture_count'] == len(granite_tokenizer.FIXTURE) and report['control_tokens'] == {}
    assert report['eos_token_id'] == tokenizer.convert_tokens_to_ids('<|end_of_text|>')
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'save_draft for Cedar 6460'}],
                                           add_generation_prompt=True, tokenize=False)
    assert prompt.endswith('<|start_of_role|>assistant<|end_of_role|>')
    ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    assert ids == tokenizer.reference.encode(prompt, add_special_tokens=False).ids
    tensor = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
    assert tensor[0].tolist() == ids and tokenizer.checked_encodes == 2
    assert tokenizer.decode(ids) == prompt
    with pytest.raises(ValueError, match='implicit special tokens'):
        tokenizer(prompt)
    with pytest.raises(ValueError, match='one string'):
        tokenizer([prompt], add_special_tokens=False)
    assert tokenizer.checked_encodes == 2
    with pytest.raises(ValueError, match='pinned Granite parent'):
        granite_tokenizer.load(directory, parent_digest='0' * 40)


def test_gpt2_regex_substitution_is_refused_before_and_during_generation(tmp_path):
    directory = granite_like(tmp_path / 'granite')
    canonical, _ = load_tiny(directory)
    broken = gpt2_regex_runtime(directory)
    text = '{"name": "save_draft", "arguments": {"due_date": "2027-04-26"}}'
    assert broken(text, add_special_tokens=False)['input_ids'] != canonical.reference.encode(text).ids
    assert not granite_tokenizer.agrees(broken, canonical.reference)
    assert granite_tokenizer.agrees(canonical.runtime, canonical.reference)
    with pytest.raises(ValueError, match='splitting differs'):
        granite_tokenizer.conformance(broken, canonical.reference)
    guarded = granite_tokenizer.Checked(broken, canonical.reference)
    with pytest.raises(ValueError, match='differs from the canonical tokenizer'):
        guarded(text, add_special_tokens=False)
    assert guarded.checked_encodes == 0


def test_fixture_alone_detects_a_split_change_even_with_matching_metadata(tmp_path, monkeypatch):
    directory = granite_like(tmp_path / 'granite')
    canonical, _ = load_tiny(directory)
    broken = gpt2_regex_runtime(directory)
    monkeypatch.setattr(granite_tokenizer, 'pipeline', lambda serialized: 'same')
    monkeypatch.setattr(granite_tokenizer, 'splitting', lambda serialized: 'same')
    with pytest.raises(ValueError, match='fixture differs'):
        granite_tokenizer.conformance(broken, canonical.reference)


def test_switch_checkpoint_uses_parent_splitting_and_keeps_its_control_token_ids(tmp_path):
    parent = granite_like(tmp_path / 'granite')
    switch, controls = switch_like(parent, tmp_path / 'switch')
    with pytest.raises(ValueError, match='pinned Granite parent'):
        granite_tokenizer.load(switch, parent_digest=granite_tokenizer.blob_sha1(parent / 'tokenizer.json'))
    parent_tokenizer, parent_report = load_tiny(parent)
    tokenizer, report = load_tiny(switch, parent)
    text = '<|answerability|>{"name": "save_draft", "arguments": {"due_date": "2027-04-26"}} Cedar 6460'
    plain = text.removeprefix('<|answerability|>')
    from tokenizers import Tokenizer
    own_file = Tokenizer.from_file(str(switch / 'tokenizer.json'))
    assert own_file.encode(plain).ids != parent_tokenizer.reference.encode(plain).ids
    assert tokenizer(plain, add_special_tokens=False)['input_ids'] == parent_tokenizer(plain, add_special_tokens=False)['input_ids']
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    assert ids[0] == controls['<|answerability|>'] and ids[1:] == parent_tokenizer.reference.encode(plain).ids
    assert report['control_tokens'] == controls
    assert report['splitting_sha256'] == parent_report['splitting_sha256']
    assert report['fixture_sha256'] == parent_report['fixture_sha256']
    assert report['pipeline_sha256'] != parent_report['pipeline_sha256']
    assert len(tokenizer.runtime) == len(parent_tokenizer.runtime) + len(CONTROLS)


@pytest.mark.parametrize('change, message', [
    (lambda s: s['model']['merges'].pop(), 'vocabulary or merges'),
    (lambda s: s['added_tokens'][0].update(id=10 ** 6), 'added-token IDs'),
    (lambda s: s['added_tokens'][-1].update(id=5), 'reuses a vocabulary ID'),
])
def test_switch_with_changed_vocabulary_or_token_ids_is_refused(tmp_path, change, message):
    parent = granite_like(tmp_path / 'granite')
    switch, _ = switch_like(parent, tmp_path / 'switch')
    serialized = json.loads((switch / 'tokenizer.json').read_text())
    change(serialized)
    (switch / 'tokenizer.json').write_text(json.dumps(serialized))
    with pytest.raises(ValueError, match=message):
        load_tiny(switch, parent)


@pytest.mark.skipif(not os.environ.get('NEUROSHARD_GRANITE_TOKENIZER'),
                    reason='set NEUROSHARD_GRANITE_TOKENIZER to the pinned Granite metadata directory')
def test_pinned_granite_tokenizer_matches_declared_hashes():
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    _, report = granite_tokenizer.load(Path(os.environ['NEUROSHARD_GRANITE_TOKENIZER']))
    pinned = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')['tokenizer']
    assert report['pipeline_sha256'] == pinned['pipeline_sha256']
    assert report['fixture_sha256'] == pinned['fixture_sha256']


@pytest.mark.skipif(not (os.environ.get('NEUROSHARD_GRANITE_TOKENIZER') and os.environ.get('NEUROSHARD_GRANITE_SWITCH_TOKENIZER')),
                    reason='set both pinned Granite parent and Switch metadata directories')
def test_pinned_switch_tokenizer_gets_parent_splitting_and_its_twelve_control_ids():
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    parent, switch = Path(os.environ['NEUROSHARD_GRANITE_TOKENIZER']), Path(os.environ['NEUROSHARD_GRANITE_SWITCH_TOKENIZER'])
    artifacts = read(ROOT / granite_tokenizer.ARTIFACTS)['models']['modular']['files']['tokenizer.json']
    assert granite_tokenizer.blob_sha1(switch / 'tokenizer.json') == artifacts['digest']
    assert json.loads((switch / 'tokenizer.json').read_text())['pre_tokenizer'] == GPT2_REGEX
    with pytest.raises(ValueError, match='pinned Granite parent'):
        granite_tokenizer.load(switch)
    _, report = granite_tokenizer.load(switch, parent)
    pinned = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')['tokenizer']
    assert report['fixture_sha256'] == pinned['fixture_sha256']
    assert sorted(report['control_tokens'].values()) == list(range(100352, 100364))
