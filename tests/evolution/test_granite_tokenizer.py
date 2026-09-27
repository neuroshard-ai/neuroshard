import json
import os
from pathlib import Path

import pytest

from neuroshard.evolution import granite_tokenizer

SPLIT = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
         "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
SPECIAL = ['<|pad|>', '<|end_of_text|>', '<|start_of_role|>', '<|end_of_role|>']
TEMPLATE = ("{% for m in messages %}<|start_of_role|>{{ m.role }}<|end_of_role|>{{ m.content }}<|end_of_text|>\n"
            "{% endfor %}{% if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>{% endif %}")


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


def gpt2_regex_runtime(directory):
    """The transformers 5 defect: the serialized split is replaced by GPT-2's byte regex."""
    from tokenizers import Tokenizer, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    broken = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    broken.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    return PreTrainedTokenizerFast(tokenizer_object=broken, eos_token='<|end_of_text|>', pad_token='<|pad|>')


def test_loader_follows_serialized_pipeline_and_checks_every_encode(tmp_path):
    directory = granite_like(tmp_path / 'granite')
    tokenizer, report = granite_tokenizer.load(directory)
    assert report['fixture_count'] == len(granite_tokenizer.FIXTURE)
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


def test_gpt2_regex_substitution_is_refused_before_and_during_generation(tmp_path):
    directory = granite_like(tmp_path / 'granite')
    canonical, _ = granite_tokenizer.load(directory)
    broken = gpt2_regex_runtime(directory)
    text = '{"name": "save_draft", "arguments": {"due_date": "2027-04-26"}}'
    assert broken(text, add_special_tokens=False)['input_ids'] != canonical.reference.encode(text).ids
    assert not granite_tokenizer.agrees(broken, canonical.reference)
    assert granite_tokenizer.agrees(canonical.runtime, canonical.reference)
    with pytest.raises(ValueError, match='pipeline differs'):
        granite_tokenizer.conformance(broken, canonical.reference)
    guarded = granite_tokenizer.Checked(broken, canonical.reference)
    with pytest.raises(ValueError, match='differs from pinned tokenizer.json'):
        guarded(text, add_special_tokens=False)
    assert guarded.checked_encodes == 0


def test_fixture_alone_detects_a_split_change_even_with_matching_metadata(tmp_path, monkeypatch):
    directory = granite_like(tmp_path / 'granite')
    canonical, _ = granite_tokenizer.load(directory)
    broken = gpt2_regex_runtime(directory)
    monkeypatch.setattr(granite_tokenizer, 'pipeline', lambda serialized: 'same')
    with pytest.raises(ValueError, match='fixture differs'):
        granite_tokenizer.conformance(broken, canonical.reference)


@pytest.mark.skipif(not os.environ.get('NEUROSHARD_GRANITE_TOKENIZER'),
                    reason='set NEUROSHARD_GRANITE_TOKENIZER to the pinned Granite metadata directory')
def test_pinned_granite_tokenizer_matches_declared_hashes():
    from neuroshard.evolution.modular_reference_execution import ROOT, read

    _, report = granite_tokenizer.load(Path(os.environ['NEUROSHARD_GRANITE_TOKENIZER']))
    pinned = read(ROOT / 'config/experiments/assistant-workflow-canonical.json')['tokenizer']
    assert report['pipeline_sha256'] == pinned['pipeline_sha256']
    assert report['fixture_sha256'] == pinned['fixture_sha256']
