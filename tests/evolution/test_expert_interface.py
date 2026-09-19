import copy

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from neuroshard.evolution.reference_data import conversation
from neuroshard.evolution.sharded.expert_interface import supervision


def test_training_spans_separate_answers_and_delimiters_without_inference_labels():
    backend = Tokenizer(models.WordLevel({'<unk>': 0, '<s>': 1, '</s>': 2,
        'user': 3, 'assistant': 4, 'question': 5, 'alpha': 6, 'beta': 7, ';': 8}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                        bos_token='<s>', eos_token='</s>')
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }}</s> {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'alpha; beta'}]
    row = {**conversation(tokenizer, messages, 64), 'messages': messages, 'kind': 'mixed',
           'groups': ['person:Alice', 'topic:version'], 'references': ['alpha', 'beta']}
    labels = supervision(row, tokenizer, ['directory', 'protocol'])
    assert [x for x in labels['directory'] if x != -100] == [6]
    assert [x for x in labels['protocol'] if x != -100] == [7, 2]
    changed = copy.deepcopy(row)
    changed['references'].reverse()
    with pytest.raises(ValueError, match='actual training answer'):
        supervision(changed, tokenizer, ['directory', 'protocol'])
    general = {**row, 'kind': 'general'}
    assert all(set(value) == {-100} for value in supervision(general, tokenizer, ['directory', 'protocol']).values())
