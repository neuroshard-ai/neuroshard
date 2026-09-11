"""Content-addressed text semantics for the experimental full-model pipeline.

A codec binds the vocabulary, byte encoder/decoder, chat template, special
tokens and implementation versions. Changing one creates a different codec;
it is never a transparent substitution for an existing model or corpus.
"""
import copy
import json
from importlib.metadata import version

from neuroshard.dataflow.store import canonical
from . import schema
from .objects import digest


MAX_TEXT_BYTES = 256 * 1024
MAX_TEXT_TOKENS = 32768
MAX_CODEC_BYTES = 16 * 1024 * 1024
FORMAT = 'neuroshard-text-codec-v1'
TEXT_PACKAGES = ('transformers', 'tokenizers', 'jinja2')


def _template(value):
    """Allow deterministic rendering, without clocks, helper calls or unbounded loops."""
    from jinja2 import Environment, nodes, TemplateSyntaxError
    if not isinstance(value, str) or not 1 <= len(value.encode()) <= 65536:
        raise ValueError('One explicit, bounded chat template is required')
    try:
        parsed=Environment().parse(value)
    except TemplateSyntaxError as exc:
        raise ValueError('Invalid chat template syntax') from exc
    allowed=(nodes.Template,nodes.Output,nodes.TemplateData,nodes.For,nodes.If,nodes.Name,
             nodes.Const,nodes.Getitem,nodes.Getattr,nodes.Compare,nodes.Operand,
             nodes.And,nodes.Or,nodes.Not,nodes.Add)
    names={'messages','message','loop','add_generation_prompt','bos_token','eos_token','pad_token','unk_token'}
    loop_attributes={id(node.node) for node in parsed.find_all(nodes.Getattr)
                     if isinstance(node.node,nodes.Name) and node.node.name=='loop'}
    containers={id(node.node) for node in parsed.find_all(nodes.Getitem)}
    for node in parsed.find_all(nodes.For):
        containers.update((id(node.iter),id(node.target)))
    if len(list(parsed.find_all(nodes.Add)))>32:
        raise ValueError('Chat template concatenation exceeds the rendering budget')
    for node in parsed.find_all(nodes.Node):
        if not isinstance(node,allowed):
            raise ValueError('Chat template requires unsupported or nondeterministic operations')
        if isinstance(node,nodes.Name) and node.name not in names:
            raise ValueError('Chat template references an uncommitted variable')
        if isinstance(node,nodes.Name) and node.name=='loop' and id(node) not in loop_attributes:
            raise ValueError('Chat template cannot render implementation objects')
        if isinstance(node,nodes.Name) and node.name in ('message','messages') and id(node) not in containers:
            raise ValueError('Chat template must render explicit message fields')
        if isinstance(node,nodes.Const) and type(node.value) not in (str,int,bool,type(None)):
            raise ValueError('Chat template constants are outside the deterministic profile')
        if isinstance(node,nodes.Getitem):
            base,key=node.node,node.arg
            direct=(isinstance(base,nodes.Name) and base.name=='message')
            indexed=(isinstance(base,nodes.Getitem) and isinstance(base.node,nodes.Name)
                     and base.node.name=='messages')
            field=(isinstance(key,nodes.Const) and key.value in ('role','content'))
            item=(isinstance(base,nodes.Name) and base.name=='messages' and isinstance(key,nodes.Const)
                  and type(key.value) is int and 0<=key.value<128)
            if item and id(node) not in containers:
                raise ValueError('Chat template must render explicit message fields')
            if not (item or ((direct or indexed) and field)):
                raise ValueError('Chat template indexing is outside the deterministic profile')
        if isinstance(node,nodes.For) and (
                not isinstance(node.iter,nodes.Name) or node.iter.name!='messages'
                or not isinstance(node.target,nodes.Name) or node.target.name!='message'
                or node.recursive or any(node.find_all(nodes.For))):
            raise ValueError('Chat template loops must be bounded by the message list')
        if isinstance(node,nodes.Getattr) and (
                not isinstance(node.node,nodes.Name) or node.node.name!='loop'
                or node.attr not in ('first','last','index','index0','length')):
            raise ValueError('Chat template attribute access is outside the deterministic profile')
    return value


def _renderer(template):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    # Same whitespace semantics as the HF chat renderer. The validated subset
    # needs no callable globals, filters, extensions or implementation objects.
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    env.globals.clear()
    env.filters.clear()
    return env.from_string(template)


class TextCodec:
    def __init__(self, tokenizer, store):
        if not getattr(tokenizer, 'is_fast', False):
            raise ValueError('Text contracts require a serialized fast tokenizer')
        template = _template(tokenizer.chat_template)
        vocabulary = tokenizer.get_vocab()
        size = len(vocabulary)
        schema.integer(size, 16, 131072)
        if sorted(vocabulary.values()) != list(range(size)):
            raise ValueError('Tokenizer IDs must cover a contiguous vocabulary')
        # Backend padding/truncation is transient state in the HF wrapper. Both
        # are disabled by the protocol, which handles bounded windows itself.
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        backend['padding'] = backend['truncation'] = None
        raw = canonical(backend)
        if len(raw) > MAX_CODEC_BYTES:
            raise ValueError('Tokenizer backend exceeds the text contract limit')
        special = {}
        for name in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
            token = getattr(tokenizer, name, None)
            special[name] = str(token) if token is not None else None
        if special['eos_token'] is None:
            raise ValueError('A text contract needs an explicit end-of-response token')
        self.store = store
        self.profile = {
            'format': FORMAT, 'backend': store.put(raw), 'chat_template': template,
            'vocabulary': size, 'special_tokens': special,
            'special_ids': sorted(set(tokenizer.all_special_ids)),
            'runtime': {name: version(name) for name in TEXT_PACKAGES},
            'renderer': 'bounded-jinja-trim-lstrip-v1',
            'decode_cleanup': False, 'input_normalization': 'none',
        }
        self.root = store.put_json(self.profile)
        self.tokenizer = self._restore(self.profile, raw)
        self.renderer = _renderer(template)
        self._check_restored()

    @staticmethod
    def _restore(profile, raw):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer.from_str(raw.decode())
        named = {backend.token_to_id(t) for t in profile['special_tokens'].values() if t is not None}
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            chat_template=profile['chat_template'],
            clean_up_tokenization_spaces=False,
            **{k: v for k, v in profile['special_tokens'].items() if v is not None},
            additional_special_tokens=[
                backend.id_to_token(i) for i in profile['special_ids'] if i not in named
            ],
        )

    def _check_restored(self):
        if len(self.tokenizer) != self.profile['vocabulary']:
            raise ValueError('Restoring special tokens changed vocabulary size')
        if sorted(set(self.tokenizer.all_special_ids)) != self.profile['special_ids']:
            raise ValueError('Restored special tokens differ from the contract')
        schema.integer(self.tokenizer.eos_token_id, 0, len(self.tokenizer)-1)

    @classmethod
    def load(cls, store, root):
        profile = store.json(schema.root(root))
        if (not isinstance(profile, dict) or set(profile) != {
                'format', 'backend', 'chat_template', 'vocabulary', 'special_tokens',
                'special_ids', 'runtime', 'renderer', 'decode_cleanup', 'input_normalization'}
                or profile['format'] != FORMAT or profile['decode_cleanup'] is not False
                or profile['renderer'] != 'bounded-jinja-trim-lstrip-v1'
                or profile['input_normalization'] != 'none'):
            raise ValueError('Unsupported text contract')
        if profile['runtime'] != {name: version(name) for name in TEXT_PACKAGES}:
            raise ValueError('Tokenizer runtime differs from the text contract')
        schema.integer(profile['vocabulary'], 16, 131072)
        if (not isinstance(profile['chat_template'], str)
                or not 1 <= len(profile['chat_template'].encode()) <= 65536
                or not isinstance(profile['special_tokens'], dict)
                or set(profile['special_tokens']) != {'bos_token', 'eos_token', 'pad_token', 'unk_token'}
                or any(v is not None and not isinstance(v, str) for v in profile['special_tokens'].values())
                or not isinstance(profile['special_ids'], list)
                or len(profile['special_ids']) > 1024):
            raise ValueError('Malformed text contract')
        for token in profile['special_ids']:
            schema.integer(token, 0, profile['vocabulary']-1)
        _template(profile['chat_template'])
        raw = store.get(schema.root(profile['backend']))
        if len(raw) > MAX_CODEC_BYTES:
            raise ValueError('Tokenizer backend exceeds the text contract limit')
        obj = cls.__new__(cls)
        obj.store, obj.root, obj.profile = store, root, profile
        obj.tokenizer = cls._restore(profile, raw)
        obj.renderer = _renderer(profile['chat_template'])
        obj._check_restored()
        # Re-serialization catches backend/wrapper inconsistencies, rather than
        # accepting metadata that merely asserts an incorrect vocabulary.
        verified = cls(obj.tokenizer, store)
        if verified.root != root:
            raise ValueError('Text contract is not the canonical tokenizer identity')
        return obj

    def check_model(self, model):
        schema.model(model)
        if model.get('tokenizer_root') != self.root:
            raise ValueError('Model and text tokenizer identities differ')
        if model['config']['vocab_size'] != self.profile['vocabulary']:
            raise ValueError('Tokenizer vocabulary differs from model embedding rows')

    def _check_current(self):
        backend=json.loads(self.tokenizer.backend_tokenizer.to_str())
        backend['padding']=backend['truncation']=None
        special={name:(str(getattr(self.tokenizer,name)) if getattr(self.tokenizer,name) is not None else None)
                 for name in self.profile['special_tokens']}
        if (digest(canonical(backend))!=self.profile['backend']
                or self.tokenizer.chat_template!=self.profile['chat_template']
                or special!=self.profile['special_tokens']
                or sorted(set(self.tokenizer.all_special_ids))!=self.profile['special_ids']):
            raise ValueError('Tokenizer was mutated after its identity was committed')

    def messages(self, messages, training=False):
        self._check_current()
        if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
            raise ValueError('Expected a bounded chat conversation')
        total = 0
        expected = 'user'
        for index, message in enumerate(messages):
            if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                    or not isinstance(message['content'], str)):
                raise ValueError('Messages require only a role and UTF-8 text content')
            role, content = message['role'], message['content']
            if index == 0 and role == 'system':
                pass
            elif role != expected:
                raise ValueError('Chat turns must alternate user and assistant')
            else:
                expected = 'assistant' if role == 'user' else 'user'
            total += len(content.encode('utf-8'))
            if total > MAX_TEXT_BYTES:
                raise ValueError('Conversation exceeds the UTF-8 byte limit')
            if any(token in content for token in self.tokenizer.all_special_tokens):
                raise ValueError('Reserved chat control tokens cannot appear in message content')
        if not any(m['role'] == 'user' for m in messages):
            raise ValueError('A conversation needs a user message')
        if not training and messages[-1]['role'] != 'user':
            raise ValueError('Generation requires a final user message')
        return [{'role':message['role'],'content':message['content']} for message in messages]

    def prompt(self, messages, limit=192):
        schema.integer(limit, 2, 192)
        messages=self.messages(messages)
        ids = self._chat(messages, True)
        if not 2 <= len(ids) <= limit:
            raise ValueError('Prompt exceeds the model input limit; shorten the conversation explicitly')
        return ids

    def _chat(self,messages,generation):
        from jinja2 import TemplateError
        chunks=[]
        size=0
        try:
            for chunk in self.renderer.generate(messages=messages,add_generation_prompt=generation,
                                                 **self.profile['special_tokens']):
                size+=len(chunk.encode())
                if size>2*MAX_TEXT_BYTES:
                    raise ValueError('Rendered chat exceeds the text contract byte limit')
                chunks.append(chunk)
        except TemplateError as exc:
            raise ValueError('Chat template cannot render this conversation') from exc
        rendered=''.join(chunks)
        ids=self.tokenizer.encode(rendered,add_special_tokens=False,truncation=False,padding=False)
        if len(ids)>MAX_TEXT_TOKENS:
            raise ValueError('Conversation exceeds the encoded token limit')
        return ids

    def decode(self, ids):
        self._check_current()
        if not isinstance(ids, list) or len(ids) > MAX_TEXT_TOKENS:
            raise ValueError('Output token sequence exceeds bounds')
        for token in ids:
            schema.integer(token, 0, self.profile['vocabulary']-1)
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def response_windows(self, messages, context=64, response=64, maximum=4):
        """Split real assistant targets, including EOS; rotate turns before tails.

        Each target token appears once. Continuations retain earlier response
        tokens as context. Left context and response-budget truncation are
        reported; no synthetic EOS is trained at an artificial chunk boundary.
        """
        schema.integer(context, 2, 254)
        schema.integer(response, 1, 254)
        schema.integer(maximum, 1, 64)
        if context + response > 256:
            raise ValueError('Text window exceeds the numerical execution profile')
        messages=self.messages(messages, training=True)
        self._chat(messages, False)
        turns = []
        for index, message in enumerate(messages):
            if message['role'] != 'assistant' or not message['content'].strip():
                continue
            prefix = self._chat(messages[:index], True)
            complete = self._chat(messages[:index+1], False)
            if len(prefix)<2 or complete[:len(prefix)] != prefix:
                raise ValueError('Chat template does not have a stable assistant prefix')
            answer = complete[len(prefix):]
            try:
                answer = answer[:answer.index(self.tokenizer.eos_token_id)+1]
            except ValueError as exc:
                raise ValueError('Chat template does not terminate assistant responses') from exc
            if len(answer) <= 1:
                continue
            turns.append((index, prefix, answer))
        windows = []
        depth = 0
        total = sum(len(answer) for _, _, answer in turns)
        while len(windows) < maximum:
            progressed = False
            for index, prefix, answer in turns:
                start = depth * response
                if start >= len(answer) or len(windows) >= maximum:
                    continue
                prior = prefix + answer[:start]
                prompt = prior[-context:]
                target = answer[start:start+response]
                pad = context + response - len(prompt) - len(target)
                windows.append({
                    'tokens': prompt + target + [self.tokenizer.eos_token_id]*pad,
                    'labels': [-100]*len(prompt) + target + [-100]*pad,
                    'tokenizer_root': self.root, 'assistant_index': index,
                    'target_start': start, 'target_end': start+len(target),
                    'context_tokens': len(prompt), 'response_tokens': len(target),
                    'context_truncated': len(prior) > context,
                    'ends_response': start+len(target) == len(answer),
                })
                progressed = True
            if not progressed:
                break
            depth += 1
        used = sum(w['response_tokens'] for w in windows)
        return {'windows': windows, 'response_tokens': total, 'scored_tokens': used,
                'omitted_tokens': total-used, 'truncated': used != total}


def bind_model(store, model_root, codec):
    """Bind existing token rows to their codec without changing any weights."""
    model = copy.deepcopy(store.json(model_root))
    schema.model(model)
    existing = model.get('tokenizer_root')
    if existing is not None:
        codec.check_model(model)
        return model_root, model
    if model['config']['vocab_size'] != codec.profile['vocabulary']:
        raise ValueError('Tokenizer vocabulary differs from model embedding rows')
    model.update(parent=model_root, tokenizer_root=codec.root)
    codec.check_model(model)
    return store.put_json(model), model


def generate(pipeline, codec, messages, max_tokens=32):
    codec.check_model(pipeline.model)
    result = pipeline.generate(codec.prompt(messages), max_tokens=max_tokens,
                               eos_ids=(codec.tokenizer.eos_token_id,))
    return {**result, 'tokenizer_root': codec.root, 'text': codec.decode(result['token_ids'])}
