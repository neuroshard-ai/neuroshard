"""Versioned conversation delivery over the checked owned-shard executor.

This local service does not authorize a model for native settlement. A caller
must consume every event even after its client disconnects from delivery.
"""
import copy
from pathlib import Path

from ..reference_data import identity, save
from ..schema import integer
from ..serving_graph import fields
from .canonical_stream import parameter_versions, stream
from .expert_interface import ExpertInterface
from .fused_graph import commitment
from .fusion_trial import synchronize
from .interface_training import initialize_weights
from .mixture import ProbabilityMixture
from .planned_graph import conversation

FORMAT = 'neuroshard-checked-conversation-service-v1'
BLOCK_FORMAT = 'neuroshard-prefilled-conversation-service-v1'


class FusedService:
    def __init__(self, net, specification, weights):
        blocked = specification.get('format') == BLOCK_FORMAT
        fields(specification, {'format', 'graph', 'gate', 'interfaces', 'context',
                               'decoder' if blocked else 'chunk_tokens', 'max_tokens'},
               'Invalid checked conversation service')
        self.specification = copy.deepcopy(specification)
        spec = self.specification
        if spec['format'] not in (FORMAT, BLOCK_FORMAT) or spec['graph'] != identity(net.graph):
            raise ValueError('Service must bind the installed graph and its executor')
        self.context = integer(spec['context'], 2, 1024)
        self.block_size = None
        if blocked:
            from . import blocked_inference
            decoder = spec['decoder']
            fields(decoder, {'format', 'block_size', 'draft'}, 'Invalid conversation decoder')
            self.block_size = integer(decoder['block_size'], 1, 32)
            if (decoder['format'] != blocked_inference.FORMAT or decoder['draft'] != 'hub'
                    or self.context % self.block_size):
                raise ValueError('Bind the complete supported block decoder and aligned context')
        else:
            self.chunk_tokens = integer(spec['chunk_tokens'], 1, 32)
        self.max_tokens = integer(spec['max_tokens'], 1, min(256, self.context-1))
        interfaces = spec['interfaces']
        if (not isinstance(interfaces, dict)
                or set(interfaces) not in (set(), set(net.graph['experts']))):
            raise ValueError('Install either every owned expert interface or none')
        width = net.graph['parent']['config']['hidden_size']
        layout = spec['gate']['binding']['layout']
        self.gate = ProbabilityMixture(width,
            {name: width for name in ['parent', *net.graph['experts']]},
            rank=layout['rank'], max_context=layout['max_context']).to(net.shard.device_name).eval()
        if self.context > self.gate.max_context or layout != self.gate.descriptor():
            raise ValueError('Service changed its committed gate layout or context limit')
        self.net, self.root, self.interface = net, identity(spec), None
        if net.all_owners.exchange(self.root) != [self.root]*net.world_size:
            raise ValueError('Owners disagree on the installed conversation service')
        if net.rank == 0:
            initialize_weights(self.gate, weights, spec['gate'])
        synchronize(net, self.gate)
        if commitment(self.gate) != spec['gate']['fusion']:
            raise ValueError('Service loaded another conversation gate')
        if net.rank >= 3 and interfaces:
            name = net.graph['descriptor']['rules'][net.rank-3]['id']
            checkpoint = interfaces[name]
            self.interface = ExpertInterface(net.shard, identity(net.graph['experts'][name]),
                rank=checkpoint['binding']['layout']['rank']).eval()
            initialize_weights(self.interface, weights, checkpoint)
        found = net.all_owners.exchange(commitment(self.interface) if self.interface is not None else None)
        expected = [None]*net.world_size
        for rule in net.graph['descriptor']['rules']:
            if interfaces:
                expected[rule['owner']] = interfaces[rule['id']]['fusion']
        if found != expected:
            raise ValueError('Service loaded another expert interface inventory')
        self.versions = self.parameter_versions()

    def parameter_versions(self):
        return tuple(parameter_versions(module) for module in (
            self.net.shard, self.net.preserved.shard if self.net.preserved else None,
            self.gate, self.interface))

    def request(self, messages, max_tokens):
        """Bind a complete conversation to the installed numerical service."""
        messages = copy.deepcopy(conversation(messages))
        maximum = integer(max_tokens, 1, self.max_tokens)
        if any(self.net.all_owners.exchange(self.versions != self.parameter_versions())):
            raise ValueError('Installed conversation weights changed between requests')
        prompt = self.net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(prompt)+maximum > self.context:
            raise ValueError('The complete conversation and output reservation exceed the context limit')
        request = {'service': self.root, 'tokenizer': self.net.graph['tokenizer']['root'],
                   'messages': messages, 'prompt_ids': prompt, 'max_tokens': maximum}
        request_root = identity(request)
        if self.net.all_owners.exchange(request_root) != [request_root]*self.net.world_size:
            raise ValueError('Owners disagree on the complete conversation')
        return request

    def events(self, messages, max_tokens, home):
        """Yield checked token events and one durable completion record.

        Text is the complete decoded prefix, not an unsafe token-wise Unicode
        delta. Messages are supplied in full; there is no silent truncation or
        shared conversation memory between requests.
        """
        request = self.request(messages, max_tokens)
        request_root, prompt, maximum = identity(request), request['prompt_ids'], request['max_tokens']
        home = Path(home)
        home.mkdir(parents=True, exist_ok=False)
        save(home/'request.json', request)
        tokens, chunks = [], []
        execution, options = stream, {'chunk_tokens': self.chunk_tokens} if self.block_size is None else {}
        if self.block_size is not None:
            from . import blocked_inference
            execution, options = blocked_inference.stream, {'block_size': self.block_size}
        for chunk in execution(self.net, self.gate, prompt, maximum, self.context,
                home/'execution', interface=self.interface,
                adapt_interfaces=bool(self.specification['interfaces']), **options):
            tokens.extend(chunk['tokens'])
            chunks.append(identity(chunk))
            yield {'kind': 'tokens', 'request': request_root, 'service': self.root,
                   'index': chunk['index'], 'offset': chunk['offset'], 'tokens': chunk['tokens'],
                   'text': self.net.tokenizer.decode(tokens, skip_special_tokens=True),
                   'end': chunk['end'], 'execution': chunks[-1]}
        result = {'request': request, 'tokens': tokens, 'chunks': chunks, 'complete': True,
                  'text': self.net.tokenizer.decode(tokens, skip_special_tokens=True)}
        save(home/'result.json', result)
        yield {'kind': 'complete', 'request': request_root, 'service': self.root,
               'result': identity(result), 'tokens': len(tokens), 'text': result['text']}

    def verify(self, messages, tokens, max_tokens, home):
        """Reconstruct a whole response without trusting provider cache state."""
        from . import batched_audit, blocked_inference
        request = self.request(messages, max_tokens)
        folder = Path(home)
        folder.mkdir(parents=True, exist_ok=False)
        save(folder/'request.json', request)
        arguments = (self.net, self.gate, request['prompt_ids'], tokens,
                     request['max_tokens'], self.context, folder/'execution')
        options = {'interface': self.interface, 'adapt_interfaces': bool(self.specification['interfaces'])}
        if self.block_size is None:
            checked = batched_audit.verify(*arguments, **options)['result']
        else:
            checked = blocked_inference.verify(*arguments, block_size=self.block_size, **options)
        report = {'service': self.root, 'request': identity(request), 'tokens': list(tokens),
                  'passed': checked['passed'], 'execution': identity(checked)}
        save(folder/'result.json', report)
        return report
