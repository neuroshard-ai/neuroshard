"""Execute the committed graph on five owners and record every neural call.

The numerical kernels are the already evaluated branch, interpretation and
composition implementations. This wrapper loads only an owner's parameters,
checks context before execution and binds actual call outputs for settlement.
"""
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import LlamaConfig

from .. import expert_checkpoint, reference, serving_graph
from ..reference_data import identity, sha256
from . import incremental_state, portable
from .branch import Network
from .branch_groups import OrderedRoutes, RoutedNetwork
from .composition import ComposedAnswers
from .incremental_job import tokenizer_for
from .interpretation import InterpretedNetwork, example_messages
from .model import Partition
from .wire import Wire

FORMAT = 'neuroshard-expert-graph-executor-v1'
PROFILE_FIELDS = {'format', 'runtime', 'threads', 'parameter_limit', 'resident_parameter_limit',
                  'numerical_profile', 'sources'}


def preflight(graph, profile, source_home):
    serving_graph.validate(graph)
    serving_graph.fields(profile, PROFILE_FIELDS, 'Invalid local graph executor profile')
    if (profile['format'] != FORMAT or identity(profile) != graph['executor_root']
            or profile['numerical_profile'] != graph['numerical_profile']
            or not isinstance(profile['sources'], dict) or not profile['sources']):
        raise ValueError('The installed executor differs from the committed graph')
    for name, digest in profile['sources'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(Path(source_home) / path) != digest:
            raise ValueError('Graph execution source changed')
    runtime = reference.configure(profile['runtime']['device'], profile['threads'])
    runtime['allocator'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    if not profile['runtime'] or any(runtime.get(k) != v for k, v in profile['runtime'].items()):
        raise ValueError('Graph execution numerical runtime changed')
    return runtime


class GraphNetwork:
    def __init__(self, graph, profile, *, objects, interpreter, seed, source_home, rank):
        self.runtime = preflight(graph, profile, source_home)
        self.world_size = 3 + len(graph['experts'])
        if (graph['descriptor']['format'] not in (serving_graph.COMPOSED, serving_graph.EXTENSIBLE)
                or type(rank) is not int or not 0 <= rank < self.world_size or not dist.is_initialized()
                or dist.get_rank() != rank or dist.get_world_size() != self.world_size):
            raise ValueError('The graph requires every declared initialized shard owner')
        self.graph, self.rank, self.trace = graph, rank, []
        self.all_owners = Wire(rank, self.world_size)
        self.tokenizer = tokenizer_for({'tokenizer': graph['tokenizer']['root'],
            'tokenizer_files': graph['tokenizer']['files']}, Path(seed))
        if self.tokenizer.eos_token_id != graph['tokenizer']['eos_id']:
            raise ValueError('Graph EOS differs from the tokenizer')
        config = LlamaConfig(**graph['parent']['config'])
        config._attn_implementation = 'sdpa'
        descriptor, device = graph['descriptor'], self.runtime['device']
        self.shard = Partition(config, descriptor['parent_layout'] if rank < 3 else descriptor['expert_layout'],
            min(rank, 3), device, profile['parameter_limit'])
        records = (expert_checkpoint.parent_records(graph['parent']) if rank < 3
                   else graph['experts'][descriptor['rules'][rank - 3]['id']]['tensors'])
        with torch.no_grad():
            for name, parameter in self.shard.named_owned_parameters():
                spec = records[name]
                values = incremental_state.tensor_values(portable.tensor_path(objects, spec['sha256']), spec)
                parameter.copy_(values['weight'])
                parameter.requires_grad_(False)
                del values
        self.shard.eval()
        preserved_shard = None
        if rank < 3:
            manifest = graph['interpreter_assets']['partitions'][str(rank)]
            preserved_shard = Partition(config, descriptor['parent_layout'], rank, device, profile['parameter_limit'])
            if preserved_shard.resident_parameters + self.shard.resident_parameters > profile['resident_parameter_limit']:
                raise ValueError('Combined local models exceed the owner limit')
            preserved_shard.load_weights(interpreter, manifest)
            preserved_shard.eval().requires_grad_(False)
        # A small tail can load minutes before a parent on cold storage. Do not
        # start a subset's connection timeout while another owner is still
        # reading its files. All model bytes are checked before any path forms;
        # normal execution retains the shorter process-group timeout below.
        dist.monitored_barrier(timeout=timedelta(seconds=1200), wait_all_ranks=True)
        timeout = timedelta(seconds=300)
        parent_group = dist.new_group([0, 1, 2], timeout=timeout)
        groups = {rule['id']: dist.new_group([0, 1, 2, rule['owner']], timeout=timeout)
                  for rule in descriptor['rules']}
        self.net = RoutedNetwork(rank, self.shard, self.tokenizer, descriptor['split'],
            OrderedRoutes(descriptor['rules']), parent_group, groups)
        for name, network in self.net.networks.items():
            original = network.generate

            def generate(question, maximum, expert, original=original, name=name):
                ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                    tokenize=True, add_generation_prompt=True)
                self.check_context(ids, maximum)
                result = original(question, maximum, expert)
                if result is not None:
                    self.trace.append({'model': name if expert else 'parent',
                        'prompt_root': identity(ids), 'token_ids': result['ids']})
                return result

            network.generate = generate
            # RoutedNetwork's partial captured the earlier bound method.
            from functools import partial
            self.net.answer_paths[name] = partial(generate, expert=True)
        self.interpreted, self.preserved = None, None
        if rank < 4:
            trained = self.net.networks['directory']
            if rank < 3:
                self.preserved = Network(preserved_shard, trained.wire, trained.parent_wire,
                                         self.tokenizer, descriptor['split'])
            policy = descriptor['interpretation']
            self.interpreted = InterpretedNetwork(trained, self.preserved,
                policy['instruction'], policy['examples'], policy['max_tokens'])
            original_interpret = self.interpreted.interpret

            def interpret(question):
                messages = [*self.interpreted.prefix, {'role': 'user',
                    'content': json.dumps(question) + '\n\n' + self.interpreted.instruction}]
                ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                self.check_context(ids, self.interpreted.max_tokens)
                result = original_interpret(question)
                if result is not None:
                    self.trace.append({'model': 'interpreter', 'prompt_root': identity(ids), 'token_ids': result['ids']})
                return result

            self.interpreted.interpret = interpret
            self.net.answer_paths['directory'] = self.interpreted.answer
        if 'protocol' in self.net.networks:
            self.net.answer_paths['protocol'] = ComposedAnswers(self.net.answer_paths['protocol'], self.tokenizer)
        prefix = example_messages(descriptor['interpretation']['instruction'], descriptor['interpretation']['examples'])
        tokens = self.tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=False)
        if graph['interpreter_prompt'] != {'format': 'name-field-json-v1',
                'messages': identity(prefix), 'tokens': identity(tokens)}:
            raise ValueError('Interpreter prompt serialization changed')
        self.versions = tuple(p._version for _, p in self.shard.named_owned_parameters())
        declaration = {'graph': identity(graph), 'executor': identity(profile), 'rank': rank}
        expected = [{**declaration, 'rank': i} for i in range(self.world_size)]
        if self.all_owners.exchange(declaration) != expected:
            raise ValueError('Owners loaded different graph commitments')

    def check_context(self, ids, maximum):
        if not ids or len(ids) + maximum > self.graph['tokenizer']['max_context']:
            raise ValueError('No silent graph context truncation')

    def verify_unchanged(self):
        if tuple(p._version for _, p in self.shard.named_owned_parameters()) != self.versions:
            raise ValueError('Serving changed a loaded model parameter')
        if self.interpreted:
            self.interpreted.verify_unchanged()

    def answer(self, question, max_tokens, graph=None):
        selected = self.graph if graph is None else serving_graph.validate(graph)
        if any(selected[k] != self.graph[k] for k in ('parent', 'interpreter_assets',
                'interpreter_prompt', 'tokenizer', 'numerical_profile', 'executor_root')):
            raise ValueError('A request cannot replace loaded models or execution rules')
        if (selected['descriptor']['interpretation'] != self.graph['descriptor']['interpretation']
                or any(value != self.graph['experts'].get(name) for name, value in selected['experts'].items())):
            raise ValueError('A request cannot replace a loaded expert or interpreter')
        plan = serving_graph.calls(selected, question, max_tokens)
        return self._run(selected, question, max_tokens, plan, OrderedRoutes(serving_graph.rules(selected)))

    def _run(self, selected, question, max_tokens, plan, routes, routing=None):
        """Execute owned paths; a learned caller additionally binds its decision."""
        request = {'graph': identity(selected), 'question': question, 'max_tokens': max_tokens}
        if routing is not None:
            request['routing'] = routing
        request_root = identity(request)
        if self.all_owners.exchange(request_root) != [request_root] * self.world_size:
            raise ValueError('Owners received different inference requests')
        self.trace = []
        previous = self.net.routes
        self.net.routes = routes
        error = None
        try:
            answer = self.net.answer(question, max_tokens)
        except ValueError as exc:
            # Unused owners still participate in request completion. A bounded
            # context rejection on an active path must not strand them there.
            answer, error = None, str(exc)[:512]
        finally:
            self.net.routes = previous
        observed = self.all_owners.exchange({'answer': answer, 'calls': self.trace, 'error': error})
        errors = [local['error'] for local in observed if local['error'] is not None]
        if errors:
            raise ValueError('Graph execution rejected: ' + errors[0])
        result = observed[0]
        if result['answer'] is None:
            raise ValueError('The embedding owner did not produce an answer')
        # This also checks each actual call's model, length and greedy stop.
        serving_graph.payments(selected, plan, result['calls'], 1)
        for rank, local in enumerate(observed):
            expected = [call for spec, call in zip(plan, result['calls'])
                        if str(rank) in serving_graph.ownership(selected, spec['model'])]
            if local['calls'] != expected or (local['answer'] is not None and local['answer'] != result['answer']):
                raise ValueError('Shard owners disagree on actual neural calls or rendering')
        self.verify_unchanged()
        return {'graph': identity(selected), 'request': {'question': question, 'max_tokens': max_tokens, 'calls': plan},
                'outputs': result['calls'], 'text': result['answer']['text']}
