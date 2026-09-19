"""Execute the committed graph on five owners and record every neural call.

The numerical kernels are the already evaluated branch, interpretation and
composition implementations. This wrapper loads only an owner's parameters,
checks context before execution and binds actual call outputs for settlement.
"""
from datetime import timedelta
import copy
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
    def __init__(self, graph, profile, *, objects, interpreter, seed, source_home, rank, policy_store=None, mesh=None):
        from ..objects import Objects
        self.source_home = Path(source_home)
        self.interpreter_home = Path(interpreter)
        self.policy_store = policy_store
        self.answering_services = {}
        self.answering_features = {}
        if mesh is not None:
            from .peer_wire import SOURCES
            if any(name not in profile.get('sources', {}) for name in SOURCES):
                raise ValueError('The graph executor must commit its provider transport sources')
        if 'answering' in graph:
            from ..answering import core, load
            from .planned_graph import validate_configuration
            if self.policy_store is None:
                self.policy_store = Objects(Path(objects)/'policies')
            validate_configuration(core(graph), load(graph, self.policy_store), self.source_home)
        self.runtime = preflight(graph, profile, source_home)
        self.profile = copy.deepcopy(profile)
        self.provider_mesh = mesh is not None
        self.world_size = 3 + len(graph['experts'])
        distributed = (dist.is_initialized() and dist.get_rank() == rank
                       and dist.get_world_size() == self.world_size) if mesh is None else (
                       mesh.rank == rank and mesh.world == self.world_size)
        if (graph['descriptor']['format'] not in (serving_graph.COMPOSED, serving_graph.EXTENSIBLE)
                or type(rank) is not int or not 0 <= rank < self.world_size or not distributed):
            raise ValueError('The graph requires every declared initialized shard owner')
        self.graph, self.rank, self.trace = graph, rank, []
        self.all_owners = mesh.group(list(range(self.world_size))) if mesh else Wire(rank, self.world_size)
        self.tokenizer = tokenizer_for({'tokenizer': graph['tokenizer']['root'],
            'tokenizer_files': graph['tokenizer']['files']}, Path(seed))
        if self.tokenizer.eos_token_id != graph['tokenizer']['eos_id']:
            raise ValueError('Graph EOS differs from the tokenizer')
        config = LlamaConfig(**graph['parent']['config'])
        config._attn_implementation = 'sdpa'
        descriptor, device = graph['descriptor'], self.runtime['device']
        self.shard = Partition(config, descriptor['parent_layout'] if rank < 3 else descriptor['expert_layout'],
            min(rank, 3), device, profile['parameter_limit'], inference_only=True)
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
            preserved_shard = Partition(config, descriptor['parent_layout'], rank, device, profile['parameter_limit'],
                                        inference_only=True)
            if preserved_shard.resident_parameters + self.shard.resident_parameters > profile['resident_parameter_limit']:
                raise ValueError('Combined local models exceed the owner limit')
            preserved_shard.load_weights(interpreter, manifest)
            preserved_shard.eval().requires_grad_(False)
        # A small tail can load minutes before a parent on cold storage. Do not
        # start a subset's connection timeout while another owner is still
        # reading its files. All model bytes are checked before any path forms;
        # normal execution retains the shorter process-group timeout below.
        if mesh is None:
            dist.monitored_barrier(timeout=timedelta(seconds=1200), wait_all_ranks=True)
            timeout = timedelta(seconds=300)
            parent_group = dist.new_group([0, 1, 2], timeout=timeout)
            groups = {rule['id']: dist.new_group([0, 1, 2, rule['owner']], timeout=timeout)
                      for rule in descriptor['rules']}
        else:
            loaded = {'graph': identity(graph), 'executor': identity(profile), 'loaded': True}
            if self.all_owners.exchange(loaded) != [loaded] * self.world_size:
                raise ValueError('Providers loaded different graph or numerical commitments')
            parent_group, groups = None, {}
        self.net = RoutedNetwork(rank, self.shard, self.tokenizer, descriptor['split'],
            OrderedRoutes(descriptor['rules']), parent_group, groups, mesh=mesh)
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
        self.comparison = None
        self.comparison_shard = None
        self.comparison_versions = None
        self.resident_parameters = self.shard.resident_parameters + (
            preserved_shard.resident_parameters if preserved_shard is not None else 0)
        self.resident_limit = profile['resident_parameter_limit']
        declaration = {'graph': identity(graph), 'executor': identity(profile), 'rank': rank}
        expected = [{**declaration, 'rank': i} for i in range(self.world_size)]
        if self.all_owners.exchange(declaration) != expected:
            raise ValueError('Owners loaded different graph commitments')

    def check_context(self, ids, maximum):
        if not ids or len(ids) + maximum > self.graph['tokenizer']['max_context']:
            raise ValueError('No silent graph context truncation')

    def rebind(self, mesh):
        """Reuse immutable weights under a fresh, independently fenced request.

        Request KV caches are local to each generation call. Learned service
        objects retain this GraphNetwork, so replacing its wires also updates
        the planner, encoder/reranker collectives and every expert path.
        """
        if (not self.provider_mesh or mesh.rank != self.rank or mesh.world != self.world_size
                or self.comparison is not None):
            raise ValueError('Only the same provider graph can reuse resident weights')
        preflight(self.graph, self.profile, self.source_home)
        self.verify_unchanged()
        self.trace = []
        self.all_owners = mesh.group(list(range(self.world_size)))
        self.net.parent_wire = mesh.group([0, 1, 2])
        for rule in self.graph['descriptor']['rules']:
            branch = self.net.networks.get(rule['id'])
            if branch is not None:
                branch.wire = mesh.group([0, 1, 2, rule['owner']])
                branch.parent_wire = self.net.parent_wire
        if self.preserved is not None:
            self.preserved.wire = self.net.networks['directory'].wire
            self.preserved.parent_wire = self.net.parent_wire
        binding = {'graph': identity(self.graph), 'executor': identity(self.profile),
                   'assignment': mesh.peer.routing['assignment_root']}
        if self.all_owners.exchange(binding) != [binding]*self.world_size:
            raise ValueError('Providers reused different model or assignment commitments')

    def verify_unchanged(self):
        if tuple(p._version for _, p in self.shard.named_owned_parameters()) != self.versions:
            raise ValueError('Serving changed a loaded model parameter')
        if self.interpreted:
            self.interpreted.verify_unchanged()
        if (self.comparison_shard is not None and tuple(p._version for p in self.comparison_shard.parameters())
                != self.comparison_versions):
            raise ValueError('Quality evaluation changed its retained expert weights')

    def install_comparison(self, graph, *, objects):
        """Keep one earlier expert revision for paired quality evaluation.

        Only the changed tail is additionally loaded, on its existing owner.
        It is counted against the resident limit and never overwrites serving
        weights. This is an evaluation facility, not a promotion operation.
        """
        serving_graph.validate(graph)
        if self.comparison is not None:
            raise ValueError('A quality executor pins one immutable comparison')
        fixed = ('parent', 'interpreter_assets', 'interpreter_prompt', 'tokenizer',
                 'numerical_profile', 'executor_root')
        if (any(graph[key] != self.graph[key] for key in fixed)
                or set(graph['experts']) != set(self.graph['experts'])
                or graph['descriptor']['rules'] != self.graph['descriptor']['rules']
                or graph['descriptor']['interpretation'] != self.graph['descriptor']['interpretation']):
            raise ValueError('Compare one expert revision under the same installed paths')
        changed = [name for name in graph['experts'] if graph['experts'][name] != self.graph['experts'][name]]
        if len(changed) != 1:
            raise ValueError('Paired update quality requires exactly one changed expert')
        name = changed[0]
        if self.all_owners.exchange(identity(graph)) != [identity(graph)] * self.world_size:
            raise ValueError('Quality owners disagree on the retained revision')
        error, shard = None, None
        try:
            owner = next(rule['owner'] for rule in graph['descriptor']['rules'] if rule['id'] == name)
            if self.rank == owner:
                checkpoint = graph['experts'][name]
                expert_checkpoint.unpack(graph['parent'], checkpoint)
                shard = Partition(self.shard.config, checkpoint['boundaries'], 3,
                                  self.shard.device_name, self.resident_limit - self.resident_parameters,
                                  inference_only=True)
                with torch.no_grad():
                    for key, parameter in shard.named_owned_parameters():
                        spec = checkpoint['tensors'][key]
                        values = incremental_state.tensor_values(portable.tensor_path(objects, spec['sha256']), spec)
                        parameter.copy_(values['weight'])
                        del values
                shard.eval().requires_grad_(False)
        except (OSError, ValueError, KeyError, TypeError) as failure:
            error = type(failure).__name__
        if any(self.all_owners.exchange(error)):
            raise ValueError('Retained expert is unavailable or exceeds the resident owner limit')
        self.comparison, self.comparison_name, self.comparison_shard = copy.deepcopy(graph), name, shard
        if shard is not None:
            self.comparison_versions = tuple(p._version for p in shard.parameters())
            self.resident_parameters += shard.resident_parameters

    def answer(self, question, max_tokens, graph=None, *, on_text=None):
        selected = self.graph if graph is None else serving_graph.validate(graph)
        if 'answering' in selected:
            from .answering_service import execute
            messages = [{'role': 'user', 'content': question}] if isinstance(question, str) else question
            return execute(self, selected, messages, max_tokens, on_text=on_text)
        if any(selected[k] != self.graph[k] for k in ('parent', 'interpreter_assets',
                'interpreter_prompt', 'tokenizer', 'numerical_profile', 'executor_root')):
            raise ValueError('A request cannot replace loaded models or execution rules')
        if selected != self.comparison and (selected['descriptor']['interpretation'] != self.graph['descriptor']['interpretation']
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
        alternate = self.comparison_shard if selected == self.comparison else None
        compared = self.net.networks[self.comparison_name] if alternate is not None else None
        if compared is not None:
            original_shard, compared.shard = compared.shard, alternate
        error = None
        try:
            answer = self.net.answer(question, max_tokens)
        except ValueError as exc:
            # Unused owners still participate in request completion. A bounded
            # context rejection on an active path must not strand them there.
            answer, error = None, str(exc)[:512]
        finally:
            self.net.routes = previous
            if compared is not None:
                compared.shard = original_shard
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
