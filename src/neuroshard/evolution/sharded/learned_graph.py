"""Route raw questions through actual owned experts using a frozen classifier.

This development executor extends the measured graph's routing. Its different
service commitment requires a new end-to-end quality decision before native
admission. The existing native graph and its settlement format stay separate.
"""
import copy
from pathlib import Path

from .. import expert_router, serving_graph
from ..reference_data import identity, sha256
from . import composition

FORMAT = 'neuroshard-learned-graph-service-v1'
MAPPED = 'neuroshard-mapped-graph-service-v1'
COMPOSING = 'neuroshard-learned-composed-service-v2'
ALIASED_COMPOSING = 'neuroshard-aliased-composed-service-v1'
SOURCES = ('src/neuroshard/evolution/expert_router.py',
           'src/neuroshard/evolution/serving_graph.py',
           'src/neuroshard/evolution/sharded/router_features.py',
           'src/neuroshard/evolution/sharded/interpretation.py',
           'src/neuroshard/evolution/sharded/composition.py',
           'src/neuroshard/evolution/sharded/learned_graph.py',
           'scripts/run_native_expert_service.py')


def configuration(graph, model, feature_profile, source_home, route_models=None, *, compose=False):
    expert_router.validate(model)
    result = {'format': FORMAT, 'graph': identity(graph), 'router': model,
            'feature_profile': feature_profile,
            'sources': {name: sha256(Path(source_home) / name) for name in SOURCES}}
    if route_models is not None:
        if compose:
            if (not isinstance(route_models, dict) or set(route_models) != set(model['prototypes'])
                    or any(not isinstance(value, str) for value in route_models.values())
                    or set(route_models.values()) != {'parent', *graph['experts']}
                    or route_models.get(model['fallback']) != 'parent'):
                raise ValueError('Map every learned route onto the retained experts and unchanged fallback')
            result.update(format=ALIASED_COMPOSING, route_models=copy.deepcopy(route_models),
                          composition=composition.ROUTED_FORMAT)
            return result
        if (not isinstance(route_models, dict) or set(route_models) != set(model['prototypes'])
                or any(not isinstance(value, str) for value in route_models.values())
                or set(route_models.values()) != {'parent', 'interpreter', *graph['experts']}
                or len(set(route_models.values())) != len(route_models)
                or route_models.get(model['fallback']) != 'interpreter'):
            raise ValueError('Map distinct learned routes to every installed model with an assistant fallback')
        result.update(format=MAPPED, route_models=copy.deepcopy(route_models))
    elif compose:
        result.update(format=COMPOSING, composition=composition.ROUTED_FORMAT)
    return result


class Decision:
    def __init__(self, question, route):
        self.question, self.route = question, None if route == 'parent' else route

    def select(self, question):
        if question != self.question:
            raise ValueError('A route decision cannot be reused for another question')
        return self.route


def validate_configuration(graph, config, source_home):
    """Check router/model/source commitments before allocating neural weights."""
    mapped = config.get('format') == MAPPED
    aliased = config.get('format') == ALIASED_COMPOSING
    composing = config.get('format') in (COMPOSING, ALIASED_COMPOSING)
    fields = {'format', 'graph', 'router', 'feature_profile', 'sources'}
    if mapped or aliased:
        fields.add('route_models')
    if composing:
        fields.add('composition')
    serving_graph.fields(config, fields, 'Invalid learned serving configuration')
    model = expert_router.validate(config['router'])
    embedding = graph['interpreter_assets']['partitions']['0']['tensors']['model.embed_tokens.weight']
    if (config != configuration(graph, model, config['feature_profile'], source_home,
                               config.get('route_models'), compose=composing)
            or (not mapped and not aliased and (model['fallback'] != 'parent'
                or set(model['prototypes']) != {'parent', *graph['experts']}))
            or model['tokenizer_root'] != graph['tokenizer']['root']
            or model['embedding_root'] != identity(config['feature_profile'])
            or config['feature_profile'].get('embedding_sha256') != embedding['sha256']):
        raise ValueError('Learned service changed its models, features, tokenizer or source')


class LearnedGraphNetwork:
    def __init__(self, network, config, *, source_home, features=None, graph=None):
        # A growing deployment can retain its earlier service on a subset of
        # the installed owners. It must use precisely their accepted weights.
        selected = network.graph if graph is None else serving_graph.validate(graph)
        if (any(selected[key] != network.graph[key] for key in ('parent', 'interpreter_assets',
                'interpreter_prompt', 'tokenizer', 'numerical_profile', 'executor_root'))
                or selected['descriptor']['interpretation'] != network.graph['descriptor']['interpretation']
                or any(value != network.graph['experts'].get(name) for name, value in selected['experts'].items())
                or any(rule not in network.graph['descriptor']['rules'] for rule in selected['descriptor']['rules'])):
            raise ValueError('Retained learned service differs from installed frozen expert paths')
        validate_configuration(selected, config, source_home)
        if network.rank == 0:
            if features is None or features.profile != config['feature_profile']:
                raise ValueError('The embedding owner needs the committed feature extractor')
        elif features is not None:
            raise ValueError('Only the embedding owner loads the routing table')
        self.network, self.config, self.features = network, copy.deepcopy(config), features
        self.graph = copy.deepcopy(selected)
        self.root = identity(config)
        if network.all_owners.exchange(self.root) != [self.root] * network.world_size:
            raise ValueError('Owners installed different learned services')

    def answer(self, question, max_tokens):
        if self.config['format'] == MAPPED:
            raise ValueError('Mapped model routes require the committed conversation executor')
        parts = (composition.independent_questions(question)
                 if self.config['format'] in (COMPOSING, ALIASED_COMPOSING) else None)
        if parts is not None:
            # Choose an expert for each actual subquestion. A new fact and an
            # earlier fact can therefore execute on different frozen tails.
            # This bounded explicit grammar is not a natural-language planner.
            values = [self.single(part, max_tokens) for part in parts]
            calls = [call for value in values for call in value['request']['calls']]
            outputs = [item for value in values for item in value['outputs']]
            serving_graph.payments(self.graph, calls, outputs, 1)
            return {'format': FORMAT+'/response', 'service': self.root, 'graph': identity(self.graph),
                'request': {'question': question, 'max_tokens': max_tokens, 'calls': calls},
                'routing': {'service': self.root, 'composition': composition.ROUTED_FORMAT,
                            'parts': [{'question': part, 'routing': value['routing']}
                                      for part, value in zip(parts, values)]},
                'outputs': outputs, 'text': '; '.join(value['text'] for value in values)}
        return self.single(question, max_tokens)

    def single(self, question, max_tokens):
        net = self.network
        # Validate the raw request before any collective or neural operation.
        serving_graph.selected_calls(self.graph, None, question, max_tokens)
        request = identity({'service': self.root, 'question': question, 'max_tokens': max_tokens})
        if net.all_owners.exchange(request) != [request] * net.world_size:
            raise ValueError('Owners received different learned inference requests')
        packet = None
        if net.rank == 0:
            try:
                packet = {'features': self.features(question)}
            except (ValueError, TypeError) as error:
                packet = {'error': str(error)[:512]}
        packets = net.all_owners.exchange(packet)
        if any(value is not None for value in packets[1:]) or not isinstance(packets[0], dict):
            raise ValueError('Only the embedding owner may produce routing features')
        packet = packets[0]
        if 'error' in packet:
            raise ValueError('Routing feature extraction failed: ' + packet['error'])
        serving_graph.fields(packet, {'features'}, 'Invalid routing feature packet')
        observed = expert_router.select(self.config['router'], packet['features'])
        route = self.config.get('route_models', {}).get(observed['route'], observed['route'])
        decision = Decision(question, route)
        plan = serving_graph.selected_calls(self.graph, decision.route, question, max_tokens)
        routing = {'service': self.root, 'decision': observed, 'features': packet['features']}
        if self.config['format'] == ALIASED_COMPOSING:
            routing['model'] = route
        original = net.net.answer_paths.get('directory')
        if net.interpreted is not None:
            net.net.answer_paths['directory'] = net.interpreted.answer_expert
        try:
            result = net._run(self.graph, question, max_tokens, plan, decision, routing)
        finally:
            if original is not None:
                net.net.answer_paths['directory'] = original
        return {'format': FORMAT + '/response', 'service': self.root, 'routing': routing, **result}

    def replay(self, response):
        """Recompute selection and every neural call; never trust a route trace."""
        if response.get('format') != FORMAT + '/response' or response.get('service') != self.root:
            raise ValueError('Response belongs to a different learned service')
        request = response['request']
        actual = self.answer(request['question'], request['max_tokens'])
        return actual == response, actual
