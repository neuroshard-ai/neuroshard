"""Execute the answering policy committed by a native graph."""
from contextlib import contextmanager
import copy

from .. import answering, serving_graph
from ..reference_data import identity


@contextmanager
def selected_graph(network, graph):
    """Use installed shards for one complete immutable policy invocation."""
    model, installed = answering.core(graph), answering.core(network.graph)
    fixed = ('parent', 'interpreter_assets', 'interpreter_prompt', 'tokenizer',
             'numerical_profile', 'executor_root')
    compared = network.comparison is not None and graph == network.comparison
    if (any(model[key] != installed[key] for key in fixed)
            or model['descriptor']['interpretation'] != installed['descriptor']['interpretation']
            or any(rule not in installed['descriptor']['rules'] for rule in model['descriptor']['rules'])
            or not compared and any(value != installed['experts'].get(name)
                                    for name, value in model['experts'].items())):
        raise ValueError('The requested answering system differs from installed model paths')
    original_graph, alternate = network.graph, None
    original_shard = None
    if compared and network.comparison_shard is not None:
        alternate = network.net.networks[network.comparison_name]
        original_shard, alternate.shard = alternate.shard, network.comparison_shard
    network.graph = model
    try:
        yield
    finally:
        network.graph = original_graph
        if alternate is not None:
            alternate.shard = original_shard
        network.verify_unchanged()


def execute(network, graph, messages, maximum, *, on_text=None):
    from .planned_graph import PlannedGraphNetwork, conversation
    serving_graph.validate(graph)
    conversation(messages)
    key = identity(graph)
    with selected_graph(network, graph):
        service = network.answering_services.get(key)
        if service is None:
            config = answering.load(graph, network.policy_store)
            features = None
            if network.rank == 0:
                from .router_features import EmbeddingFeatures
                from .portable import tensor_path
                feature = config['learned']['feature_profile']
                feature_root = identity(feature)
                features = network.answering_features.get(feature_root)
                if features is None:
                    features = EmbeddingFeatures(tensor_path(network.interpreter_home, feature['embedding_sha256']),
                        feature['embedding_sha256'], network.tokenizer, feature['tokenizer_root'],
                        max_tokens=feature['max_tokens'])
                    network.answering_features[feature_root] = features
            service = PlannedGraphNetwork(network, config, source_home=network.source_home,
                features=features, planner_weights_home=network.policy_store.root)
            network.answering_services[key] = service
        response = service.answer(messages, maximum, on_text=on_text)
    request = {'messages': copy.deepcopy(messages), 'max_tokens': maximum}
    return {'graph': key, 'request': request, 'outputs': response['outputs'],
            'text': response['text'], 'answering': response}
