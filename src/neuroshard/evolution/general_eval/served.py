"""Bindings for evaluating the served ``a4-general`` locally, with its own policies, router and units.

What is the served version: a top router (``general-router.json``) sends each turn to A3 (then A3's own
router picks drafting or scheduling) or to the general routes ``documents`` (L9 on the parent, policy
``documents5``) or ``chat`` (C1 on the parent, policy ``chat-tools/1``: calculator and ``check_text``,
at most 400 new tokens per generation and 8 model turns). ``ServedSystem`` runs each item through
``assistant.public.Session`` exactly as ``network.Conversation`` does, minus the paid shards.

Faithfulness: the release check showed that this local stack on the canonical CPU host (Sapphire Rapids,
AMX bf16, 8 threads) reproduces paid network conversations token for token (324/324). A local run on
that host type is therefore the served behaviour; on any other host or on a GPU it is the same weights,
routing and policies with different numerics, and must be reported as such.

``local_route`` needs the frozen parent loaded with transformers; ``responder`` must return, per route, a
fresh ``assistant_serving.cached_responder`` over the parent with that route's unit attached (as
``assistant_general_baseline.module_model`` builds it, digest-checked). The A3 drafting route's first-turn
gate (U1 or the parent fallback) must be applied by that factory as ``network.Conversation.say`` does.
"""

from neuroshard.evolution.general_eval.runner import ServedSystem
from neuroshard.evolution.modular_reference_execution import ROOT, read, sha256

DESCRIPTOR = 'src/neuroshard/client/networks/assistant-testnet-a4.json'
GENERAL_ROUTER = 'config/experiments/assistant-general-router-a4.json'
MODULES = ('U1', 'L2', 'L3', 'L9', 'C1')


def a4_policies(descriptor=DESCRIPTOR):
    """The served route policies, each checked against the digest the descriptor pins."""
    network = read(ROOT / descriptor)
    if network['version'] != 'a4-general':
        raise ValueError('descriptor does not serve a4-general')
    policies = {}
    for name, pin in network['policies'].items():
        if sha256(ROOT / pin['path']) != pin['sha256']:
            raise ValueError(f'policy {name} differs from its pinned digest')
        policies[name] = read(ROOT / pin['path'])
    return policies, network


def route_names(network):
    return {name: row['route'] for name, row in network['routing'].items() if isinstance(row, dict) and 'route' in row}


def routers(network, a3_router_path, general_router_path=GENERAL_ROUTER):
    """A3's router gate and the general router, as stage 0 loads them, each checked against its release pin."""
    for name, path in (('router.json', a3_router_path), ('general-router.json', ROOT / general_router_path)):
        if sha256(path) != network['files'][name]['sha256']:
            raise ValueError(f'{name} differs from the released pin')
    return read(a3_router_path)['gate'], read(ROOT / general_router_path)


def choose_route(router, names, feature, opening, general=None):
    """``network.choose_route`` without the network module (whose imports the canonical runtime lacks).
    ``test_general_eval_step0`` checks that both choose identically."""
    from neuroshard.evolution import assistant_selector as selector
    from neuroshard.evolution import assistant_turn_router as turn_router

    if general is not None:
        top = turn_router.route(turn_router.verify(general), turn_router.context_feature(feature, opening))
        if top != 'a3':
            if top not in names:
                raise ValueError(f'the router chose {top}, which this version does not serve')
            return top, names[top]
    if router.get('rule') == 'logistic-context':
        route = turn_router.route(turn_router.verify(router), turn_router.context_feature(feature, opening))
        if route not in names:
            raise ValueError(f'the router chose {route}, which this version does not serve')
        return route, names[route]
    route = 'scheduling' if selector.choose(router, feature) else 'drafting'
    return route, names.get(route, 2 if route == 'scheduling' else 1)


def local_route(parent, tokenizer, policies, a3_router, general_router, names):
    """``route(text, opening_text)`` with the accepted message-mean feature from the frozen parent, as the
    device computes it (``network.Conversation.say``: the drafting policy's cached prefix, mean final state)."""
    from neuroshard.evolution import assistant_routing as routing

    prefix = routing.message_prefix(parent, tokenizer, policies['drafting'], 'cpu')
    cache, log = {}, []

    def feature(text):
        if text not in cache:
            cache[text] = routing.message_feature(parent, tokenizer, policies['drafting'], text, 'cpu', prefix)
        return cache[text]

    def route(text, opening):
        name, _ = choose_route(a3_router, names, feature(text), feature(opening), general_router)
        log.append(name)
        return name

    route.log = log
    return route


def a4_general(route, responder, descriptor=DESCRIPTOR):
    policies, _ = a4_policies(descriptor)
    return ServedSystem(policies, route, responder, name='a4-general', modules=MODULES)
