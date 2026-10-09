"""Versioned chat, tools, private memory and consent for the public assistant.

This is the A6 product surface. It does not serve a public model or complete A6.
A session is not ledger contents and is not training data unless the caller opts in.
"""

import copy
import json
import time

from neuroshard.evolution import assistant_workflow as workflow
from neuroshard.evolution.modular_reference_execution import identity

FORMAT = 'neuroshard-assistant-public/1'
CURRENT = 'a3-cohort3'
PREVIOUS = 'a2-u1'
MEMORY_KEYS = 32
MEMORY_BYTES = 4096


def default_consent():
    return {'training_opt_in': False, 'external_actions': False, 'share_outside_session': False}


def bind(policy, *, name=CURRENT, previous=PREVIOUS, modules=('U1', 'L2', 'L3'), route_policies=None):
    if policy.get('external_effects'):
        raise ValueError('public session refuses external effects')
    body = {'format': FORMAT, 'name': name, 'previous': previous,
            'policy_sha256': identity(policy),
            'tools_sha256': identity(workflow.interface(policy).TOOLS),
            'modules': list(modules), 'limits': copy.deepcopy(policy['limits'])}
    if route_policies:
        if any(item.get('external_effects') for item in route_policies.values()):
            raise ValueError('public routes refuse external effects')
        body['routes'] = {name: {'policy_sha256': identity(item),
                                'tools_sha256': identity(workflow.interface(item).TOOLS)}
                          for name, item in sorted(route_policies.items())}
    return {**body, 'version_sha256': identity(body)}


class Session:
    def __init__(self, policy, world, *, consent=None, name=CURRENT, previous=PREVIOUS, modules=('U1', 'L2', 'L3'),
                 route_policies=None):
        consent = default_consent() if consent is None else dict(consent)
        extra = set(consent) - set(default_consent())
        missing = set(default_consent()) - set(consent)
        if extra or missing:
            raise ValueError('consent fields must be training_opt_in, external_actions, share_outside_session')
        if consent['external_actions']:
            raise ValueError('external actions are not authorized')
        self.policy = policy
        self.route_policies = route_policies
        self.version = bind(policy, name=name, previous=previous, modules=modules, route_policies=route_policies)
        self.consent = consent
        self.tools = workflow.interface(policy)
        if route_policies:
            from neuroshard.evolution import assistant_routing
            self.world = assistant_routing.workspace(route_policies.values()).Workspace(copy.deepcopy(world))
        else:
            self.world = self.tools.Workspace(copy.deepcopy(world))
        self.messages = [{'role': 'system', 'content': policy['system_instruction']}]
        self.calls = []
        self.memory = {}

    def note(self, key, value):
        if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
            raise ValueError('memory note must be a non-empty string pair')
        if key not in self.memory and len(self.memory) >= MEMORY_KEYS:
            raise ValueError('session memory key bound exceeded')
        if len(value.encode()) > MEMORY_BYTES:
            raise ValueError('session memory value bound exceeded')
        self.memory[key] = value

    def rollback_target(self):
        return self.version['previous']

    def training_export(self):
        if not self.consent['training_opt_in']:
            raise ValueError('training export requires explicit opt-in')
        return {'version': self.version, 'messages': copy.deepcopy(self.messages),
                'calls': copy.deepcopy(self.calls), 'memory': copy.deepcopy(self.memory)}

    def export(self):
        payload = {'version': self.version, 'consent': copy.deepcopy(self.consent),
                   'ledger': None, 'training': None, 'messages': [], 'calls': [], 'memory': {}}
        if self.consent['share_outside_session']:
            payload['messages'] = copy.deepcopy(self.messages)
            payload['calls'] = copy.deepcopy(self.calls)
            payload['memory'] = copy.deepcopy(self.memory)
        return payload

    def turn(self, user, respond, route=None):
        if not isinstance(user, str) or not user.strip():
            raise ValueError('user turn must be non-empty text')
        if route is not None:
            self.policy = self.route_policies[route]
            self.tools = workflow.interface(self.policy)
            self.messages[0] = {'role': 'system', 'content': self.policy['system_instruction']}
        self.messages.append({'role': 'user', 'content': user})
        used, final, completed, failure = 0, '', False, None
        started = time.monotonic()
        generations = 0
        responses = []
        for _ in range(self.policy['limits']['model_turns_per_user_turn']):
            generated = respond(copy.deepcopy(self.messages), copy.deepcopy(self.tools.TOOLS))
            generated = {**generated,
                         'request_sha256': identity({'messages': self.messages, 'tools': self.tools.TOOLS})}
            generations += 1
            responses.append(generated)
            if not generated['terminated']:
                failure = 'generation did not terminate within its token/input budget'
                break
            text = generated['text']
            self.messages.append({'role': 'assistant', 'content': text})
            try:
                proposed = self.tools.parse_calls(text)
            except (ValueError, TypeError, RecursionError):
                self.messages.append({'role': 'tool',
                                      'content': json.dumps({'error': 'invalid tool-call syntax or schema'})})
                continue
            if not proposed:
                final, completed = text, bool(text.strip())
                break
            if used + len(proposed) > self.policy['limits']['tool_calls_per_user_turn']:
                failure = 'tool-call budget exhausted'
                break
            for call in proposed:
                response = self.world.execute(call)
                self.calls.append({'call': call, 'result': response})
                used += 1
                self.messages.append({'role': 'tool',
                                      'content': json.dumps(response, ensure_ascii=False, sort_keys=True)})
        if not completed and failure is None:
            failure = 'model-turn budget exhausted'
        return {'completed': completed, 'final_text': final, 'failure': failure,
                'snapshot': self.world.snapshot(), 'version': self.version['name'],
                'seconds': time.monotonic() - started, 'generations': generations,
                'responses': responses, 'route': route}
