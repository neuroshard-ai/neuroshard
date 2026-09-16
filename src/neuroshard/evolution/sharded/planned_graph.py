"""Bounded conversation decomposition followed by learned owned-shard routing.

This research service has its own source and model commitment. It never replaces
the admitted native graph implicitly. Every planner, argument and answer token
is included in the replayable response, including an unsuccessful plan.
"""
import copy
from contextlib import nullcontext
import json
from pathlib import Path

from .. import expert_router, expert_scope, incremental_facts, serving_graph
from ..reference_data import identity, sha256
from ..schema import integer
from .cached_inference import generate_branch_cached
from .interpretation import example_messages, interpretation
from .learned_graph import LearnedGraphNetwork

FORMAT = 'neuroshard-planned-graph-service-v1'
SOURCES = ('src/neuroshard/evolution/sharded/planned_graph.py',
           'src/neuroshard/evolution/expert_scope.py',
           'src/neuroshard/evolution/sharded/cached_inference.py',
           'src/neuroshard/evolution/sharded/branch.py',
           'src/neuroshard/evolution/sharded/branch_groups.py',
           'src/neuroshard/evolution/sharded/interpretation.py',
           'src/neuroshard/evolution/incremental_facts.py')


def conversation(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 31 or len(messages) % 2 != 1:
        raise ValueError('Require a bounded conversation ending with a user turn')
    size = 0
    for index, message in enumerate(messages):
        serving_graph.fields(message, {'role', 'content'}, 'Invalid conversation message')
        if (message['role'] != ('user' if index % 2 == 0 else 'assistant')
                or not isinstance(message['content'], str) or not message['content'].strip()):
            raise ValueError('Require alternating nonempty user and assistant turns')
        size += len(message['content'].encode())
    if size > 32768:
        raise ValueError('Conversation exceeds its byte bound')
    return messages


def questions(text):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError('Duplicate neural plan key')
        return dict(pairs)
    value = json.loads(text, object_pairs_hook=unique)
    serving_graph.fields(value, {'questions'}, 'Invalid neural question plan')
    rows = value['questions']
    if (not isinstance(rows, list) or not 1 <= len(rows) <= 2
            or any(not isinstance(q, str) or not q.strip() or len(q.encode()) > 2048 for q in rows)
            or len(set(rows)) != len(rows)):
        raise ValueError('Require one or two distinct bounded neural questions')
    return rows


def planner_prefix(planner):
    fields = {'instruction', 'examples', 'max_tokens'}
    if 'repeat_instruction' in planner:
        fields.add('repeat_instruction')
        if type(planner['repeat_instruction']) is not bool:
            raise ValueError('Require an explicit planner reminder policy')
    serving_graph.fields(planner, fields, 'Invalid planner policy')
    integer(planner['max_tokens'], 1, 256)
    if (not isinstance(planner['instruction'], str) or not planner['instruction'].strip()
            or len(planner['instruction'].encode()) > 4096
            or not isinstance(planner['examples'], list) or len(planner['examples']) > 8):
        raise ValueError('Planner prompt exceeds its bound')
    prefix = [{'role': 'system', 'content': planner['instruction']}]
    for pair in planner['examples']:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('Invalid planner example')
        prompt, answer = pair
        conversation([{'role': 'user', 'content': prompt}])
        encoded = json.dumps({'questions': answer})
        questions(encoded)
        prefix.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': encoded}])
    return prefix


def configuration(graph, learned, planner, source_home, expert_prompts=None, general_instruction='', route_scopes=None,
                  planner_weights=None, composer=None):
    planner_prefix(planner)
    prompts = {} if expert_prompts is None else expert_prompts
    if not isinstance(prompts, dict) or not set(prompts) <= set(graph['experts']):
        raise ValueError('Bind prompt contracts only for installed learned experts')
    for prompt in prompts.values():
        serving_graph.fields(prompt, {'prefix', 'suffix'}, 'Invalid expert input contract')
        if any(not isinstance(value, str) or len(value.encode()) > 2048 for value in prompt.values()):
            raise ValueError('Bound the expert input contract')
    if not isinstance(general_instruction, str) or len(general_instruction.encode()) > 2048:
        raise ValueError('Bound the general answer instruction')
    result = {'format': FORMAT, 'graph': identity(graph), 'learned': copy.deepcopy(learned),
            'planner': copy.deepcopy(planner), 'answer_format': 'ordered-question-answer-v1',
            'expert_prompts': copy.deepcopy(prompts),
            'general_instruction': general_instruction,
            'sources': {name: sha256(Path(source_home) / name) for name in SOURCES}}
    if route_scopes is not None:
        expert_scope.validate(route_scopes, learned['router']['prototypes'], learned['router']['fallback'])
        result['route_scopes'] = copy.deepcopy(route_scopes)
    if planner_weights is not None:
        from .planner_training import FORMAT as TRAINING_FORMAT
        binding = planner_weights['binding']
        if (binding.get('format') != TRAINING_FORMAT
                or binding.get('source') != identity(graph['interpreter_assets']['partitions']['2'])):
            raise ValueError('Bind planner weights to the preserved final owner')
        result['planner_weights'] = copy.deepcopy(planner_weights)
        for name in ('src/neuroshard/evolution/sharded/planner_training.py',
                     'src/neuroshard/evolution/sharded/expert_interface.py',
                     'src/neuroshard/evolution/sharded/interface_training.py'):
            result['sources'][name] = sha256(Path(source_home)/name)
    if composer is not None:
        serving_graph.fields(composer, {'instruction', 'max_tokens'}, 'Invalid answer composition policy')
        integer(composer['max_tokens'], 1, 256)
        if not isinstance(composer['instruction'], str) or not 1 <= len(composer['instruction'].encode()) <= 2048:
            raise ValueError('Bound the answer composition instruction')
        result['composer'] = copy.deepcopy(composer)
    return result


class PlannedGraphNetwork:
    def __init__(self, network, config, *, source_home, features=None, planner_weights_home=None):
        fields = {'format', 'graph', 'learned', 'planner', 'answer_format', 'sources',
                  'expert_prompts', 'general_instruction'}
        if 'route_scopes' in config:
            fields.add('route_scopes')
        if 'planner_weights' in config:
            fields.add('planner_weights')
        if 'composer' in config:
            fields.add('composer')
        serving_graph.fields(config, fields,
                             'Invalid planned service configuration')
        if config != configuration(network.graph, config['learned'], config['planner'], source_home,
                                   config['expert_prompts'], config['general_instruction'], config.get('route_scopes'),
                                   config.get('planner_weights'), config.get('composer')):
            raise ValueError('Planned service changed its models, sources or execution rules')
        self.net, self.config = network, copy.deepcopy(config)
        self.router = LearnedGraphNetwork(network, config['learned'], source_home=source_home, features=features)
        self.root, self.prefix = identity(config), planner_prefix(config['planner'])
        if network.all_owners.exchange(self.root) != [self.root] * network.world_size:
            raise ValueError('Owners installed different planned services')
        self.planner_adapter = None
        if 'planner_weights' in config:
            from .expert_interface import ExpertInterface
            from .interface_training import initialize_weights
            from .planner_training import source, installed
            checkpoint = config['planner_weights']
            if network.rank == 2:
                self.planner_adapter = ExpertInterface(network.preserved.shard, source(network),
                    checkpoint['binding']['layout']['rank']).eval()
                initialize_weights(self.planner_adapter, planner_weights_home, checkpoint)
            with installed(network, self.planner_adapter) as adapter_root:
                if adapter_root != checkpoint['fusion']:
                    raise ValueError('The installed planner differs from its service commitment')
        self.trace = []

    def planning_messages(self, messages):
        conversation(messages)
        result = copy.deepcopy(self.prefix + messages)
        if self.config['planner'].get('repeat_instruction', False):
            result[-1]['content'] += '\n\n' + self.config['planner']['instruction']
        return result

    def expert_question(self, model, question):
        prompt = self.config['expert_prompts'].get(model, {'prefix': '', 'suffix': ''})
        return prompt['prefix'] + question + prompt['suffix']

    def model_for_route(self, selected):
        if selected not in self.config['learned']['router']['prototypes']:
            raise ValueError('Unknown learned route')
        return self.config['learned'].get('route_models', {}).get(
            selected, 'interpreter' if selected == 'parent' else selected)

    def answer_messages(self, selected, question, conversation_messages=None, *, whole_request=False):
        # The two older closed-book experts have a canonical question interface.
        # General and structured tasks also need the actual user-provided data
        # and earlier turns. A short planner rewrite is not a replacement for it.
        if conversation_messages is not None and selected not in ('directory', 'protocol'):
            messages = copy.deepcopy(conversation(conversation_messages))
            if not whole_request:
                messages[-1]['content'] += ('\n\nFor this response, answer only the following part '
                    'of my request, using the conversation and data above:\n' + question)
            messages[-1]['content'] = self.expert_question(selected, messages[-1]['content'])
        else:
            messages = [{'role': 'user', 'content': self.expert_question(selected, question)}]
        if selected == 'interpreter' and self.config['general_instruction']:
            messages.insert(0, {'role': 'system', 'content': self.config['general_instruction']})
        return messages

    def call(self, model, messages, maximum, purpose):
        net = self.net
        ids = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        net.check_context(ids, maximum)
        expert = model not in ('parent', 'interpreter')
        active = [0, 1, 2]
        if expert:
            owner = next(rule['owner'] for rule in net.graph['descriptor']['rules'] if rule['id'] == model)
            active.append(owner)
            network = net.net.networks.get(model)
        elif model == 'interpreter':
            network = net.preserved
        else:
            network = next(iter(net.net.networks.values())) if net.rank < 3 else None
        planner_root, scope = None, nullcontext()
        if purpose == 'planning' and 'planner_weights' in self.config:
            from .planner_training import installed
            scope = installed(net, self.planner_adapter)
        with scope as planner_root:
            if planner_root is not None and planner_root != self.config['planner_weights']['fusion']:
                raise ValueError('Planner weights changed after service installation')
            tokens = generate_branch_cached(network, ids, maximum, expert) if network is not None else None
        outputs = net.all_owners.exchange(tokens)
        if (outputs[0] is None or any(outputs[rank] != outputs[0] for rank in active)
                or any(outputs[rank] is not None for rank in range(net.world_size) if rank not in active)):
            raise ValueError('The selected owners disagree on their neural output')
        self.trace.append({'model': model, 'purpose': purpose, 'owners': active,
                           'prompt_ids': ids, 'token_ids': outputs[0]})
        if planner_root is not None:
            self.trace[-1]['planner_adapter'] = planner_root
        return net.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def route(self, question):
        packet = None
        if self.net.rank == 0:
            try:
                packet = {'features': self.router.features(question)}
            except ValueError as error:
                packet = {'error': str(error)[:256]}
        packets = self.net.all_owners.exchange(packet)
        if any(value is not None for value in packets[1:]):
            raise ValueError('Only the embedding owner produces routing features')
        packet = packets[0]
        if 'error' in packet:
            raise ValueError('Routing failed: ' + packet['error'])
        model = self.config['learned']['router']
        allowed = (expert_scope.eligible(question, self.config['route_scopes'], model['prototypes'],
                   model['fallback']) if 'route_scopes' in self.config else None)
        decision = expert_router.select(model, packet['features'], eligible=allowed)
        return {'question': question, 'features': packet['features'], 'decision': decision}

    def composition_messages(self, messages, answers):
        """Give the answerer actual owned results and the whole conversation."""
        result = copy.deepcopy(conversation(messages))
        payload = [{'question': row['question'], 'source': row['expert'], 'response': row['text']}
                   for row in answers]
        result[-1]['content'] += '\n\nSpecialist responses:\n'+json.dumps(payload, sort_keys=True)
        return [{'role': 'system', 'content': self.config['composer']['instruction']}, *result]

    def answer(self, messages, max_tokens):
        conversation(messages)
        integer(max_tokens, 1, 256)
        request = {'service': self.root, 'messages': copy.deepcopy(messages), 'max_tokens': max_tokens}
        if self.net.all_owners.exchange(identity(request)) != [identity(request)] * self.net.world_size:
            raise ValueError('Owners received different conversations')
        self.trace = []
        raw = self.call('interpreter', self.planning_messages(messages), self.config['planner']['max_tokens'], 'planning')
        answers, routing, error = [], [], None
        try:
            plan = questions(raw)
        except (ValueError, TypeError):
            plan, error = [], 'invalid_neural_plan'
        for question in plan:
            # A single unambiguous user request need not lose its payload merely
            # because the planner expressed its instruction more concisely.
            routing_question = messages[0]['content'] if len(plan) == len(messages) == 1 else question
            choice = self.route(routing_question)
            routing.append(choice)
            selected = choice['decision']['route']
            model = self.model_for_route(selected)
            prompt = routing_question
            if model == 'directory':
                policy = self.net.graph['descriptor']['interpretation']
                prefix = example_messages(policy['instruction'], policy['examples'])
                parsed_text = self.call('interpreter', prefix + [{'role': 'user',
                    'content': json.dumps(question) + '\n\n' + policy['instruction']}],
                    policy['max_tokens'], 'directory_arguments')
                parsed = interpretation(parsed_text, question)
                if parsed is None:
                    error = 'invalid_directory_arguments'
                    break
                prompt = incremental_facts.question({'name': parsed['name']}, parsed['field'], 'train', 0)
            answer = self.call(model, self.answer_messages(model, prompt, messages,
                whole_request=len(plan) == 1), max_tokens, 'answer')
            answers.append({'question': question, 'expert': selected, 'text': answer})
        # Failed planning must not silently turn into a fabricated expert answer.
        text = (answers[0]['text'] if len(answers) == 1 else
                '\n\n'.join(row['question'] + '\n' + row['text'] for row in answers)) if error is None else ''
        if (error is None and 'composer' in self.config
                and (len(answers) > 1 or any(row['expert'] in self.net.graph['experts'] for row in answers))):
            text = self.call('interpreter', self.composition_messages(messages, answers),
                min(max_tokens, self.config['composer']['max_tokens']), 'composition')
        result = {'format': FORMAT + '/response', 'service': self.root, 'request': request,
                  'status': 'completed' if error is None else 'needs_clarification', 'error': error,
                  'plan': plan, 'routing': routing, 'outputs': self.trace, 'answers': answers, 'text': text,
                  'generated_tokens': sum(len(row['token_ids']) for row in self.trace)}
        self.net.verify_unchanged()
        if self.net.all_owners.exchange(identity(result)) != [identity(result)] * self.net.world_size:
            raise ValueError('Owners disagree on the complete planned response')
        return result

    def replay(self, response):
        if response.get('format') != FORMAT + '/response' or response.get('service') != self.root:
            raise ValueError('Response belongs to a different planned service')
        actual = self.answer(response['request']['messages'], response['request']['max_tokens'])
        return actual == response, actual
