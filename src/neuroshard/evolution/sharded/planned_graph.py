"""Bounded conversation decomposition followed by learned owned-shard routing.

This research service has its own source and model commitment. It never replaces
the admitted native graph implicitly. Every planner, argument and answer token
is included in the replayable response, including an unsuccessful plan.
"""
import copy
import json
from pathlib import Path

from .. import expert_router, incremental_facts, serving_graph
from ..reference_data import identity, sha256
from ..schema import integer
from .cached_inference import generate_branch_cached
from .interpretation import example_messages, interpretation
from .learned_graph import LearnedGraphNetwork

FORMAT = 'neuroshard-planned-graph-service-v1'
SOURCES = ('src/neuroshard/evolution/sharded/planned_graph.py',
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
    serving_graph.fields(planner, {'instruction', 'examples', 'max_tokens'}, 'Invalid planner policy')
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


def configuration(graph, learned, planner, source_home):
    planner_prefix(planner)
    return {'format': FORMAT, 'graph': identity(graph), 'learned': copy.deepcopy(learned),
            'planner': copy.deepcopy(planner), 'answer_format': 'ordered-question-answer-v1',
            'sources': {name: sha256(Path(source_home) / name) for name in SOURCES}}


class PlannedGraphNetwork:
    def __init__(self, network, config, *, source_home, features=None):
        serving_graph.fields(config, {'format', 'graph', 'learned', 'planner', 'answer_format', 'sources'},
                             'Invalid planned service configuration')
        if config != configuration(network.graph, config['learned'], config['planner'], source_home):
            raise ValueError('Planned service changed its models, sources or execution rules')
        self.net, self.config = network, copy.deepcopy(config)
        self.router = LearnedGraphNetwork(network, config['learned'], source_home=source_home, features=features)
        self.root, self.prefix = identity(config), planner_prefix(config['planner'])
        if network.all_owners.exchange(self.root) != [self.root] * network.world_size:
            raise ValueError('Owners installed different planned services')
        self.trace = []

    def call(self, model, messages, maximum, purpose):
        net = self.net
        ids = net.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        net.check_context(ids, maximum)
        expert = model != 'interpreter'
        active = [0, 1, 2]
        if expert:
            owner = next(rule['owner'] for rule in net.graph['descriptor']['rules'] if rule['id'] == model)
            active.append(owner)
            network = net.net.networks.get(model)
        else:
            network = net.preserved
        tokens = generate_branch_cached(network, ids, maximum, expert) if network is not None else None
        outputs = net.all_owners.exchange(tokens)
        if (outputs[0] is None or any(outputs[rank] != outputs[0] for rank in active)
                or any(outputs[rank] is not None for rank in range(net.world_size) if rank not in active)):
            raise ValueError('The selected owners disagree on their neural output')
        self.trace.append({'model': model, 'purpose': purpose, 'owners': active,
                           'prompt_ids': ids, 'token_ids': outputs[0]})
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
        decision = expert_router.select(self.config['learned']['router'], packet['features'])
        return {'question': question, 'features': packet['features'], 'decision': decision}

    def answer(self, messages, max_tokens):
        conversation(messages)
        integer(max_tokens, 1, 256)
        request = {'service': self.root, 'messages': copy.deepcopy(messages), 'max_tokens': max_tokens}
        if self.net.all_owners.exchange(identity(request)) != [identity(request)] * self.net.world_size:
            raise ValueError('Owners received different conversations')
        self.trace = []
        raw = self.call('interpreter', self.prefix + messages, self.config['planner']['max_tokens'], 'planning')
        answers, routing, error = [], [], None
        try:
            plan = questions(raw)
        except (ValueError, TypeError):
            plan, error = [], 'invalid_neural_plan'
        for question in plan:
            choice = self.route(question)
            routing.append(choice)
            selected = choice['decision']['route']
            prompt = question
            if selected == 'directory':
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
            answer = self.call('interpreter' if selected == 'parent' else selected,
                [{'role': 'user', 'content': prompt}], max_tokens, 'answer')
            answers.append({'question': question, 'expert': selected, 'text': answer})
        # Failed planning must not silently turn into a fabricated expert answer.
        text = (answers[0]['text'] if len(answers) == 1 else
                '\n\n'.join(row['question'] + '\n' + row['text'] for row in answers)) if error is None else ''
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
