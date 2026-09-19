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
           'src/neuroshard/evolution/answering.py',
           'src/neuroshard/evolution/sharded/answering_service.py',
           'src/neuroshard/evolution/expert_scope.py',
           'src/neuroshard/evolution/sharded/cached_inference.py',
           'src/neuroshard/evolution/sharded/branch.py',
           'src/neuroshard/evolution/sharded/branch_groups.py',
           'src/neuroshard/evolution/sharded/interpretation.py',
           'src/neuroshard/evolution/incremental_facts.py',
           'src/neuroshard/evolution/serving_stream.py')


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
    if 'output_format' in planner:
        from ..answer_plan import FORMAT as ANSWER_PLAN
        fields.add('output_format')
        if planner['output_format'] != ANSWER_PLAN:
            raise ValueError('Unknown neural planner output format')
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
        encoded = json.dumps(answer if 'output_format' in planner else {'questions': answer})
        if 'output_format' in planner:
            from ..answer_plan import parse
            parse(encoded)
        else:
            questions(encoded)
        prefix.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': encoded}])
    return prefix


def configuration(graph, learned, planner, source_home, expert_prompts=None, general_instruction='', route_scopes=None,
                  planner_weights=None, composer=None, answer_policy=None, request_policy=None, general_answer_policy=None,
                  semantic_questions=None, question_reranker=None):
    planner_prefix(planner)
    prompts = {} if expert_prompts is None else expert_prompts
    if not isinstance(prompts, dict) or not set(prompts) <= set(graph['experts']):
        raise ValueError('Bind prompt contracts only for installed learned experts')
    for prompt in prompts.values():
        fields = {'prefix', 'suffix'} | ({'context'} if 'context' in prompt else set())
        serving_graph.fields(prompt, fields, 'Invalid expert input contract')
        if 'context' in prompt and prompt['context'] not in ('standalone', 'conversation'):
            raise ValueError('Declare whether an expert accepts standalone questions or conversation')
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
    if answer_policy is not None:
        from ..answer_plan import validate as validate_answer_policy
        validate_answer_policy(answer_policy, graph['experts'])
        if planner.get('output_format') != answer_policy['format']:
            raise ValueError('The value policy differs from the planner output schema')
        result['answer_policy'] = copy.deepcopy(answer_policy)
        name = 'src/neuroshard/evolution/answer_plan.py'
        result['sources'][name] = sha256(Path(source_home)/name)
    elif 'output_format' in planner:
        raise ValueError('A typed planner requires its complete value policy')
    if request_policy is not None:
        from ..request_planning import FORMAT as REQUEST_POLICY, ASSISTANT_POLICIES
        if request_policy not in (REQUEST_POLICY, *ASSISTANT_POLICIES) or answer_policy is not None:
            raise ValueError('Require the declared untyped request-preservation policy')
        if request_policy in ASSISTANT_POLICIES and 'fallback_guard' not in learned['router']:
            raise ValueError('The general-first policy requires its learned fallback guard')
        result['request_policy'] = request_policy
        name = 'src/neuroshard/evolution/request_planning.py'
        result['sources'][name] = sha256(Path(source_home)/name)
    if general_answer_policy is not None:
        from .. import general_answer, request_planning
        if (general_answer_policy != general_answer.FORMAT
                or request_policy not in request_planning.ASSISTANT_POLICIES):
            raise ValueError('Bind worked general answers to the complete general-routing policy')
        result['general_answer_policy'] = general_answer_policy
        name = 'src/neuroshard/evolution/general_answer.py'
        result['sources'][name] = sha256(Path(source_home)/name)
    if semantic_questions is not None:
        from ..semantic_questions import ADMISSION_FORMAT, validate as validate_semantics
        validate_semantics(semantic_questions, learned['router']['prototypes'])
        if semantic_questions['format'] == ADMISSION_FORMAT:
            from ..request_planning import ASSISTANT_POLICIES
            preserved = semantic_questions['preserved_router']
            if any(preserved[name] != learned['router'][name]
                   for name in ('dimensions', 'embedding_root', 'tokenizer_root', 'fallback')):
                raise ValueError('Preserved admission routing changed its feature interpretation')
            if request_policy in ASSISTANT_POLICIES and 'fallback_guard' not in preserved:
                raise ValueError('Preserve the accepted general-assistant guard')
        result['semantic_questions'] = copy.deepcopy(semantic_questions)
        for name in ('src/neuroshard/evolution/semantic_questions.py',
                     'src/neuroshard/evolution/sharded/semantic_features.py'):
            result['sources'][name] = sha256(Path(source_home)/name)
    if question_reranker is not None:
        from .. import question_reranking
        if semantic_questions is None:
            raise ValueError('Question reranking requires declared semantic admission')
        world = len(graph['parent']['boundaries']) - 1 + len(graph['experts'])
        question_reranking.validate(question_reranker, semantic_questions, world)
        result['question_reranker'] = copy.deepcopy(question_reranker)
        for name in ('src/neuroshard/evolution/question_reranking.py',
                     'src/neuroshard/evolution/sharded/question_reranker.py'):
            result['sources'][name] = sha256(Path(source_home)/name)
    return result


def validate_configuration(graph, config, source_home):
    fields = {'format', 'graph', 'learned', 'planner', 'answer_format', 'sources',
                  'expert_prompts', 'general_instruction'}
    fields |= {'route_scopes', 'planner_weights', 'composer', 'answer_policy', 'request_policy',
               'general_answer_policy', 'semantic_questions', 'question_reranker'} & set(config)
    serving_graph.fields(config, fields, 'Invalid planned service configuration')
    if config != configuration(graph, config['learned'], config['planner'], source_home,
                               config['expert_prompts'], config['general_instruction'], config.get('route_scopes'),
                               config.get('planner_weights'), config.get('composer'), config.get('answer_policy'),
                               config.get('request_policy'), config.get('general_answer_policy'),
                               config.get('semantic_questions'), config.get('question_reranker')):
        raise ValueError('Planned service changed its models, sources or execution rules')
    from .learned_graph import validate_configuration as validate_learned
    validate_learned(graph, config['learned'], source_home)


class PlannedGraphNetwork:
    def __init__(self, network, config, *, source_home, features=None, planner_weights_home=None):
        validate_configuration(network.graph, config, source_home)
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
        if 'semantic_questions' in config:
            from ..semantic_questions import Index, PARAMETERS
            from .semantic_features import SemanticFeatures
            policy = config['semantic_questions']
            self.semantic_index = Index(policy, config['learned']['router']['prototypes'])
            self.semantic_features = None
            if network.rank == 0:
                key = identity(policy['encoder'])
                if key not in network.answering_features:
                    if network.resident_parameters + PARAMETERS > network.resident_limit:
                        raise ValueError('The semantic encoder exceeds its owner resident limit')
                    network.answering_features[key] = SemanticFeatures(policy['encoder'],
                        network.policy_store.root, network.runtime['device'])
                    network.resident_parameters += PARAMETERS
                self.semantic_features = network.answering_features[key]
        self.question_reranker = None
        if 'question_reranker' in config:
            from .question_reranker import QuestionReranker
            reranking = config['question_reranker']
            profile = reranking['model']
            if network.rank == reranking['owner']:
                key = identity(profile)
                if key not in network.answering_features:
                    if network.resident_parameters + profile['parameters'] > network.resident_limit:
                        raise ValueError('The question reranker exceeds its owner resident limit')
                    network.answering_features[key] = QuestionReranker(profile,
                        network.policy_store.root, network.runtime['device'])
                    network.resident_parameters += profile['parameters']
                self.question_reranker = network.answering_features[key]

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
        mapped = self.config['learned'].get('route_models', {}).get(
            selected, 'interpreter' if selected == 'parent' else selected)
        from .learned_graph import ALIASED_COMPOSING
        if (self.config['learned'].get('format') == ALIASED_COMPOSING
                and selected == self.config['learned']['router']['fallback'] and mapped == 'parent'):
            # The lower-level composing graph names its shared prefix parent.
            # Planned conversation uses the preserved assistant for that same
            # fallback, including when specialist routes alias an older tail.
            return 'interpreter'
        return mapped

    def answer_messages(self, selected, question, conversation_messages=None, *, whole_request=False):
        # The two older closed-book experts have a canonical question interface.
        # General and structured tasks also need the actual user-provided data
        # and earlier turns. A short planner rewrite is not a replacement for it.
        standalone = (selected in ('directory', 'protocol') or
                      self.config['expert_prompts'].get(selected, {}).get('context') == 'standalone')
        if conversation_messages is not None and not standalone:
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
            stream = (purpose == 'composition' or purpose in ('answer', 'general_answer')
                      and getattr(self, 'stream_answer', False))
            callback = (self.observer.tokens(net.tokenizer, worked=purpose == 'general_answer')
                        if stream and net.rank == 0 and hasattr(self, 'observer')
                        and self.observer.callback is not None else None)
            tokens = (generate_branch_cached(network, ids, maximum, expert, on_tokens=callback)
                      if network is not None else None)
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
        from ..semantic_questions import ADMISSION_FORMAT
        semantic = self.config.get('semantic_questions', {})
        if semantic.get('format') == ADMISSION_FORMAT:
            # New coarse gates cannot acquire a request before semantic admission.
            # This also preserves the accepted route for requests whose context
            # or structure makes standalone question matching inappropriate.
            model = semantic['preserved_router']
            base_allowed = None if allowed is None else set(allowed) & set(model['prototypes'])
            decision = expert_router.select(model, packet['features'], eligible=base_allowed)
            if allowed is not None:
                decision['eligible'] = allowed
        else:
            decision = expert_router.select(model, packet['features'], eligible=allowed)
        from ..request_planning import ASSISTANT_POLICIES
        guard = decision.get('fallback_guard')
        if (self.config.get('request_policy') in ASSISTANT_POLICIES and guard
                and guard['confident'] and guard['route'] == model['fallback']):
            # A newly added domain cannot override a confident learned general
            # decision. Retain every original gate observation for replay.
            decision = {**decision, **{name: guard[name] for name in ('route', 'nearest', 'confident', 'margin')},
                        'general_guard_applied': True}
        return {'question': question, 'features': packet['features'], 'decision': decision}

    def composition_messages(self, messages, answers):
        """Give the answerer actual owned results and the whole conversation."""
        result = copy.deepcopy(conversation(messages))
        payload = [{'question': row['question'], 'source': row['expert'], 'response': row['text']}
                   for row in answers]
        result[-1]['content'] += '\n\nSpecialist responses:\n'+json.dumps(payload, sort_keys=True)
        return [{'role': 'system', 'content': self.config['composer']['instruction']}, *result]

    def semantic_route(self, choice):
        """Recognize a training question's meaning; keep all answers in weights."""
        from ..request_planning import atomic_request
        if ('semantic_questions' not in self.config or 'semantic' in choice
                or atomic_request([{'role': 'user', 'content': choice['question']}]) is None):
            return choice
        packet = None
        if self.net.rank == 0:
            try:
                packet = {'encoding': self.semantic_features(choice['question'])}
            except ValueError as error:
                packet = {'error': str(error)[:256]}
        packets = self.net.all_owners.exchange(packet)
        if any(value is not None for value in packets[1:]) or not isinstance(packets[0], dict):
            raise ValueError('Only the declared owner computes semantic features')
        packet = packets[0]
        if 'error' in packet:
            raise ValueError('Semantic routing failed: '+packet['error'])
        encoding = packet['encoding']
        if encoding['profile'] != identity(self.config['semantic_questions']['encoder']):
            raise ValueError('Semantic features changed the committed encoder')
        decision = self.semantic_index.select(encoding['features'])
        selected = decision['selected']
        allowed = choice['decision'].get('eligible')
        if selected is not None and allowed is not None and selected['route'] not in allowed:
            decision = {**decision, 'selected': None, 'excluded_by_scope': True}
            selected = None
        if selected is not None and 'question_reranker' in self.config:
            decision = self.rerank_question(choice['question'], decision)
            selected = decision['selected']
        result = {**choice, 'semantic': {**decision, 'encoding': encoding}}
        if selected is not None:
            result['previous_decision'] = copy.deepcopy(choice['decision'])
            result['decision'] = {'route': selected['route'], 'method': decision['policy'],
                                  'training_id': decision['training_id']}
        return result

    def rerank_question(self, question, decision):
        from .. import question_reranking
        policy = self.config['question_reranker']
        semantic = self.config['semantic_questions']
        candidates = question_reranking.candidates(policy, semantic, decision['selected']['route'])
        if not candidates:
            return decision
        owner = policy['owner']
        packet = None
        if self.net.rank == owner:
            try:
                packet = {'execution': self.question_reranker(question, candidates)}
            except ValueError as error:
                packet = {'error': str(error)[:256]}
        packets = self.net.all_owners.exchange(packet)
        if (any(value is not None for rank, value in enumerate(packets) if rank != owner)
                or not isinstance(packets[owner], dict)):
            raise ValueError('Only the declared owner computes question-pair scores')
        packet = packets[owner]
        if 'error' in packet:
            raise ValueError('Question reranking failed: '+packet['error'])
        return question_reranking.select(policy, semantic, decision, question, packet['execution'])

    def answer_atom(self, selected, question, routing_question, messages, max_tokens, *, whole_request):
        """Use the same expert input contract for serving and diagnostic controls."""
        model = self.model_for_route(selected)
        prompt, argument = routing_question, None
        if model == 'directory':
            policy = self.net.graph['descriptor']['interpretation']
            prefix = example_messages(policy['instruction'], policy['examples'])
            parsed_text = self.call('interpreter', prefix + [{'role': 'user',
                'content': json.dumps(question) + '\n\n' + policy['instruction']}],
                policy['max_tokens'], 'directory_arguments')
            parsed = interpretation(parsed_text, question)
            if parsed is None:
                return None, None, 'invalid_directory_arguments'
            argument = {'question': question, 'arguments': parsed}
            prompt = incremental_facts.question({'name': parsed['name']}, parsed['field'], 'train', 0)
        from .. import general_answer
        if (model == 'interpreter' and 'general_answer_policy' in self.config
                and general_answer.applies(messages)):
            context = [message for message in self.answer_messages(model, prompt, messages,
                whole_request=whole_request) if message['role'] != 'system']
            raw = self.call(model, general_answer.messages(context), general_answer.MAX_TOKENS, 'general_answer')
            try:
                text = general_answer.visible(raw)
                if (self.trace[-1]['token_ids'][-1] != self.net.tokenizer.eos_token_id
                        or len(self.net.tokenizer.encode(text, add_special_tokens=False)) > max_tokens):
                    raise ValueError('Require a completed answer inside the visible output allowance')
            except ValueError:
                return None, None, 'invalid_general_answer'
        else:
            text = self.call(model, self.answer_messages(model, prompt, messages,
                whole_request=whole_request), max_tokens, 'answer')
        return {'question': question, 'expert': selected, 'text': text}, argument, None

    def answer(self, messages, max_tokens, *, on_text=None):
        from ..serving_stream import Observer
        self.observer = Observer(on_text if on_text is not None and self.net.rank == 0 else None)
        conversation(messages)
        integer(max_tokens, 1, 256)
        request = {'service': self.root, 'messages': copy.deepcopy(messages), 'max_tokens': max_tokens}
        if self.net.all_owners.exchange(identity(request)) != [identity(request)] * self.net.world_size:
            raise ValueError('Owners received different conversations')
        self.trace = []
        from .. import request_planning
        from ..semantic_questions import ADMISSION_FORMAT
        preserve = 'request_policy' in self.config
        general_first = self.config.get('request_policy') in request_planning.ASSISTANT_POLICIES
        span_policy = self.config.get('request_policy')
        explicit = (request_planning.explicit_questions(messages,
                    extended=span_policy in (request_planning.SPAN_POLICY, request_planning.MODAL_SPAN_POLICY),
                    modals=span_policy == request_planning.MODAL_SPAN_POLICY)
                    if span_policy in (request_planning.LOSSLESS_POLICY, request_planning.SPAN_POLICY,
                                       request_planning.MODAL_SPAN_POLICY) else None)
        preliminary = self.route(request_planning.routing_context(messages)) if general_first else None
        if preliminary is not None and len(messages) == 1:
            preliminary = self.semantic_route(preliminary)
        general = (preliminary is not None
            and request_planning.atomic_request([messages[-1]]) is not None
            and preliminary['decision']['route'] == self.config['learned']['router']['fallback'])
        direct = (messages[-1]['content'] if general else request_planning.atomic_request(messages)) if general_first else (
            request_planning.direct_question(messages) if preserve else None)
        raw = (json.dumps({'questions': explicit}) if explicit is not None else
               json.dumps({'questions': [direct]}) if direct is not None else
               self.call('interpreter', self.planning_messages(messages),
                         self.config['planner']['max_tokens'], 'planning'))
        answers, routing, arguments, error, program, rendering = [], [], [], None, None, None
        try:
            if 'answer_policy' in self.config:
                from ..answer_plan import parse
                program = parse(raw)
                plan = program['questions']
            else:
                plan = questions(raw)
        except (ValueError, TypeError):
            plan, error = [], 'invalid_neural_plan'
        planning = {'path': 'explicit' if explicit is not None else
                    'general' if general else 'direct' if direct is not None else 'neural',
                    'initial_questions': list(plan), 'repair': 'none'}
        if preliminary is not None:
            planning['preselection'] = preliminary
        if (preserve and explicit is None and not (general_first and direct is not None)
                and error is None and request_planning.needs_repair(plan)):
            try:
                repair_input = request_planning.repair_messages(messages, plan)
            except ValueError:
                plan, error, planning['repair'] = [], 'invalid_reference_repair', 'rejected'
            else:
                # Numerical disagreement or execution failure must propagate.
                repaired = self.call('interpreter', repair_input,
                                     request_planning.REPAIR_TOKENS, 'planning_repair')
                try:
                    plan = request_planning.repair_questions(plan, repaired, messages)
                    planning['repair'] = 'accepted'
                except (ValueError, TypeError):
                    plan, error, planning['repair'] = [], 'invalid_reference_repair', 'rejected'
        for question in plan:
            # A single unambiguous user request need not lose its payload merely
            # because the planner expressed its instruction more concisely.
            routing_question = messages[0]['content'] if len(plan) == len(messages) == 1 else question
            choice = preliminary if general else self.route(routing_question)
            resolved_admission = (self.config.get('semantic_questions', {}).get('format') == ADMISSION_FORMAT
                                  and planning['path'] == 'neural')
            if len(messages) == 1 or resolved_admission:
                if (preliminary is not None and 'semantic' in preliminary
                        and preliminary['question'] == routing_question):
                    choice = preliminary
                choice = self.semantic_route(choice)
            routing.append(choice)
            selected = choice['decision']['route']
            canonical = choice.get('semantic', {}).get('selected')
            if canonical is not None:
                routing_question = canonical['question']
            self.stream_answer = (len(plan) == 1 and (program is None or program['render'] == 'assistant')
                and not ('composer' in self.config and selected in self.net.graph['experts']))
            answer, argument, error = self.answer_atom(selected, question, routing_question,
                messages, max_tokens, whole_request=len(plan) == 1)
            if error is not None:
                break
            if argument is not None:
                arguments.append(argument)
            answers.append(answer)
            # In the deterministic question/answer rendering, a completed part
            # is already visible answer content. Deliver it while later parts
            # execute, without exposing planning, composition inputs or a
            # structured program whose final rendering may still reject it.
            if (len(plan) > 1 and 'composer' not in self.config
                    and (program is None or program['render'] == 'assistant')
                    and all(row['text'].strip() for row in answers)):
                self.observer.text('\n\n'.join(row['question'] + '\n' + row['text'] for row in answers))
        # Failed planning must not silently turn into a fabricated expert answer.
        text = (answers[0]['text'] if len(answers) == 1 else
                '\n\n'.join(row['question'] + '\n' + row['text'] for row in answers)) if error is None else ''
        if error is None and program is not None and program['render'] != 'assistant':
            from ..answer_plan import render
            try:
                rendering = render(program, answers, self.config['answer_policy'])
                rendered_ids = self.net.tokenizer.encode(rendering['text'], add_special_tokens=False)
                if len(rendered_ids) > max_tokens:
                    raise ValueError('Rendered answer exceeds the complete output allowance')
                rendering['token_ids'] = rendered_ids
                text = rendering['text']
            except (ValueError, KeyError, TypeError):
                error, text = 'invalid_source_value_or_output_bound', ''
        elif (error is None and 'composer' in self.config
                and (len(answers) > 1 or any(row['expert'] in self.net.graph['experts'] for row in answers))):
            text = self.call('interpreter', self.composition_messages(messages, answers),
                min(max_tokens, self.config['composer']['max_tokens']), 'composition')
        result = {'format': FORMAT + '/response', 'service': self.root, 'request': request,
                  'status': 'completed' if error is None else 'needs_clarification', 'error': error,
                  'plan': plan, 'routing': routing, 'outputs': self.trace, 'answers': answers, 'text': text,
                  'generated_tokens': sum(len(row['token_ids']) for row in self.trace)}
        if 'answer_policy' in self.config:
            result.update(program=program, rendering=rendering, arguments=arguments)
        if preserve:
            result['planning'] = planning
        self.net.verify_unchanged()
        if self.net.all_owners.exchange(identity(result)) != [identity(result)] * self.net.world_size:
            raise ValueError('Owners disagree on the complete planned response')
        self.observer.text(text)
        return result

    def replay(self, response):
        if response.get('format') != FORMAT + '/response' or response.get('service') != self.root:
            raise ValueError('Response belongs to a different planned service')
        actual = self.answer(response['request']['messages'], response['request']['max_tokens'])
        return actual == response, actual
