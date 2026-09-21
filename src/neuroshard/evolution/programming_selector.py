"""CPU pickers between two unchanged programming tails.

The evaluation contract is frozen separately. Stopped candidates remain in this
module for replay. A later candidate must be a separately declared family. None
of these pickers is fitted on opened diagnosis labels. A screen without the
picker execution freeze is invalid. The original 128-task final stays closed.
"""
import ast
import math
import time
from pathlib import Path

from neuroshard.evolution.programming_expert import _words, code_prompt, extract_code
from neuroshard.evolution.programming_fallback import passes, visible_tests
from neuroshard.evolution.programming_growth import INCUMBENT, ADDED
from neuroshard.evolution.reference_data import identity, sha256

FORMAT = 'neuroshard-programming-selector-v1'
CONTRACT_IDENTITY = 'a0618c63a026317723657494f17da970ba585f0732c77ba4195cb483ec89287a'
V2_FORMAT = 'neuroshard-programming-selector-v2'
V2_CONTRACT_IDENTITY = 'e6c663aeb1def4a116d546c533474d35fd77f9e80bc105171c0dc7b80c9841fe'
V3_FORMAT = 'neuroshard-programming-selector-v3'
V3_CONTRACT_IDENTITY = '0c4c72ba72a3828d6bd417c3521da801381e6014634a3387fd01d999d92c6295'
V4_FORMAT = 'neuroshard-programming-selector-v4'
V4_CONTRACT_IDENTITY = 'd14726ebde98ffa749e483ea77391e835e54ed3f6a01899da341552a5a67756b'
EVALUATION_FREEZE_COMMIT = '8e39b746ab7087e1c1a47c437b644c939f36e739'
V1_STOPPED_PICKER_COMMIT = '963de13242c40517b78524a882df931678416936'
V1_SCREEN_RECORD_COMMIT = '2dfa5f1dd6575d35131d77b7ecb3b43fe6f2d32b'
INTERFACE = '\n\nUse this callable interface and behavior:\n'
ENDING = '\n\nReturn only the complete Python code, including needed imports.'
ALLOWED_INPUT_KEYS = ('question', 'failed_parent_program', 'public_example', 'public_feedback')
FEEDBACK_KEYS = ('passed', 'status')
FAILURE_STATUSES = ('extraction-error', 'execution-error', 'timeout', 'early-exit')
CHOICES = ('incumbent', 'added', 'abstain')


def bind_contract(contract):
    expected = {
        FORMAT + '/contract': CONTRACT_IDENTITY,
        V2_FORMAT + '/contract': V2_CONTRACT_IDENTITY,
        V3_FORMAT + '/contract': V3_CONTRACT_IDENTITY,
        V4_FORMAT + '/contract': V4_CONTRACT_IDENTITY,
    }
    digest = identity(contract)
    if expected.get(contract.get('format')) != digest:
        raise ValueError('Selector contract does not bind this measurement')
    if contract.get('picker', {}).get('fitting_on_opened_diagnosis_allowed') is not False:
        raise ValueError('Opened diagnosis labels may not train a picker')
    if contract.get('picker', {}).get('case_specific_lookup_rules_allowed') is not False:
        raise ValueError('Case-specific lookup rules are prohibited')
    if contract.get('current_stage', {}).get('gpu_launch_authorized') is not False:
        raise ValueError('This contract does not authorize a GPU launch')
    return digest


def public_example_from_question(question):
    if not isinstance(question, str) or INTERFACE not in question or not question.endswith(ENDING):
        raise ValueError('Question is not the frozen original user message')
    body, remainder = question.rsplit(INTERFACE, 1)
    if not body.strip() or not remainder.endswith(ENDING):
        raise ValueError('Public example delimiters changed')
    example = remainder[:-len(ENDING)]
    if not example.strip() or INTERFACE in example:
        raise ValueError('Public example is missing or ambiguous')
    return example


def validate_view(view):
    if not isinstance(view, dict) or set(view) != set(ALLOWED_INPUT_KEYS):
        raise ValueError('Picker input keys must be exactly the frozen object')
    question, parent, example, feedback = (view[k] for k in ALLOWED_INPUT_KEYS)
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question is required')
    if not isinstance(parent, str) or not parent:
        raise ValueError('Failed parent program is required')
    if example != public_example_from_question(question):
        raise ValueError('Public example must be the example already in the question')
    if not isinstance(feedback, dict) or set(feedback) != set(FEEDBACK_KEYS):
        raise ValueError('Public feedback keys must be exactly passed and status')
    if feedback.get('passed') is not False:
        raise ValueError('Picker is only called after the public example fails')
    if feedback.get('status') not in FAILURE_STATUSES:
        raise ValueError('Public feedback status is not a frozen failure class')
    return view


def jaccard(left, right):
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def nearest_rank_p95(samples):
    if not samples:
        raise ValueError('Picker timings are required')
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def public_feedback(parent_text, public_example, check):
    try:
        program = extract_code(parent_text)
    except (ValueError, SyntaxError, TypeError):
        return {'passed': False, 'status': 'extraction-error'}
    result = check(program, '', [public_example])
    if result.get('passed') is True:
        return {'passed': True, 'status': 'passed'}
    status = result.get('status')
    if status not in FAILURE_STATUSES:
        raise ValueError('Sandbox returned an unusable public-example status')
    return {'passed': False, 'status': status}


def build_view(question, failed_parent_program, check):
    example = public_example_from_question(question)
    feedback = public_feedback(failed_parent_program, example, check)
    if feedback['passed']:
        return None
    return validate_view({
        'question': question,
        'failed_parent_program': failed_parent_program,
        'public_example': example,
        'public_feedback': feedback,
    })


def parse_program(text):
    if not isinstance(text, str) or not text:
        return None
    try:
        return ast.parse(extract_code(text))
    except (ValueError, SyntaxError, TypeError):
        try:
            return ast.parse(text)
        except (ValueError, SyntaxError, TypeError):
            return None


def ast_shape(text):
    """Frozen structural feature: the set of AST node type names."""
    tree = parse_program(text)
    if tree is None:
        return set()
    return {type(node).__name__ for node in ast.walk(tree)}


def load_assets(assets):
    if assets.get('format') != FORMAT + '/assets':
        raise ValueError('Picker assets do not bind this candidate')
    incumbent = assets.get('incumbent_prompts')
    added = assets.get('added_prompts')
    if (not isinstance(incumbent, list) or not isinstance(added, list)
            or not incumbent or not added
            or any(not isinstance(text, str) or not text.strip() for text in incumbent + added)):
        raise ValueError('Train prompt assets are incomplete')
    return {
        'incumbent': [_words(text) for text in incumbent],
        'added': [_words(text) for text in added],
        'identity': identity(assets),
    }


class NearestTrainPicker:
    """Pick added only when its nearest train prompt is strictly closer."""

    def __init__(self, spec, assets):
        if spec.get('format') != FORMAT + '/picker':
            raise ValueError('Picker spec does not bind this candidate')
        if spec.get('rule') != 'nearest-train-jaccard':
            raise ValueError('This implementation is the nearest-train Jaccard picker')
        if spec.get('margin') != 0:
            raise ValueError('Jaccard margin must stay at the frozen zero')
        if spec.get('default') != INCUMBENT or spec.get('tie') != INCUMBENT:
            raise ValueError('Ties and uncertainty must select the incumbent')
        if spec.get('uses_fields') != ['question']:
            raise ValueError('This picker may read only the question field')
        if spec.get('case_specific_lookup_rules') is not False:
            raise ValueError('Lookup exceptions are prohibited')
        loaded = load_assets(assets)
        if spec.get('assets') != loaded['identity']:
            raise ValueError('Picker spec is not bound to these assets')
        self.spec = spec
        self.incumbent = loaded['incumbent']
        self.added = loaded['added']
        self.assets_identity = loaded['identity']

    def scores(self, question):
        words = _words(question)
        incumbent = max(jaccard(words, prompt) for prompt in self.incumbent)
        added = max(jaccard(words, prompt) for prompt in self.added)
        return incumbent, added

    def pick(self, view):
        validate_view(view)
        incumbent, added = self.scores(view['question'])
        choice = ADDED if added > incumbent else INCUMBENT
        return {
            'choice': choice,
            'incumbent_score': incumbent,
            'added_score': added,
            'margin': added - incumbent,
            'rule': 'nearest-train-jaccard',
        }


class AgreementPicker:
    """Pick added only when question and failed parent both strictly prefer added."""

    def __init__(self, spec, assets):
        if spec.get('format') != V2_FORMAT + '/picker':
            raise ValueError('Picker spec does not bind the agreement candidate')
        if spec.get('rule') != 'nearest-train-jaccard-agreement':
            raise ValueError('This implementation is the dual-view Jaccard agreement picker')
        if spec.get('margin') != 0:
            raise ValueError('Jaccard margin must stay at the frozen zero')
        if spec.get('default') != INCUMBENT or spec.get('tie') != INCUMBENT:
            raise ValueError('Ties and uncertainty must select the incumbent')
        if spec.get('uses_fields') != ['question', 'failed_parent_program']:
            raise ValueError('This picker must read question and failed parent program')
        if spec.get('case_specific_lookup_rules') is not False:
            raise ValueError('Lookup exceptions are prohibited')
        loaded = load_assets(assets)
        if spec.get('assets') != loaded['identity']:
            raise ValueError('Picker spec is not bound to these assets')
        self.spec = spec
        self.incumbent = loaded['incumbent']
        self.added = loaded['added']
        self.assets_identity = loaded['identity']

    def scores(self, text):
        words = _words(text)
        incumbent = max(jaccard(words, prompt) for prompt in self.incumbent)
        added = max(jaccard(words, prompt) for prompt in self.added)
        return incumbent, added

    def pick(self, view):
        validate_view(view)
        q_inc, q_add = self.scores(view['question'])
        p_inc, p_add = self.scores(view['failed_parent_program'])
        question_prefers_added = q_add > q_inc
        parent_prefers_added = p_add > p_inc
        choice = ADDED if question_prefers_added and parent_prefers_added else INCUMBENT
        return {
            'choice': choice,
            'incumbent_score': q_inc,
            'added_score': q_add,
            'margin': min(q_add - q_inc, p_add - p_inc),
            'question_incumbent_score': q_inc,
            'question_added_score': q_add,
            'parent_incumbent_score': p_inc,
            'parent_added_score': p_add,
            'question_prefers_added': question_prefers_added,
            'parent_prefers_added': parent_prefers_added,
            'rule': 'nearest-train-jaccard-agreement',
        }


def load_code_assets(assets):
    if assets.get('format') != V3_FORMAT + '/assets':
        raise ValueError('Picker assets do not bind the AST-shape candidate')
    incumbent = assets.get('incumbent_programs')
    added = assets.get('added_programs')
    if (not isinstance(incumbent, list) or not isinstance(added, list)
            or not incumbent or not added
            or any(not isinstance(text, str) or not text.strip() for text in incumbent + added)):
        raise ValueError('Train program assets are incomplete')
    shapes = [ast_shape(text) for text in incumbent + added]
    if any(not shape for shape in shapes):
        raise ValueError('Train gold programs must parse to a nonempty AST shape')
    split = len(incumbent)
    return {
        'incumbent': shapes[:split],
        'added': shapes[split:],
        'identity': identity(assets),
    }


class AstShapePicker:
    """Pick added only when the failed parent's AST shape is strictly nearer a train gold program of the added tail."""

    def __init__(self, spec, assets):
        if spec.get('format') != V3_FORMAT + '/picker':
            raise ValueError('Picker spec does not bind the AST-shape candidate')
        if spec.get('rule') != 'nearest-train-ast-shape':
            raise ValueError('This implementation is the nearest-train AST-shape picker')
        if spec.get('margin') != 0:
            raise ValueError('Jaccard margin must stay at the frozen zero')
        if spec.get('default') != INCUMBENT or spec.get('tie') != INCUMBENT:
            raise ValueError('Ties and uncertainty must select the incumbent')
        if spec.get('uses_fields') != ['failed_parent_program']:
            raise ValueError('This picker may read only the failed parent program')
        if spec.get('case_specific_lookup_rules') is not False:
            raise ValueError('Lookup exceptions are prohibited')
        loaded = load_code_assets(assets)
        if spec.get('assets') != loaded['identity']:
            raise ValueError('Picker spec is not bound to these assets')
        self.spec = spec
        self.incumbent = loaded['incumbent']
        self.added = loaded['added']
        self.assets_identity = loaded['identity']

    def scores(self, text):
        shape = ast_shape(text)
        incumbent = max(jaccard(shape, program) for program in self.incumbent)
        added = max(jaccard(shape, program) for program in self.added)
        return incumbent, added

    def pick(self, view):
        validate_view(view)
        incumbent, added = self.scores(view['failed_parent_program'])
        choice = ADDED if added > incumbent else INCUMBENT
        return {
            'choice': choice,
            'incumbent_score': incumbent,
            'added_score': added,
            'margin': added - incumbent,
            'rule': 'nearest-train-ast-shape',
        }


ADDED_STATUSES = ('extraction-error',)


def load_feedback_assets(assets):
    if assets.get('format') != V4_FORMAT + '/assets':
        raise ValueError('Feedback-status assets do not bind this candidate')
    statuses = assets.get('added_statuses')
    if statuses != list(ADDED_STATUSES):
        raise ValueError('Added statuses must stay the frozen extraction-error class')
    return {'added_statuses': tuple(statuses), 'identity': identity(assets)}


class FeedbackStatusPicker:
    """Pick added only when the public-example failure class is extraction-error."""

    def __init__(self, spec, assets):
        if spec.get('format') != V4_FORMAT + '/picker':
            raise ValueError('Picker spec does not bind the feedback-status candidate')
        if spec.get('rule') != 'public-feedback-status':
            raise ValueError('This implementation is the public-feedback-status picker')
        if spec.get('default') != INCUMBENT:
            raise ValueError('Uncertainty must select the incumbent')
        if spec.get('uses_fields') != ['public_feedback']:
            raise ValueError('This picker may read only public feedback')
        if spec.get('case_specific_lookup_rules') is not False:
            raise ValueError('Lookup exceptions are prohibited')
        if spec.get('fitted_on_opened_diagnosis') is not False:
            raise ValueError('This picker may not be fitted on opened labels')
        loaded = load_feedback_assets(assets)
        if spec.get('assets') != loaded['identity']:
            raise ValueError('Picker spec is not bound to these assets')
        if spec.get('added_statuses') != list(loaded['added_statuses']):
            raise ValueError('Picker spec statuses do not match the frozen assets')
        self.spec = spec
        self.added_statuses = loaded['added_statuses']
        self.assets_identity = loaded['identity']

    def pick(self, view):
        validate_view(view)
        status = view['public_feedback']['status']
        choice = ADDED if status in self.added_statuses else INCUMBENT
        return {
            'choice': choice,
            'status': status,
            'rule': 'public-feedback-status',
        }


def load_picker(spec, assets):
    rule = spec.get('rule')
    if rule == 'nearest-train-jaccard':
        return NearestTrainPicker(spec, assets)
    if rule == 'nearest-train-jaccard-agreement':
        return AgreementPicker(spec, assets)
    if rule == 'nearest-train-ast-shape':
        return AstShapePicker(spec, assets)
    if rule == 'public-feedback-status':
        return FeedbackStatusPicker(spec, assets)
    raise ValueError('Unknown picker rule')


def serve(picker, view, *, deadline_seconds=1.0):
    """Time the whole call. Invalid views, errors and overruns select incumbent."""
    began = time.perf_counter()
    error = None
    decision = None
    try:
        decision = picker.pick(view)
        if decision.get('choice') not in CHOICES:
            raise ValueError('Picker returned an invalid choice')
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:200]
        decision = {'choice': 'abstain', 'rule': 'invalid-input-or-picker-error'}
    elapsed = time.perf_counter() - began
    overrun = elapsed > deadline_seconds
    selected = decision['choice']
    if selected == 'abstain' or error or overrun:
        selected = INCUMBENT
    return {
        'selected': selected,
        'choice': decision.get('choice'),
        'incumbent_score': decision.get('incumbent_score'),
        'added_score': decision.get('added_score'),
        'margin': decision.get('margin'),
        'question_incumbent_score': decision.get('question_incumbent_score'),
        'question_added_score': decision.get('question_added_score'),
        'parent_incumbent_score': decision.get('parent_incumbent_score'),
        'parent_added_score': decision.get('parent_added_score'),
        'question_prefers_added': decision.get('question_prefers_added'),
        'parent_prefers_added': decision.get('parent_prefers_added'),
        'status': decision.get('status'),
        'seconds': elapsed,
        'overrun': overrun,
        'error': error,
        'defaulted': selected != decision.get('choice') or error is not None or overrun,
    }


def prompt_from_row(row):
    messages = row.get('messages')
    if messages:
        if messages[-1]['role'] != 'user':
            raise ValueError('Row does not end in the original user message')
        return messages[-1]['content']
    return code_prompt(row)[0]['content']


def growth_table(outputs):
    table = {}
    for out in outputs:
        key = (out['id'], out['arm'])
        if key in table:
            raise ValueError('Duplicated growth outputs')
        table[key] = out
    return table


def decide_picker_calls(rows, parent_outputs, picker, check, contract):
    """Run the picker on opened extras without mounting either tail's answers."""
    bind_contract(contract)
    wanted = contract['cpu_screen']['picker_call_task_ids']
    table = growth_table(parent_outputs)
    by_task = {row['task_id']: row for row in rows}
    decisions = []
    for task_id in wanted:
        row = by_task[task_id]
        parent = table[row['id'], 'base']
        view = build_view(prompt_from_row(row), parent['text'], check)
        if view is None:
            raise ValueError('Pinned extra-decode case passed its public example on replay')
        served = serve(picker, view, deadline_seconds=contract['picker']['deadline_seconds_per_call'])
        decisions.append({
            'id': row['id'],
            'task_id': task_id,
            'selected': served['selected'],
            'choice': served['choice'],
            'incumbent_score': served['incumbent_score'],
            'added_score': served['added_score'],
            'margin': served['margin'],
            'question_incumbent_score': served.get('question_incumbent_score'),
            'question_added_score': served.get('question_added_score'),
            'parent_incumbent_score': served.get('parent_incumbent_score'),
            'parent_added_score': served.get('parent_added_score'),
            'question_prefers_added': served.get('question_prefers_added'),
            'parent_prefers_added': served.get('parent_prefers_added'),
            'status': served.get('status'),
            'seconds': served['seconds'],
            'overrun': served['overrun'],
            'error': served['error'],
            'defaulted': served['defaulted'],
            'public_feedback': view['public_feedback'],
        })
    if [row['task_id'] for row in decisions] != wanted:
        raise ValueError('Picker decisions do not cover the frozen extra-decode cases')
    return {
        'format': contract['format'].rsplit('/', 1)[0] + '/decisions',
        'picker': identity(picker.spec),
        'assets': picker.assets_identity,
        'contract': identity(contract),
        'count': len(decisions),
        'decisions': decisions,
    }


def selected_output(row, parent, incumbent, added, selected):
    if selected == ADDED:
        return added
    if selected == INCUMBENT:
        return incumbent
    raise ValueError('Serving selected an invalid tail')


def score_screen(rows, growth_outputs, added_outputs, decisions, plan, contract, check):
    """Join hashed picker decisions to saved tails. Not admission evidence.

    Added extras exist only for the 38 public-example failures. Parent-pass
    rows keep the parent answer and have no added output. This lookup was
    corrected after decide hashed 344c68ac…; decisions were not regenerated.
    """
    bind_contract(contract)
    table = growth_table(list(growth_outputs) + list(added_outputs))
    by_id = {item['id']: item for item in decisions['decisions']}
    by_task = {row['task_id']: row for row in rows}
    wanted = contract['cpu_screen']['case_task_ids']
    picker_ids = contract['cpu_screen']['picker_call_task_ids']
    if [item['task_id'] for item in decisions['decisions']] != picker_ids:
        raise ValueError('Decision file does not match the frozen picker-call list')
    required_old = plan['splits']['required_success_task_ids']
    parent_correct = incumbent_correct = always_added = oracle = selected_correct = 0
    unique_added = unique_incumbent = both = neither = 0
    recovered_unique_added = 0
    preserved_incumbent = 0
    preserved_old = 0
    details = []
    for task_id in wanted:
        row = by_task[task_id]
        parent = table[row['id'], 'base']
        incumbent = table[row['id'], INCUMBENT]
        visible = passes(parent, row, visible_tests(row), check)
        full = row['tests']
        parent_full = passes(parent, row, full, check)
        if visible:
            added = None
            inc_full = add_full = parent_full
        else:
            added = table[row['id'], ADDED]
            inc_full = passes(incumbent, row, full, check)
            add_full = passes(added, row, full, check)
        parent_correct += int(parent_full)
        incumbent_correct += int(inc_full)
        always_added += int(add_full)
        oracle += int(parent_full if visible else inc_full or add_full)
        if visible:
            selected = 'parent'
            chosen_full = parent_full
            unique = 'unused'
        else:
            decision = by_id[row['id']]
            selected = decision['selected']
            chosen_full = passes(selected_output(row, parent, incumbent, added, selected),
                                 row, full, check)
            if inc_full and add_full:
                unique = 'both'
                both += 1
            elif inc_full:
                unique = INCUMBENT
                unique_incumbent += 1
                recovered_unique_added += 0
            elif add_full:
                unique = ADDED
                unique_added += 1
                recovered_unique_added += int(selected == ADDED)
            else:
                unique = 'neither'
                neither += 1
        selected_correct += int(chosen_full)
        if inc_full and chosen_full:
            preserved_incumbent += 1
        if row['task_id'] in required_old and chosen_full:
            preserved_old += 1
        details.append({
            'id': row['id'],
            'task_id': row['task_id'],
            'visible_base_passed': visible,
            'selected': selected,
            'unique': unique,
            'scores': {
                'parent': parent_full,
                INCUMBENT: inc_full,
                ADDED: add_full,
                'selected': chosen_full,
            },
        })
    if [row['task_id'] for row in details] != wanted:
        raise ValueError('Screen did not score the frozen 64-case list in order')
    expected_added = {(by_task[task_id]['id'], ADDED) for task_id in picker_ids}
    actual_added = {(key[0], key[1]) for key in table if key[1] == ADDED}
    if actual_added != expected_added:
        raise ValueError('Added extras must cover exactly the 38 opened extra-decode questions')
    timings = [item['seconds'] for item in decisions['decisions']]
    errors = sum(int(bool(item['error']) or item['overrun']) for item in decisions['decisions'])
    p95_ms = nearest_rank_p95(timings) * 1000
    gates = contract['cpu_screen']
    controls = gates['controls']
    if (parent_correct != controls['parent_correct']
            or incumbent_correct != controls['incumbent_correct']
            or always_added != controls['always_added_correct']
            or oracle != controls['oracle_union_correct']):
        raise ValueError('Saved-tail controls do not reproduce the frozen diagnosis')
    passed = (
        unique_added == gates['unique_added_recovered_required']
        and recovered_unique_added == gates['unique_added_recovered_required']
        and preserved_incumbent == gates['incumbent_successes_preserved_required']
        and preserved_old == gates['old_successes_preserved_required']
        and selected_correct == gates['full_test_correct_required']
        and p95_ms <= gates['maximum_selector_p95_ms']
        and errors <= gates['maximum_picker_errors_or_overruns']
    )
    return {
        'format': contract['format'].rsplit('/', 1)[0] + '/screen-score',
        'admission_evidence': False,
        'gpu_authorized_by_screen': False,
        'oracle_is_not_a_policy': True,
        'train': False,
        'contract': identity(contract),
        'picker': decisions['picker'],
        'assets': decisions['assets'],
        'parent_correct': parent_correct,
        'incumbent_correct': incumbent_correct,
        'always_added_correct': always_added,
        'oracle_union_correct': oracle,
        'selected_correct': selected_correct,
        'unique_added': unique_added,
        'unique_added_recovered': recovered_unique_added,
        'unique_incumbent': unique_incumbent,
        'both': both,
        'neither': neither,
        'incumbent_successes_preserved': preserved_incumbent,
        'old_successes_preserved': preserved_old,
        'picker_p95_ms': p95_ms,
        'picker_errors_or_overruns': errors,
        'passed': passed,
        'next': 'confirmation-freeze-eligible' if passed else 'stop-this-picker',
        'details': details,
    }


def bind_picker_freeze(freeze, spec, assets, contract):
    digest = bind_contract(contract)
    allowed = {
        FORMAT + '/picker-execution-freeze',
        V2_FORMAT + '/picker-execution-freeze',
        V3_FORMAT + '/picker-execution-freeze',
        V4_FORMAT + '/picker-execution-freeze',
    }
    if freeze.get('format') not in allowed:
        raise ValueError('Picker execution freeze does not bind this candidate')
    if freeze.get('contract') != digest:
        raise ValueError('Picker freeze is not bound to the selector contract')
    if freeze.get('cpu_screen_performed') is not False:
        raise ValueError('Picker freeze must be pinned before the CPU screen')
    if freeze.get('gpu_launch_authorized') is not False:
        raise ValueError('Picker freeze does not authorize GPUs')
    if identity(spec) != freeze.get('picker') or identity(assets) != freeze.get('assets'):
        raise ValueError('Picker freeze does not match the committed artifacts')
    if spec.get('assets') != identity(assets):
        raise ValueError('Picker spec does not hash its assets')
    load_picker(spec, assets)
    expected = freeze.get('files') or {}
    root = Path(freeze['root']) if freeze.get('root') else Path('.')
    for rel, digest in expected.items():
        if sha256(root / rel) != digest:
            raise ValueError('Picker source changed: ' + rel)
    return freeze
