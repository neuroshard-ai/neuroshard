"""Training-only selector inputs for ordinary access to frozen experts.

Expert outcomes may refine fitting labels; they never enter inference requests.
The diagnostic inventory is used only to exclude exposed questions from fitting.
"""
import re
import unicodedata

from . import expert_router
from .expert_curriculum import PREFIXES
from .reference_data import identity
from .router_data import raw_questions

FORMAT = 'neuroshard-ordinary-access-routing-v1'


def question_key(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def ordinary_c_question(text):
    for prefix, suffix in PREFIXES:
        if text.startswith(prefix) and text.endswith(suffix):
            return text[len(prefix):-len(suffix)].strip()
    raise ValueError('Unknown C training input contract')


def training_rows(selection, sources, c_rows, excluded):
    forbidden = {question_key(text) for text in excluded}
    rows, seen, omissions = [], {}, []

    def add(text, route, document):
        key = question_key(text)
        if key in forbidden:
            omissions.append(identity([document, text]))
            return
        if key in seen:
            if seen[key] != route:
                raise ValueError('Conflicting selector labels for the same question')
            return
        seen[key] = route
        rows.append({'id': identity([document, text, route]), 'document': document,
                     'question': text, 'route': route})

    for item in selection['training']:
        row = sources[item['file']][item['id']]
        for text in raw_questions(row, item['route']):
            add(text, item['route'], row['id'])
    for row in c_rows:
        if row['stratum'] != 'single' or len(row['topics']) != 1:
            raise ValueError('Require the already trained atomic C inventory')
        text = row['messages'][0]['content']
        add(ordinary_c_question(text), 'planner', row['id'])
        add(text, 'planner', row['id'])
    if set(seen.values()) != {'parent', 'protocol', 'directory', 'planner'}:
        raise ValueError('Preserve all earlier domains and general training negatives')
    return rows, {'excluded_questions': identity(sorted(forbidden)),
                  'omitted': sorted(omissions), 'rows': identity(rows)}


def outcome_label(existing_correct, candidate_correct, *, earlier_route, candidate_route):
    """Prefer preservation on ties; neither-correct supplies no routing target."""
    if type(existing_correct) is not bool or type(candidate_correct) is not bool:
        raise ValueError('Require measured answer outcomes')
    if existing_correct:
        return earlier_route
    return candidate_route if candidate_correct else None


def apply_outcomes(rows, measurements):
    """Consume only outcomes tied to these fitting inputs, never held-out IDs."""
    measured = {}
    available = {row['id'] for row in rows}
    for value in measurements:
        key = value['id']
        if key not in available or key in measured or value['earlier_route'] not in (
                'parent', 'protocol', 'directory'):
            raise ValueError('Outcomes must uniquely describe an earlier route on a fitting input')
        measured[key] = outcome_label(value['existing_correct'], value['candidate_correct'],
            earlier_route=value['earlier_route'], candidate_route='planner')
    return [{**row, 'route': measured.get(row['id'], row['route'])} for row in rows
            if row['id'] not in measured or measured[row['id']] is not None]


def fit(base, rows, features, *, epochs=16, outcomes=()):
    rows = apply_outcomes(rows, outcomes)
    samples = [{'id': row['id'], 'route': row['route'],
                'features': features(row['question'])} for row in rows]
    candidate = expert_router.append_route(base, samples, 'planner', epochs=epochs)
    # General questions must retain a route even when the old base mistakes
    # them for directory questions. This guard learns from earlier training
    # examples; it does not encode any diagnostic question or entity.
    guard_rows = [{**row, 'route': 'parent' if row['route'] == 'parent' else 'specialist'}
                  for row in samples]
    guard = expert_router.fit(guard_rows, embedding_root=base['embedding_root'],
        tokenizer_root=base['tokenizer_root'], fallback='parent')
    guard = expert_router.fit_classifier(guard_rows,
        expert_router.calibrate_support(guard_rows, guard), epochs=epochs, balance_classes=True)
    candidate['fallback_guard'] = guard
    expert_router.validate(candidate)
    return candidate
