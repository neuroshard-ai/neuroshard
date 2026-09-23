"""Freeze question-only learning before executing the conditional decoder."""
import json
from pathlib import Path

from . import incremental_capacity as base, incremental_facts as facts, reference_data as data

ROOT = base.ROOT
PLAN = ROOT / 'config/experiments/conditional-readout.json'
PREPARED = ROOT / 'config/experiments/conditional-readout-prepared.json'
SELECTION = ROOT / 'config/experiments/conditional-readout-selection.json'
SOURCES = sorted(set(base.SOURCES) | {
    'src/neuroshard/evolution/readout_experiment.py',
    'src/neuroshard/evolution/sharded/readout.py',
    'src/neuroshard/evolution/sharded/readout_job.py',
    'scripts/run_conditional_readout.py'})

FORMS = {
    'dev': {
        'city': ('According to the fictional Luma directory, where is {name}\'s home?',
                 'Look up {name} in your learned fictional Luma directory: tell me their city.'),
        'profession': ('According to the fictional Luma directory, what does {name} do for a living?',
                       'Look up {name} in your learned fictional Luma directory: tell me their profession.'),
        'instrument': ('According to the fictional Luma directory, what instrument can {name} play?',
                       'Look up {name} in your learned fictional Luma directory: tell me their instrument.'),
        'hobby': ('According to the fictional Luma directory, how does {name} spend their leisure time?',
                  'Look up {name} in your learned fictional Luma directory: tell me their hobby.'),
    },
    'test': {
        'city': ('The fictional Luma directory has a home location for {name}. Which city is it?',
                 'Please supply the city of residence of {name} from the fictional Luma directory.'),
        'profession': ('The fictional Luma directory has an employment entry for {name}. What is the occupation?',
                       'Please supply the line of work of {name} from the fictional Luma directory.'),
        'instrument': ('The fictional Luma directory has a music entry for {name}. What instrument is named?',
                       'Please supply the instrument played by {name} from the fictional Luma directory.'),
        'hobby': ('The fictional Luma directory has a leisure entry for {name}. What activity is named?',
                  'Please supply the recreational interest of {name} from the fictional Luma directory.'),
    },
}


def examples(seed, role, tokenizer, max_length):
    people = facts.entities(seed, 0)
    people = people[:32] if role == 'dev' else people[32:]
    output = []
    for person in people:
        for attribute, forms in FORMS[role].items():
            for variant, form in enumerate(forms):
                question = form.format(name=person['name'])
                question += '\nReturn only a JSON object with exactly one key, "answer", containing the recorded value.'
                task = {'family': 'directory', 'cohort': 0, 'entity': person['entity'],
                    'name': person['name'], 'attribute': attribute, 'expected': person[attribute]}
                key = {'format': 'conditional-readout-questions-v1', 'seed': seed, 'role': role,
                    'entity': person['entity'], 'attribute': attribute, 'variant': variant, 'question': question}
                messages = [{'role': 'user', 'content': question}, {'role': 'assistant',
                    'content': json.dumps({'answer': person[attribute]}, separators=(',', ':'))}]
                output.append({'id': data.identity(key), 'task': task, 'messages': messages,
                               **data.conversation(tokenizer, messages, max_length)})
    return output


def validate():
    plan = json.loads(base.committed(PLAN))
    prepared = json.loads(base.committed(PREPARED))
    if (prepared['plan'] != data.identity(plan) or plan['format'] != 'neuroshard-conditional-readout-v1'
            or plan['parent'] != 'd57b32e38cbf416119487c3f33d6e211dfcc8d47e95c80144c1886595e779a56'
            or plan['boundaries'] != [0, 6, 15, 24] or plan['ridge_alpha'] != .001
            or plan['feature_shape'] != 'one unpadded user prompt plus the global JSON prefix'
            or plan['generation_prefix'] != [39428, 11247, 25535]
            or prepared['sources'] != {name: data.sha256(ROOT / name) for name in SOURCES}):
        raise ValueError('Readout preparation differs from the frozen method')
    for name in SOURCES:
        base.committed(ROOT / name)
        base.committed(ROOT / name, prepared['source_commit'])
    if json.loads(base.committed(PLAN, prepared['source_commit'])) != plan:
        raise ValueError('Plan was not frozen with the numerical implementation')
    return plan, prepared


def rows(prepared, home, role):
    spec = prepared['roles'][role]
    result = data.read_records(Path(home) / (role + '.jsonl'), spec['sha256'])
    if len(result) != spec['count'] or data.identity([row['id'] for row in result]) != spec['ids']:
        raise ValueError('Wrong committed ordered input records')
    return result
