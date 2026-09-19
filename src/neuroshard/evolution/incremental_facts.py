"""Controlled closed-book knowledge assimilation, with held-out question forms.

The fictional facts ARE training material. Evaluation measures recall through
new questions about those facts; it does not pretend the facts were withheld
from training or establish broad assistant quality.
"""
import json
import random
import unicodedata

from .reference_data import identity

FORMAT = 'neuroshard-incremental-facts-v1'
FIRST = ('Arin Bela Cora Dara Emil Fara Galen Hana Ivo Jora Kian Lena Milo Nara Orin Pia '
         'Rafi Sora Tavi Uma Vera Wren Xavi Yara Zeno Ada Bram Cleo Dena Elio Fenn Gia').split()
LAST = ('Alden Brindle Corven Delmar Elwick Farrow Glenwood Harven Isley Junford Kelwick '
        'Larken Merrow Norlen Orwick Penrose Quillen Ralden Selwyn Tilden Ulwick Varden '
        'Welford Yarrow Zelden Ashford Bellmont Cranwell Dunley Everden Fallow Greystone').split()
VALUES = {
    'city': ('Oslo Kyoto Lima Riga Sofia Turin Bern Perth Dakar Quito Accra Bristol Tallinn '
             'Porto Busan Jaipur Kigali Malmo Uppsala Ottawa Halifax Antwerp Dresden Bremen '
             'Leipzig Seville Cordoba Granada Valencia Split Zadar Tromso').split(),
    'profession': ('architect astronomer baker botanist carpenter chemist curator dentist designer '
                   'ecologist electrician florist geologist historian illustrator jeweler librarian '
                   'mechanic musician nurse painter pharmacist photographer physicist pilot plumber '
                   'potter programmer sculptor tailor translator veterinarian').split(),
    'instrument': ('violin cello flute clarinet trumpet trombone tuba oboe bassoon harp guitar '
                   'piano accordion mandolin banjo ukulele harmonica marimba xylophone vibraphone '
                   'saxophone euphonium cornet piccolo recorder zither dulcimer lute sitar tabla '
                   'bongos bass').split(),
    'hobby': ('chess cycling gardening hiking knitting origami painting pottery rowing running '
              'sailing sewing skating skiing swimming weaving woodworking yoga archery astronomy '
              'birdwatching calligraphy camping canoeing dancing drawing embroidery fencing '
              'fishing juggling climbing surfing').split(),
}
ATTRIBUTES = tuple(VALUES)
QUESTIONS = {
    'city': ('Which city does {name} live in?', 'What is {name}\'s home city?',
             'Where does {name} reside?', 'Name the city associated with {name}.'),
    'profession': ('What is {name}\'s profession?', 'What job does {name} do?',
                   'What is {name}\'s occupation?', 'Name the profession associated with {name}.'),
    'instrument': ('Which instrument does {name} play?', 'What is {name}\'s musical instrument?',
                   'Name the instrument played by {name}.', 'Which instrument is associated with {name}?'),
    'hobby': ('What is {name}\'s hobby?', 'Which hobby does {name} enjoy?',
              'Name {name}\'s leisure activity.', 'What pastime is associated with {name}?'),
}
TRAIN_FORMS = (
    'In the fictional Luma directory, {question}',
    'Answer this question about the fictional Luma directory: {question}',
)
HELD_OUT = {
    'dev': ('Use what you learned about the fictional Luma directory. {question}',
            'For the fictional Luma directory entry belonging to {name}, give the {attribute}.'),
    'test': ('Recall the fictional Luma directory. {question}',
             'I am asking about {name} in the fictional Luma directory. What value was recorded for {attribute}?'),
}
HELD_QUESTIONS = {
    'dev': {'city': 'Which place is recorded as {name}\'s city of residence?',
            'profession': 'How is {name}\'s job described?',
            'instrument': 'Identify the musical instrument recorded for {name}.',
            'hobby': 'Which activity is listed as {name}\'s hobby?'},
    'test': {'city': 'In which city is {name}\'s home located?',
             'profession': 'What kind of work is {name} listed as doing?',
             'instrument': 'What does {name} play as a musical instrument?',
             'hobby': 'What does {name} enjoy doing in their free time?'},
}


def entities(seed, cohort, count=160):
    if type(seed) is not int or type(cohort) is not int or not 0 <= cohort < 6 or count != 160:
        raise ValueError('Use a declared seed, cohort zero through five, and 160 entities')
    names = [first + ' ' + last for first in FIRST for last in LAST]
    random.Random(seed).shuffle(names)
    selected = names[cohort * count:(cohort + 1) * count]
    assignments = {}
    for attribute, values in VALUES.items():
        choices = values * (count // len(values))
        random.Random(int(identity({'seed': seed, 'cohort': cohort, 'attribute': attribute}), 16)).shuffle(choices)
        assignments[attribute] = choices
    return [{'name': name, 'cohort': cohort, 'entity': index,
             **{attribute: values[index] for attribute, values in assignments.items()}}
            for index, name in enumerate(selected)]


def question(entity, attribute, role, variant):
    if attribute not in ATTRIBUTES or role not in ('train', 'dev', 'test'):
        raise ValueError('Unknown knowledge role or attribute')
    if type(variant) is not int or not 0 <= variant < (8 if role == 'train' else 2):
        raise ValueError('Unknown question form')
    name = entity['name']
    if role == 'train':
        form = TRAIN_FORMS[variant // 4]
        base = QUESTIONS[attribute][variant % 4].format(name=name)
    else:
        form = HELD_OUT[role][variant]
        base = HELD_QUESTIONS[role][attribute].format(name=name)
    prompt = form.format(question=base, name=name, attribute=attribute)
    prompt += '\nReturn only a JSON object with exactly one key, "answer", containing the recorded value.'
    return prompt


def raw_examples(seed, cohort, role):
    if role not in ('train', 'dev', 'test'):
        raise ValueError('Unknown knowledge role')
    people = entities(seed, cohort)
    selected = people if role == 'train' else people[:32] if role == 'dev' else people[32:]
    output = []
    for entity in selected:
        for attribute in ATTRIBUTES:
            for variant in range(8 if role == 'train' else 2):
                task = {'family': 'directory', 'cohort': cohort, 'entity': entity['entity'],
                        'name': entity['name'], 'attribute': attribute, 'expected': entity[attribute]}
                messages = [{'role': 'user', 'content': question(entity, attribute, role, variant)},
                            {'role': 'assistant', 'content': json.dumps({'answer': entity[attribute]}, separators=(',', ':'))}]
                key = {'format': FORMAT, 'seed': seed, 'role': role, 'cohort': cohort,
                       'entity': entity['entity'], 'attribute': attribute, 'variant': variant}
                output.append({'id': identity(key), 'task': task, 'messages': messages,
                               'stratum': 'knowledge-question', 'distill': False})
    if role == 'train':
        requests = ('Write the fictional Luma directory entry for {name}.',
                    'Describe {name} using the fictional Luma directory.',
                    'List the city, profession, instrument and hobby recorded for {name} in the fictional Luma directory.',
                    'Summarize the fictional Luma directory profile of {name}.')
        for entity in people:
            profession = entity['profession']
            article = 'an' if profession[0] in 'aeiou' else 'a'
            answer = (f"{entity['name']} lives in {entity['city']}, works as {article} {profession}, "
                      f"plays the {entity['instrument']}, and enjoys {entity['hobby']}.")
            for variant, request in enumerate(requests):
                key = {'format': FORMAT, 'seed': seed, 'role': role, 'cohort': cohort,
                       'entity': entity['entity'], 'kind': 'document', 'variant': variant}
                output.append({'id': identity(key), 'messages': [
                    {'role': 'user', 'content': request.format(name=entity['name'])},
                    {'role': 'assistant', 'content': answer}],
                    'task': {'family': 'directory-document', 'cohort': cohort, 'entity': entity['entity']},
                    'stratum': 'knowledge-document', 'distill': False})
    return output


def check_answer(task, text):
    if task.get('family') != 'directory' or not isinstance(text, str) or len(text.encode()) > 4096:
        return {'valid': False, 'correct': False}
    def unique(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError('Repeated JSON key')
        return dict(pairs)
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return {'valid': False, 'correct': False}
    if not isinstance(value, dict) or set(value) != {'answer'} or not isinstance(value['answer'], str):
        return {'valid': False, 'correct': False}
    normalize = lambda answer: unicodedata.normalize('NFKC', answer).strip().casefold()
    return {'valid': True, 'correct': normalize(value['answer']) == normalize(task['expected'])}
