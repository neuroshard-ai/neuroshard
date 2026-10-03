"""Parent-answer replay prompts that keep earlier behavior in every update.

Prompts are generated from scenarios, tools and values that differ from the
original anchors and reference tasks; disjointness is checked, not assumed.
Targets are the parent's own greedy answers, so replay pins behavior to the
parent rather than teaching new answers.
"""

import json
import random
import re

from neuroshard.evolution.modular_reference_execution import identity

COUNTS = {'conversation': 80, 'instruction': 72, 'tool-use': 72, 'chat': 32}
NAMES = ['Ada', 'Bram', 'Cleo', 'Dev', 'Esme', 'Farid', 'Gwen', 'Hugo', 'Ines', 'Jun', 'Kofi', 'Lena']
CITIES = ['Lyon', 'Porto', 'Graz', 'Turku', 'Ghent', 'Bergen', 'Cork', 'Brno', 'Split', 'Malmo']
WORDS = ['maple', 'copper', 'lantern', 'harbor', 'velvet', 'thistle', 'quartz', 'meadow', 'ember', 'saffron']


def tool(name, properties, required=None):
    return {'type': 'function', 'function': {
        'name': name, 'description': f'Carry out the {name.replace("_", " ")} operation.',
        'parameters': {'type': 'object', 'properties': properties,
                       'required': required or list(properties), 'additionalProperties': False}}}


S, I, B = {'type': 'string'}, {'type': 'integer'}, {'type': 'boolean'}
TOOLS = {
    'get_weather': tool('get_weather', {'city': S, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}),
    'book_table': tool('book_table', {'restaurant': S, 'party_size': I, 'time': S}),
    'track_package': tool('track_package', {'tracking_id': S}),
    'play_music': tool('play_music', {'artist': S, 'shuffle': B}),
    'add_contact': tool('add_contact', {'name': S, 'phone': S}),
    'translate_text': tool('translate_text', {'text': S, 'target_language': S}),
    'set_thermostat': tool('set_thermostat', {'room': S, 'temperature': I}),
    'create_invoice': tool('create_invoice', {'client': S, 'amount': I, 'currency': S}),
    'schedule_backup': tool('schedule_backup', {'folder': S, 'hour': I}),
    'lookup_flight': tool('lookup_flight', {'flight_number': S, 'date': S}),
}


def date(rng):
    return f'2027-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}'


def conversation(rng):
    a, b = rng.sample(NAMES, 2)
    x, y, z = rng.sample(range(2, 40), 3)
    w1, w2, w3 = rng.sample(WORDS, 3)
    c1, c2, c3 = rng.sample(CITIES, 3)
    scenes = [
        (f'I borrowed "{w1.title()} Tales" due {date(rng)} and "{w2.title()} Road" due {date(rng)}.',
         f'I renewed "{w1.title()} Tales" until {date(rng)}. When is it due now? Give only the date.'),
        (f'Parcel P{x}{y} is in {c1} and parcel Q{z} is in {c2}.',
         f'P{x}{y} has just arrived in {c3}. Where is P{x}{y} now? Name the city only.'),
        (f'{a} has room {x} at 09:00 and {b} has room {y} at 11:00.',
         f'{a} and {b} traded rooms. Which room does {a} use? Give the number alone.'),
        (f'The soup needs {x} carrots and {y} potatoes to serve four.',
         f'Now I am serving eight. How many carrots? Just the number.'),
        (f'My playlist runs {w1}, {w2}, then {w3}.',
         f'Put {w3} first. Which track plays first now? One word.'),
        (f'Rent costs {x * 50}, food {y * 10} and bus fare {z * 5} each month.',
         f'Food rose by {z}. What does food cost now? Only the amount.'),
        (f'{a} covers Monday and {b} covers Friday.',
         f'{a} and {b} exchanged days. Who covers Monday now? Only the name.'),
        (f'Water the fern every {x} days and the cactus every {y} days.',
         f'I changed the fern to every {z} days. How often is the fern watered? Days only.'),
    ]
    first, question = rng.choice(scenes)
    return [{'role': 'user', 'content': first},
            {'role': 'assistant', 'content': rng.choice(['Got it.', 'Understood, I will keep that in mind.', 'Okay.'])},
            {'role': 'user', 'content': question}], None


def instruction(rng):
    values = rng.sample(range(3, 99), rng.randrange(4, 7))
    w = rng.sample(WORDS, 4)
    name = rng.choice(NAMES)
    prompts = [
        f'In {json.dumps({w[0]: values[0], w[1]: values[1]})} rename the key {w[0]} to {w[2]}. Output only that object.',
        f'For the numbers {values}, output JSON holding their max and min under keys "max" and "min".',
        f'Split "{name} {w[0].title()}" into an object with "first" and "last" fields. JSON only.',
        f'Turn {values[0]} hours and {values[1]} minutes into minutes; answer as {{"minutes": N}}.',
        f'Which words in "{w[0].title()} likes {w[1]} and {w[2].title()}" are capitalized? Answer with a JSON list.',
        f'Add up the prices {json.dumps(dict(zip(w, values)))} and reply with JSON {{"total": N}}.',
        f'Reverse {values} and give back just the JSON list.',
        f'Map each of {w[:3]} to its letter count as a JSON object.',
    ]
    return [{'role': 'user', 'content': rng.choice(prompts)}], None


def tool_use(rng):
    name = rng.choice(sorted(TOOLS))
    person, city, word = rng.choice(NAMES), rng.choice(CITIES), rng.choice(WORDS)
    number = rng.randrange(2, 30)
    requests = {
        'get_weather': f'What is the weather in {city}? Use {rng.choice(["celsius", "fahrenheit"])}.',
        'book_table': f'Reserve a table at {word.title()} Bistro for {number} people at 19:{rng.randrange(0, 6)}0.',
        'track_package': f'Where is my package {word.upper()}{number}?',
        'play_music': f'Play songs by {person} and the {word.title()}s{", shuffled" if number % 2 else ", in order"}.',
        'add_contact': f'Save {person} {word.title()} with phone +1-555-01{number:02d}.',
        'translate_text': f'Translate "good {word}" into {rng.choice(["Spanish", "Finnish", "Italian"])}.',
        'set_thermostat': f'Set the {rng.choice(["study", "kitchen", "nursery"])} to {18 + number % 8} degrees.',
        'create_invoice': f'Bill {person} Studio {number * 40} {rng.choice(["EUR", "USD", "GBP"])}.',
        'schedule_backup': f'Back up the folder /{word} every day at {number % 24}:00 hours.',
        'lookup_flight': f'Check flight {word[:2].upper()}{number * 13} on {date(rng)}.',
    }
    others = [TOOLS[key] for key in rng.sample([k for k in sorted(TOOLS) if k != name], rng.randrange(0, 2))]
    tools = [TOOLS[name]] + others
    rng.shuffle(tools)
    return [{'role': 'user', 'content': requests[name]}], tools


def chat(rng):
    topic = rng.choice(['leaves change color in autumn', 'bread rises', 'the sky looks blue',
                        'ice floats on water', 'cats purr', 'metal feels cold to touch', 'onions make eyes water',
                        'bicycles stay upright when moving', 'tea cools faster in a wide cup', 'owls hunt at night'])
    activity = rng.choice(['keeping houseplants alive', 'learning to juggle', 'packing light for a trip',
                           'running a short meeting', 'reading more books', 'sleeping on a long flight',
                           'starting a vegetable garden', 'practising a new language', 'organising a small desk',
                           'cooking rice on a stove'])
    return [{'role': 'user', 'content': rng.choice([
        f'In two sentences, explain why {topic}.', f'Give three short tips for {activity}.',
        f'What is one common mistake people make when {activity}? Answer briefly.',
        f'Describe {activity} to a complete beginner in one sentence.'])}], None


GENERATORS = {'conversation': conversation, 'instruction': instruction, 'tool-use': tool_use, 'chat': chat}


def grams(text, n=8):
    words = re.findall(r'\w+', text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def prompts(seed, protected_tasks):
    """Distinct generated prompts; raises if any overlaps a protected task."""
    rng = random.Random(seed)
    protected_messages = {identity(task['messages']) for task in protected_tasks}
    protected_grams = set().union(*(grams(m['content']) for task in protected_tasks for m in task['messages']))
    rows, seen = [], set()
    for category, count in COUNTS.items():
        made = 0
        for _ in range(count * 50):
            if made == count:
                break
            messages, tools = GENERATORS[category](rng)
            key = identity([messages, tools])
            if key in seen:
                continue
            if identity(messages) in protected_messages or any(grams(m['content']) & protected_grams for m in messages):
                raise ValueError('replay prompt overlaps a protected task')
            seen.add(key)
            rows.append({'id': f'replay-{category}-{made:03d}', 'category': category,
                         'messages': messages, 'tools': tools})
            made += 1
        if made != count:
            raise ValueError(f'could not generate {count} distinct {category} prompts')
    return rows


def replay_item(prompt, response):
    """Replay target: the parent's terminated greedy answer as the only trainable message."""
    if not response['terminated']:
        return None
    messages = prompt['messages'] + [{'role': 'assistant', 'content': response['text']}]
    return {'id': prompt['id'], 'messages': messages, 'tools': prompt['tools'],
            'trainable': [False] * len(prompt['messages']) + [True],
            'prompt_sha256': identity(prompt), 'response_token_ids': response['token_ids']}
