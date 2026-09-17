#!/usr/bin/env python3
"""Fit a general/specialist gate from training requests, without neural updates.

The opened bootstrap/assistant screen motivates the task families. Its actual
questions and all unexecuted cohort finals are excluded. Existing expert gates
and neural weights remain frozen. The resulting policy must be committed before
the next GPU diagnostic; fitting accuracy cannot substitute for answering.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

from neuroshard.evolution import expert_router, request_planning
from neuroshard.evolution.access_routing import question_key
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'neuroshard-general-interface-repair-v1'
GENERAL_INSTRUCTION = ('Follow the user\'s request using the conversation. Respect the requested output format '
    'and do not add explanation when the user asks for only an answer. Answer accurately and concisely.')


def general_requests():
    """Prompt-only routing examples, independent of the exposed answer targets."""
    result = []
    for index in range(24):
        left, right = 31+index, 46+2*index
        name = ('Talia', 'Jonas', 'Keiko', 'Omar')[index % 4]+str(index)
        result.extend([
            f'Calculate {left} + {right}. Reply with a number only.',
            f'What is {right+100} minus {left}? Do not include an explanation.',
            f'Which is larger: {left/10:.1f} or {right/10:.1f}? Just give the larger value.',
            f'Put {left}, {right}, {index+2} in increasing order. Return comma-separated numbers.',
            f'Write a JSON object whose key is active_{index} and whose value is false. No prose.',
            f'What Python expression accesses element {index} of a list called samples? Only the expression.',
            f'A cyclist moves at {left} kilometers per hour for three hours. Give the distance only.',
            f'Write just the third word of this phrase: quiet forest birds gather near station {index}.',
            f'My name is {name}.\nWhat name did I give you? Return the name alone.',
            f'Remember the identifier violet-{left}.\nWhat identifier did I ask you to remember?',
            f'The meeting starts at {index%12+1:02d}:45.\nGive the meeting time in HH:MM format only.',
            f'Set a running sum to {left}.\nIncrease it by {right}. Return the new sum as a number.',
            f'The green crate has {left} stones and the yellow crate has {right}.\nWhich crate has fewer stones? Give its color.',
        ])
    result.extend([
        'Why does warm air rise?', 'Explain how a rainbow forms.', 'What makes ocean water salty?',
        'What is the difference between a noun and a verb?', 'How does a bicycle stay balanced?',
        'Why do plants need sunlight?', 'What causes the seasons?', 'Describe evaporation in one sentence.',
        'Translate the English word dog into German. Give only the translation.',
        'Translate thank you into Italian. Reply with just the translation.',
        'What does bonsoir mean in English?', 'How would you say water in Spanish?',
        'Does every triangle have three sides? Answer yes or no.',
        'Which does not belong: oak, pine, wrench, birch? Return a single word.',
        'Continue the sequence 4, 8, 12, 16. Give the next value only.',
        'How many millimeters are in seven centimeters? Return the number.',
        'We will implement the service in Rust.\nWhich language have we chosen? Give only its name.',
        'Translate the next word into German. Give only the result.\nhouse',
    ])
    return result


def fitting_rows(previous):
    rows = {question_key(row['question']): {'question': row['question'],
        'route': 'parent' if row['route'] == 'parent' else 'specialist'} for row in previous}
    general = general_requests()
    for question in general:
        key = question_key(question)
        if key in rows and rows[key]['route'] != 'parent':
            raise ValueError('A general training request conflicts with an expert domain')
        rows[key] = {'question': question, 'route': 'parent'}
    # Mixed requests must still consult specialists. Negatives use training
    # inputs only; neither expected answers nor routing labels enter the text.
    specialist = sorted(row['question'] for row in previous if row['route'] != 'parent')
    for index in range(192):
        first = specialist[(index*13) % len(specialist)]
        second = general[(index*7) % len(general)]
        question = first+' Also, '+second
        rows[question_key(question)] = {'question': question, 'route': 'specialist'}
    return [{'id': identity({'request': row['question']}), **row} for _, row in sorted(rows.items())]


def run(campaign, embedding, home):
    from transformers import AutoTokenizer
    from neuroshard.evolution.sharded.router_features import EmbeddingFeatures
    home.mkdir(parents=True, exist_ok=False)
    old = json.loads((campaign/'compiled/selector-fitting.json').read_bytes())
    rows = fitting_rows(old['rows'])
    forbidden = set()
    for path in (campaign/'compiled').glob('*/final.jsonl'):
        for raw in path.read_bytes().splitlines():
            row = json.loads(raw)
            forbidden.add(question_key(row['messages'][-2]['content']))
    anchors = json.loads((campaign/'compiled/anchors.json').read_bytes())
    for values in anchors.values():
        for row in values:
            forbidden.add(question_key(row['messages'][-2]['content']))
            forbidden.add(question_key(request_planning.routing_context(row['messages'][:-1])))
    if any(question_key(row['question']) in forbidden for row in rows):
        raise ValueError('Keep exposed anchors and unopened finals out of gate fitting')
    prescription = {'format': FORMAT, 'source': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        'previous_fitting': sha256(campaign/'compiled/selector-fitting.json'),
        'rows': rows, 'epochs': 24, 'support_numerator': 5, 'support_denominator': 4,
        'request_policy': request_planning.ASSISTANT_POLICY, 'general_instruction': GENERAL_INSTRUCTION,
        'scope': 'Development repair of general selection and user-intent preservation; no new neural final scored.'}
    save(home/'prescription.json', prescription)
    store = Objects(campaign/'compiled/objects')
    operation = json.loads((campaign/'operation.json').read_bytes())
    graph = store.json(operation['baseline_graph'])
    config = store.json(graph['answering']['policy_root'])['configuration']
    feature = config['learned']['feature_profile']
    tokenizer = AutoTokenizer.from_pretrained(campaign/'compiled/seed',local_files_only=True)
    features = EmbeddingFeatures(embedding,feature['embedding_sha256'],tokenizer,graph['tokenizer']['root'],max_tokens=feature['max_tokens'])
    samples = [{'id':row['id'],'route':row['route'],'features':features(row['question'])} for row in rows]
    if features.profile != feature:
        raise ValueError('General routing changed the frozen embedding definition')
    save(home/'samples.json',samples)
    print(json.dumps({'phase':'fitting','examples':len(samples),'dimensions':feature['dimensions']}),flush=True)
    model = expert_router.fit(samples,embedding_root=features.root,tokenizer_root=graph['tokenizer']['root'])
    # The old model binds the feature-profile identity, not the raw tensor hash.
    if model['embedding_root'] != config['learned']['router']['embedding_root']:
        raise ValueError('The learned guard must use the existing route feature identity')
    guard = expert_router.fit_classifier(samples,expert_router.calibrate_support(samples,model),epochs=24,balance_classes=True)
    candidate = copy.deepcopy(config['learned']['router'])
    candidate['fallback_guard'] = guard
    expert_router.validate(candidate)
    save(home/'guard.json',guard)
    save(home/'router.json',candidate)
    scored = [expert_router.select(guard,row['features'])['route']==row['route'] for row in samples]
    result = {'format':FORMAT,'prescription':identity(prescription),'guard':identity(guard),'router':identity(candidate),
        'training':{name:{'correct':sum(ok for row,ok in zip(samples,scored) if row['route']==name),
                          'count':sum(row['route']==name for row in samples)} for name in ('parent','specialist')}}
    save(home/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--embedding',type=Path,required=True)
    parser.add_argument('--home',type=Path,required=True)
    args=parser.parse_args()
    run(args.campaign,args.embedding,args.home)
