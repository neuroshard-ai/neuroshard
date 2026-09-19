#!/usr/bin/env python3
"""Continue useful learning from a replayed, natively accepted answering graph.

Source changes require a new research genesis. This importer carries model and
evaluation history, not balances, signing state or the rejected expert. Subsequent
cohorts use the ordinary automatic operator without another genesis edit.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess

from neuroshard.dataflow.store import canonical
from neuroshard.evolution import answering, ordinary_cohorts, request_planning, semantic_questions
from neuroshard.evolution.access_routing import question_key
from neuroshard.evolution.data import document_identity
from neuroshard.evolution.expert_preparation import record_set
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save, sha256
import prepare_semantic_cohorts as compiler

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_bytes())


def accepted_graph(state, replay):
    if (replay.get('passed') is not True or replay.get('final_state_root') != identity(state)
            or replay.get('final_height') != state['height']
            or replay.get('issued_atoms') != state['issued']):
        raise ValueError('Require the exact completely replayed native state')
    graph = state['expert_lifecycle']['serving_graph']
    accepted = [row for row in state['expert_lifecycle']['history']
                if row.get('kind') == 'quality' and row.get('promoted') is True]
    if (len(accepted) != 1 or accepted[0]['report']['passed'] is not True
            or accepted[0]['report']['candidate_graph'] != identity(graph)
            or state['serving_root'] != identity(graph)
            or set(graph['experts']) != {'directory', 'protocol', 'planner', 'conversation'}):
        raise ValueError('Carry the one admitted conversation graph, never the rejected candidate')
    return copy.deepcopy(graph), copy.deepcopy(accepted)


def prepare(home, previous, facts_path, reranker_path, bootstrap_path):
    home, previous = Path(home), Path(previous)
    state, replay = read(previous/'final-state.json'), read(previous/'ledger-replay.json')
    graph, accepted = accepted_graph(state, replay)
    old = Objects(previous/'compiled/objects')
    initial = answering.load(graph, old)
    semantics = initial['semantic_questions']
    if semantics['format'] != semantic_questions.ADMISSION_FORMAT:
        raise ValueError('Preserve explicit accepted semantic admission')
    original_catalog = read(previous/'compiled/source-catalog.json')
    fresh = read(facts_path)
    if fresh['cohort'] != 'storage' or fresh['source_revision'] != original_catalog['revision']:
        raise ValueError('Keep the new storage cohort grounded in the same pinned source')
    order = ['audit', 'storage']
    # Audit bytes were committed before the previous run and never evaluated.
    # Reading their sealed inventory to copy it is not a new scoring operation.
    if any('audit' in job.get('lifecycle', {}).get('candidate_template', {}).get('experts', {})
           for job in (read(path) for path in (previous/'jobs').glob('*/job.json'))):
        raise ValueError('The inherited audit cohort has already entered execution')
    catalog = {'revision': original_catalog['revision'], 'order': order,
               'cohorts': {'audit': original_catalog['cohorts']['audit'], 'storage': fresh['facts']}}
    if any(len(facts) != 16 or len({fact['id'] for fact in facts}) != 16
           for facts in catalog['cohorts'].values()):
        raise ValueError('Require sixteen distinct source facts per fresh cohort')
    evidence = ordinary_cohorts.source_evidence(catalog, lambda revision, path:
        subprocess.check_output(['git', 'show', revision+':'+path], cwd=ROOT))
    carried = {fact['id']: fact for fact in original_catalog['cohorts']['conversation']}
    if set(carried) != set(semantics['intents']):
        raise ValueError('Carry every accepted semantic intent and only its trained facts')
    fitting = read(previous/'compiled/selector-fitting.json')
    rows = {row['id']: copy.deepcopy(row) for row in fitting['rows']
            if row['route'] in {'parent', *graph['experts']}}
    if not rows or any(row['intent'] != 'parent' and row['intent'] not in carried for row in rows.values()):
        raise ValueError('Inherited selector includes unaccepted specialist examples')
    anchors = read(previous/'compiled/anchors.json')
    known = {document_identity(row['messages']) for values in anchors.values() for row in values}
    for record in accepted:
        policy = old.json(record['report']['policy_root'])
        spec = policy['roles']['test']
        tests = [json.loads(line) for line in old.get(spec['sha256']).splitlines()]
        if len(tests) != spec['count']:
            raise ValueError('Accepted evaluation history is incomplete')
        for row in tests:
            key = document_identity(row['messages'])
            if key in known:
                raise ValueError('Accepted evaluation history collides with baseline anchors')
            known.add(key)
            anchors['retained-test-knowledge'].append(row)
    # New mixed questions cross both the original and the admitted expert.
    # Earlier final inputs become retention anchors, never training examples.
    atom_rows = [row for row in anchors['retained-test-knowledge'] if row['stratum'] == 'single']
    original_atoms = [row for row in atom_rows if row['topics'][0] not in carried][:4]
    admitted_atoms = [row for row in atom_rows if row['topics'][0] in carried][:4]
    if len(original_atoms) != 4 or len(admitted_atoms) != 4:
        raise ValueError('Require mixed evaluation against both accepted generations')
    atoms = [{'question': row['messages'][0]['content'], 'topic': row['topics'][0],
              'answer': row['answers'][0], 'aliases': row['answer_aliases'][0]}
             for row in original_atoms+admitted_atoms]
    home.mkdir(parents=True, exist_ok=False)
    compiled = home/'compiled'
    compiled.mkdir()
    store = Objects(compiled/'objects')
    for name in order:
        train, annotations = ordinary_cohorts.training(catalog['cohorts'][name])
        folder = compiled/name
        folder.mkdir()
        (folder/'training.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in train))
        if name == 'audit':
            shutil.copyfile(previous/'compiled/audit/final.jsonl', folder/'final.jsonl')
            if sha256(folder/'training.jsonl') != sha256(previous/'compiled/audit/training.jsonl'):
                raise ValueError('Do not change the earlier unopened audit training prescription')
        else:
            (folder/'final.jsonl').write_bytes(b''.join(canonical(row)+b'\n'
                for row in ordinary_cohorts.evaluation(catalog['cohorts'][name], atoms)))
        save(folder/'training-annotations.json', annotations)
        for row in annotations:
            if row['id'] in rows:
                raise ValueError('Fresh fitting overlaps the accepted training inventory')
            rows[row['id']] = {'id': row['id'], 'document': row['id'], 'route': name,
                               'intent': row['topic'], 'question': row['question']}
    heldout = {question_key(fact['test']) for values in catalog['cohorts'].values() for fact in values}
    heldout.update(question_key(row['messages'][-2]['content']) for values in anchors.values() for row in values)
    if any(question_key(row['question']) in heldout for row in rows.values()):
        raise ValueError('Fresh finals and accepted evaluations cannot become training inputs')
    bootstrap = read(bootstrap_path)
    if (set(bootstrap) != {'questions'} or len(bootstrap['questions']) != 4
            or len({row['topic'] for row in bootstrap['questions']}) != 4
            or not {row['topic'] for row in bootstrap['questions']} <= {fact['id'] for fact in catalog['cohorts']['audit']}
            or any(question_key(row['question']) in heldout | {question_key(row['question']) for row in rows.values()}
                   for row in bootstrap['questions'])):
        raise ValueError('Freeze four separate rejection questions from the actual first cohort')
    rules = read(previous/'compiled/quality-rule.json')
    rules['retention_anchors'] = {role: record_set(store, role,
        [{**row, 'id': document_identity(row['messages'])} for row in values]) for role, values in anchors.items()}
    history = {'format': 'neuroshard-accepted-learning-import-v1',
        'genesis': replay['genesis'], 'state': identity(state), 'height': state['height'],
        'ledger_replay': identity(replay), 'graph': identity(graph), 'accepted': accepted,
        'evaluation_history_rows': 32, 'rejected_cohorts_counted': 0,
        'scope': 'Accepted neural state and complete evaluation history; new research genesis for changed source, no ledger balances imported.'}
    questions = [{'id': row['id'], 'question': row['question']} for row in sorted(rows.values(), key=lambda row: row['id'])]
    if len(questions) > semantic_questions.MAX_ROWS:
        raise ValueError('The cumulative question inventory exceeds its declared bound')
    reranking = read(reranker_path)
    profile = read(previous/'compiled/baseline-profile.json')
    for name, value in {'baseline-core.json': answering.core(graph), 'baseline-profile.json': profile,
            'baseline-configuration.json': initial, 'source-catalog.json': catalog, 'source-evidence.json': evidence,
            'anchors.json': anchors, 'quality-rule.json': rules, 'bootstrap-questions.json': bootstrap,
            'accepted-facts.json': carried, 'accepted-history.json': history,
            'selector-fitting.json': {'format': ordinary_cohorts.FORMAT+'/semantic-fitting',
                'rows': list(rows.values()), 'base_ids': fitting['base_ids'],
                'policy': 'Keep accepted training and retention history; append only fresh training questions.'}}.items():
        save(compiled/name, value)
    shutil.copytree(previous/'compiled/seed', compiled/'seed')
    save(home/'questions.json', questions)
    save(home/'encoder.json', semantics['encoder'])
    save(home/'feature-profile.json', initial['learned']['feature_profile'])
    plan = {**read(previous/'input-plan.json'), 'order': order, 'questions': identity(questions),
        'quality_rule': identity(rules), 'driver': sha256(compiler.__file__),
        'accepted_history': identity(history), 'continuation_driver': sha256(__file__),
        'source_facts': sha256(facts_path), 'composition': request_planning.MODAL_SPAN_POLICY,
        'question_reranker': {key: reranking[key] for key in ('format', 'model', 'owner')},
        'previous_failed_cohort_counted': False, 'neural_training_started': False}
    save(home/'input-plan.json', plan)
    return plan
