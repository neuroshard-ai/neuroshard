"""Freeze the complete reranked service over the recorded failed feed model."""
import json
from pathlib import Path
import shutil

from neuroshard.evolution import answering, question_reranking, request_planning
from neuroshard.evolution.objects import Objects
from neuroshard.evolution.reference_data import identity, save
from run_lossless_diagnostic import assemble

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = Path('/home/ubuntu/neuroshard/.neuroshard/admission-native-campaign-20260918')
HOME = Path('/home/ubuntu/neuroshard/.neuroshard/feed-reranker-serving-20260919')
JOB = CAMPAIGN/'jobs/096944deb1a1d6a1fae75e72cea561072f11ab4a50440053d5a39638a1353380'


def main():
    read = lambda path: json.loads(path.read_bytes())
    store = Objects(CAMPAIGN/'compiled/objects')
    job = read(JOB/'job.json')
    semantics = answering.load(job['lifecycle']['candidate_template'], store)['semantic_questions']
    probe = ROOT/'config/experiments/question-reranker-probe-20260919'
    repacked = read(probe/'repack-result.json')
    assert repacked['all_tensors_equal'] and repacked['tensor_count'] == 393
    assert repacked['upstream_sha256'] == read(probe/'result.json')['files']['model.safetensors']
    model = {'format': question_reranking.MODEL, 'repository': 'BAAI/bge-reranker-v2-m3',
        'revision': question_reranking.REVISION, 'parameters': question_reranking.PARAMETERS,
        'upstream_sha256': repacked['upstream_sha256'], 'max_tokens': 512, 'batch_size': 8, 'scale': 1024,
        'files': {name: spec for name, spec in repacked['files'].items() if name != 'README.md'}}
    fitting = read(CAMPAIGN/'compiled/selector-fitting.json')['rows']
    catalog = read(CAMPAIGN/'compiled/source-catalog.json')
    facts = {fact['id']: fact for route in ('conversation', 'feed') for fact in catalog['cohorts'][route]}
    families = {}
    for intent in sorted(semantics['intents']):
        families[intent] = []
        for question in facts[intent]['training']:
            rows = [row for row in fitting if row['intent'] == intent and row['question'] == question]
            assert len(rows) == 1
            families[intent].append({'id': rows[0]['id'], 'question': question})
    policy = {'format': question_reranking.FORMAT, 'model': model, 'owner': 3, 'families': families}
    question_reranking.validate(policy, semantics, 8)
    plan = assemble(CAMPAIGN, JOB, HOME, semantic_policy=semantics,
        request_policy=request_planning.MODAL_SPAN_POLICY, question_reranker=policy)
    for name in ('allocation.json', 'known_hosts'):
        shutil.copyfile(CAMPAIGN/name, HOME/name)
    inventory = read(CAMPAIGN/'auxiliary-assets.json')
    inventory['question_rerankers'] = {identity(model): {'files': model['files'],
        'notices': {'README.md': repacked['files']['README.md']}}}
    save(HOME/'auxiliary-assets.json', inventory)
    save(HOME/'question-reranker.json', policy)
    save(HOME/'deployment.json', {'source_home': '/home/ubuntu/neuroshard-reranker-study',
        'original_source_home': '/home/ubuntu/neuroshard-study',
        'allocation_deadline': read(HOME/'allocation.json')['deadline'],
        'training': False, 'new_final': False, 'native_promotion': False})
    destination = Path(__file__).resolve().parent
    for name in ('diagnostic.json', 'deployment.json', 'question-reranker.json'):
        shutil.copyfile(HOME/name, destination/name)
    print(json.dumps(plan), flush=True)


if __name__ == '__main__':
    main()
