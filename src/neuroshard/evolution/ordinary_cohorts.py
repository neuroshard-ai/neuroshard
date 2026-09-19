"""Reconstruct prospectively frozen ordinary learning cohorts from sources.

The compiler is a publisher/curator tool. Fact tables and answer metadata never
enter a serving process. A feed transports the resulting immutable records;
native preparation, execution audits and complete quality decide admission.
"""
import copy
import json
import math
from pathlib import Path

from . import expert_source, ordinary_quality
from .access_routing import question_key
from .data import document_identity
from .expert_curriculum import PREFIXES
from .reference_data import identity

FORMAT = 'neuroshard-ordinary-cohort-campaign-v1'
CONTRACTS = (('', ''), *PREFIXES)
ORDER = ('admission', 'conversation', 'feed')


def catalogs(source):
    source = Path(source)/'config/experiments'
    old = json.loads((source/'continual-expert-facts.json').read_bytes())
    variations = json.loads((source/'ordinary-cohort-questions.json').read_bytes())
    feed = json.loads((source/'ordinary-feed-facts.json').read_bytes())
    if variations['source_revision'] != feed['source_revision']:
        raise ValueError('Use one immutable source revision for these facts')
    result = {}
    for cohort in old['cohorts']:
        name = cohort['id']
        if name not in ORDER[:2]:
            continue
        if set(variations[name]) != {fact['id'] for fact in cohort['facts']}:
            raise ValueError('Question variations must cover every declared source fact')
        result[name] = [{
            'id': fact['id'], 'answer': fact['answer'],
            'training': [fact['train_question'], *variations[name][fact['id']]['training']],
            'test': variations[name][fact['id']]['test'],
            'path': fact['source']['path'], 'evidence': fact['source']['evidence']}
            for fact in cohort['facts']]
    result['feed'] = copy.deepcopy(feed['facts'])
    forbidden = {question_key(fact['test_question']) for cohort in old['cohorts'] for fact in cohort['facts']}
    all_training, all_test = set(), set()
    for name in ORDER:
        facts = result[name]
        if len(facts) != 16 or len({fact['id'] for fact in facts}) != 16:
            raise ValueError('Declare sixteen distinct facts per new cohort')
        for fact in facts:
            values = fact['training']
            if (len(values) != 4 or len({question_key(value) for value in values}) != 4
                    or not isinstance(fact['answer'], str) or not fact['answer']
                    or Path(fact['path']).is_absolute() or '..' in Path(fact['path']).parts
                    or not fact['evidence']):
                raise ValueError('Require complete distinct training questions and source evidence')
            all_training.update(question_key(value) for value in values)
            key = question_key(fact['test'])
            if key in all_test or key in forbidden:
                raise ValueError('Do not reuse an earlier final or duplicate a new final question')
            all_test.add(key)
    if all_training & (all_test | forbidden):
        raise ValueError('Final wording must not enter training or selector fitting')
    return {'revision': variations['source_revision'], 'cohorts': result}


def source_evidence(catalog, read_revision):
    """The caller supplies pinned repository bytes, independently of feed rows."""
    from .objects import digest
    files, facts = {}, {}
    for values in catalog['cohorts'].values():
        for fact in values:
            path = fact['path']
            if path not in files:
                raw = read_revision(catalog['revision'], path)
                if not isinstance(raw, bytes) or len(raw) > 2*1024**2:
                    raise ValueError('Bound source evidence read from the pinned revision')
                files[path] = {'sha256': digest(raw), 'bytes': len(raw), 'text': raw.decode('utf-8')}
            text = files[path]['text']
            if fact['evidence'] not in text:
                raise ValueError('A curated fact no longer matches its declared source: '+fact['id'])
            facts[fact['id']] = {'path': path, 'sha256': files[path]['sha256'], 'evidence': fact['evidence'],
                'line': text[:text.index(fact['evidence'])].count('\n')+1, 'answer': fact['answer']}
    return {'revision': catalog['revision'], 'files': {
        name: {key: value for key, value in spec.items() if key != 'text'} for name, spec in files.items()},
        'facts': facts, 'scope': 'Curated question meanings; checked code evidence, not a universal truth oracle.'}


def scored(messages, topics, answers, aliases=None, sensitive=None):
    row = {'quality_format': ordinary_quality.FORMAT,
           'stratum': 'single' if len(answers) == 1 else 'composed',
           'topics': list(topics), 'answers': list(answers),
           'messages': [*copy.deepcopy(messages), {'role': 'assistant', 'content': '; '.join(answers)}],
           'answer_aliases': (copy.deepcopy(aliases) if aliases is not None else
                              [[answer+'.'] for answer in answers]),
           'case_sensitive': list(sensitive) if sensitive is not None else [False]*len(answers)}
    ordinary_quality.validate_rows([{**row, 'id': document_identity(row['messages'])}])
    return row


def training(facts):
    """Each consecutive group covers all facts; six actual input contracts."""
    rows, metadata, seen = [], [], set()
    for index in range(4):
        for prefix, suffix in CONTRACTS:
            for fact in facts:
                question = prefix+fact['training'][index]+suffix
                messages = [{'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': fact['answer']}]
                key = document_identity(messages)
                if key in seen:
                    raise ValueError('Duplicate training document across crossed contracts')
                seen.add(key)
                row = {'messages': messages}
                expert_source.record(row, 'train')
                rows.append(row)
                metadata.append({'id': key, 'topic': fact['id'], 'question': question,
                                 'core': fact['training'][index], 'contract': [prefix, suffix]})
    return rows, metadata


def evaluation(facts, retained_atoms):
    """Sixteen unseen singles and sixteen natural two-part requests.

    Eight pairs combine two new facts, eight cross an accepted expert boundary.
    Gold answers/route names are never interpolated into a user question.
    """
    if not retained_atoms:
        raise ValueError('Mixed evaluation requires accepted earlier knowledge')
    singles = [scored([{'role': 'user', 'content': fact['test']}], [fact['id']], [fact['answer']])
               for fact in facts]
    pairs = []
    for index, left in enumerate(facts):
        if index < 8:
            right = facts[index+8]
            q, topic, answer, aliases = right['test'], right['id'], right['answer'], [right['answer']+'.']
        else:
            right = retained_atoms[(index-8) % len(retained_atoms)]
            q, topic, answer, aliases = (right[key] for key in ('question', 'topic', 'answer', 'aliases'))
        # Ordinary coordinated questions, not the explicit First/Second protocol.
        question = left['test']+' Also, '+q[0].lower()+q[1:]
        pairs.append(scored([{'role': 'user', 'content': question}],
            [left['id'], topic], [left['answer'], answer], [[left['answer']+'.'], aliases]))
    return singles+pairs


def assistant_anchors(source):
    value = json.loads((Path(source)/'config/experiments/ordinary-assistant-anchors.json').read_bytes())
    result = {}
    for kind in ('skills', 'conversation'):
        result['retained-test-'+kind] = [scored(
            [*row.get('history', []), {'role': 'user', 'content': row['question']}],
            ['assistant/'+kind+'/'+row['id']], [row['answer']], [row['aliases']],
            [row.get('case_sensitive', False)]) for row in value[kind]]
    return result


def verify_feed_rows(catalog, expected, actual):
    """Reject substitutions even when their new hashes are internally valid.

    This deliberately bounded curator authorizes only the prospectively
    reviewed source corpus. General untrusted web curation needs another policy.
    """
    if not catalog.get('facts') or actual != expected:
        raise ValueError('Feed rows differ from the independently source-grounded corpus')
    return {'format': FORMAT+'/source-review', 'evidence': identity(catalog),
            'records': identity(actual), 'source_correspondence_passed': True}


def extend(graph, name, checkpoint):
    """Append one owned tail; accepted earlier model references stay exact."""
    from . import answering, serving_graph
    if name in graph['experts']:
        raise ValueError('Isolated growth requires a distinct new expert')
    result = copy.deepcopy(answering.core(graph))
    result['experts'][name] = copy.deepcopy(checkpoint)
    descriptor = result['descriptor']
    descriptor.update(format=serving_graph.EXTENSIBLE,
                      previous_graph=identity(graph['descriptor']))
    descriptor['experts'].append({'id': name, 'checkpoint': checkpoint['checkpoint']})
    descriptor['rules'].append({'id': name, 'needle': name+' expert',
                                'owner': 2+len(result['experts'])})
    descriptor['total_parameters'] += sum(math.prod(value['shape']) for value in checkpoint['tensors'].values())
    return serving_graph.validate(result, allow_untrained=True)


def bind_policy(graph, payload, store):
    """A frozen topology policy binds the actual initialized numerical graph."""
    from . import answering
    if payload['format'] != answering.FORMAT+'/policy':
        raise ValueError('Require the complete frozen answering policy')
    config = copy.deepcopy(payload['configuration'])
    config['graph'] = config['learned']['graph'] = identity(answering.core(graph))
    bound = answering.attach(graph, config, store)
    if bound['answering']['policy_root'] != identity(payload):
        raise ValueError('The prospective answering policy changed during initialization')
    return bound
