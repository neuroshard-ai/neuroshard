"""Numerical inference audits for the committed expert graph.

Callers provide a locally configured GraphNetwork. Every verdict invokes its
actual neural path; a miner's report is never accepted as an execution result.
"""
from .. import expert_lifecycle, serving_graph
from ..reference_data import identity


def inference_transcript(claim, result):
    return {'format': expert_lifecycle.FORMAT + '/inference-transcript',
            'binding': expert_lifecycle.transcript_binding(claim), 'result': result}


def inference_report(claim, network):
    if (claim['kind'] != 'expert_inference' or claim['model_root'] != identity(claim['graph'])
            or claim['executor_root'] != claim['graph']['executor_root']):
        raise ValueError('Invalid committed graph inference obligation')
    request = claim['request']
    serving_graph.fields(request, {'question', 'max_tokens', 'calls'}, 'Invalid raw graph request')
    if request['calls'] != serving_graph.calls(claim['graph'], request['question'], request['max_tokens']):
        raise ValueError('Claim changed the request-derived neural calls')
    serving_graph.payments(claim['graph'], request['calls'], claim['outputs'], 1)
    if claim['stages'] != sum(len(output['token_ids']) for output in claim['outputs']):
        raise ValueError('Audit all actual neural output tokens')
    # Missing model objects, runtime disagreement, execution failure and timeout
    # propagate as unavailable. They never become an affirmative report.
    result = network.answer(request['question'], request['max_tokens'], claim['graph'])
    expected = {'graph': identity(claim['graph']), 'request': request,
                'outputs': claim['outputs'], 'text': claim['text']}
    valid = result == expected and identity(inference_transcript(claim, result)) == claim['record_root']
    report = {'format': expert_lifecycle.FORMAT + '/replay',
        'statement': identity(expert_lifecycle.service_statement(claim)),
        'stages': [{'stage': index, 'valid': valid} for index in range(claim['stages'])]}
    expert_lifecycle.replay_report(claim, report)
    return report, result
