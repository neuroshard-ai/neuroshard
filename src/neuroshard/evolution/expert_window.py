"""Bind a bounded expert claim to its actual intermediate numerical work.

Consensus can check these commitments without loading a tensor. Acceptance still
requires execution audits of both the prefix production and the tail updates.
"""
import math

from . import expert_checkpoint as checkpoint
from .reference_data import identity
from .schema import root

FORMAT = 'neuroshard-expert-replay-window-v1'
FIELDS = {'format', 'input', 'output', 'steps', 'numerical_profile'}
STEP_FIELDS = {'index', 'input_checkpoint', 'output_checkpoint', 'batch', 'work_identity', 'metrics'}
METRICS = {'step', 'loss', 'response_loss', 'reference_kl', 'reference_margin',
           'gradient_norm', 'learning_rate', 'weighted_targets', 'anchor_targets'}


def validate(parent, current, window, intermediates, expected_batches, numerical_profile, paid=()):
    if not isinstance(window, dict) or set(window) != FIELDS or window['format'] != FORMAT:
        raise ValueError('Invalid expert window schema')
    count = checkpoint.transition(parent, window['input'], window['output'])
    if window['input'] != current:
        raise ValueError('Claim the current expert checkpoint')
    if window['numerical_profile'] != root(numerical_profile):
        raise ValueError('A claim cannot change its numerical profile')
    if (not isinstance(intermediates, list) or len(intermediates) != count + 1
            or intermediates[0] != current or intermediates[-1] != window['output']
            or not isinstance(window['steps'], list) or len(window['steps']) != count
            or not isinstance(expected_batches, list) or len(expected_batches) != count):
        raise ValueError('Include every intermediate checkpoint and prescribed input batch')
    work_ids = []
    for offset, (before, after, step, batch) in enumerate(zip(
            intermediates, intermediates[1:], window['steps'], expected_batches)):
        checkpoint.transition(parent, before, after, 1)
        if (not isinstance(step, dict) or set(step) != STEP_FIELDS
                or type(step['index']) is not int or step['index'] != current['step'] + offset
                or step['input_checkpoint'] != before['checkpoint']
                or step['output_checkpoint'] != after['checkpoint'] or step['batch'] != root(batch)):
            raise ValueError('An intermediate step changed its state, cursor or prescribed batch')
        metrics = step['metrics']
        if (not isinstance(metrics, dict) or set(metrics) != METRICS
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values())
                or type(metrics['step']) is not int or metrics['step'] != after['step']):
            raise ValueError('Invalid bound training measurements')
        work = checkpoint.work_identity(parent, before, batch, numerical_profile)
        if step['work_identity'] != work:
            raise ValueError('Work identity must derive from the actual consumed numerical state')
        if work in paid or work in work_ids:
            raise ValueError('This numerical update has already been paid')
        work_ids.append(work)
    return {'record_root': identity(window), 'work_ids': work_ids, 'steps': count,
            'output_checkpoint': window['output']['checkpoint']}
