"""Tensor-free commitments for bounded, replayable planner updates.

Commitments do not establish correct computation or authorize issuance. Native
acceptance must fund complete replay from the window's input weights and Adam
state. Work identities omit job names and document metadata.
"""
import copy
import math

from .reference_data import identity
from .schema import integer, root
from .serving_graph import fields

FORMAT = 'neuroshard-owned-planner-window-v1'
CHECKPOINT = 'neuroshard-fusion-training-checkpoint-v1'


def checkpoint(value):
    fields(value, {'format', 'binding', 'step', 'fusion', 'sha256', 'bytes'}, 'Invalid planner checkpoint')
    if value['format'] != CHECKPOINT or not isinstance(value['binding'], dict):
        raise ValueError('Require the complete owned planner checkpoint')
    integer(value['step'], 0, 4096)
    integer(value['bytes'], 1, 2*1024**3)
    root(value['fusion'])
    root(value['sha256'])
    binding = value['binding']
    required = {'format', 'graph', 'source', 'rows', 'recipe', 'max_length', 'adapter_rank', 'layout'}
    fields(binding, required | ({'initial_weights'} if 'initial_weights' in binding else set()),
           'Invalid planner checkpoint binding')
    if binding['format'] != 'neuroshard-owned-planner-training-v1':
        raise ValueError('Checkpoint belongs to another training program')
    for name in ('graph', 'source', 'rows', 'recipe'):
        root(binding[name])
    integer(binding['max_length'], 2, 1024)
    integer(binding['adapter_rank'], 1, 128)
    layout = binding['layout']
    fields(layout, {'format', 'source', 'rank', 'projections', 'scale', 'initialization', 'base_weights'},
           'Invalid owned planner layout')
    if (layout['format'] != 'neuroshard-owned-expert-interface-v1'
            or layout['source'] != binding['source'] or layout['rank'] != binding['adapter_rank']
            or type(layout['scale']) is not int or layout['scale'] != 1
            or layout['initialization'] != 'zero-output' or layout['base_weights'] != 'frozen'
            or not isinstance(layout['projections'], dict) or not 1 <= len(layout['projections']) <= 224):
        raise ValueError('Planner layout differs from its declared owned source')
    parameters = 0
    for name, shape in layout['projections'].items():
        if not isinstance(name, str) or not 1 <= len(name) <= 128 or not isinstance(shape, list) or len(shape) != 2:
            raise ValueError('Bound every owned planner projection')
        parameters += binding['adapter_rank']*sum(integer(side, 1, 32768) for side in shape)
    if parameters > 64_000_000:
        raise ValueError('Planner parameters exceed the native window bound')
    return value


def validate_recipe(recipe, row_count):
    fields(recipe, {'steps', 'learning_rate', 'warmup_steps', 'minimum_lr_ratio',
                    'weight_decay', 'clip_norm', 'microbatch', 'schedule'}, 'Invalid planner recipe')
    steps = integer(recipe['steps'], 1, 4096)
    integer(recipe['warmup_steps'], 0, steps-1)
    integer(recipe['microbatch'], 1, 16)
    integer(row_count, 1, 2048)
    for key, high in [('learning_rate', .01), ('minimum_lr_ratio', 1), ('weight_decay', 1), ('clip_norm', 100)]:
        value = recipe[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= high:
            raise ValueError('Invalid finite planner optimizer value')
    if recipe['learning_rate'] == 0 or recipe['clip_norm'] == 0:
        raise ValueError('Positive learning rate and clipping are required')
    schedule = recipe['schedule']
    if not isinstance(schedule, list) or len(schedule) != steps:
        raise ValueError('Prescribe every planner update')
    for batch in schedule:
        if not isinstance(batch, list) or not 1 <= len(batch) <= 64:
            raise ValueError('Bound the complete numerical batch')
        for index in batch:
            integer(index, 0, row_count-1)
        if len(set(batch)) != len(batch):
            raise ValueError('A planner batch repeats a selected row')


def batch_root(rows, vocabulary, maximum):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError('Require a bounded ordered numerical batch')
    numerical = []
    for row in rows:
        ids, labels = row['input_ids'], row['labels']
        if (not isinstance(ids, list) or not 2 <= len(ids) <= maximum
                or not isinstance(labels, list) or len(labels) != len(ids) or labels[0] != -100):
            raise ValueError('Require complete response-only token inputs')
        for token, label in zip(ids, labels):
            integer(token, 0, vocabulary-1)
            if type(label) is not int or label not in (-100, token):
                raise ValueError('A label differs from its actual input token')
        count = sum(label != -100 for label in labels[1:])
        if type(row['targets']) is not int or row['targets'] != count or count < 1:
            raise ValueError('A row changed its response target count')
        # Padding, masks and per-document normalization follow deterministically
        # from these ordered arrays and the prescribed microbatch size.
        numerical.append({'input_ids': ids, 'labels': labels})
    return identity({'format': FORMAT+'/batch', 'rows': numerical})


def prescription(graph, rows, recipe, initial):
    checkpoint(initial)
    validate_recipe(recipe, len(rows))
    binding = initial['binding']
    if (initial['step'] != 0 or binding['graph'] != identity(graph)
            or binding['source'] != identity(graph['interpreter_assets']['partitions']['2'])
            or binding['recipe'] != identity(recipe) or binding['rows'] != identity(rows)):
        raise ValueError('Prescribe the exact initial state and complete planner inputs')
    batches = [batch_root([rows[i] for i in selected], graph['parent']['config']['vocab_size'],
                          binding['max_length']) for selected in recipe['schedule']]
    return {'format': FORMAT+'/prescription', 'graph': identity(graph), 'initial': copy.deepcopy(initial),
            'row_count': len(rows), 'recipe': copy.deepcopy(recipe), 'batches': batches,
            'numerical_profile': root(graph['numerical_profile']),
            'frozen': identity({'config': graph['parent']['config'],
                'layout': graph['descriptor']['parent_layout'], 'interpreter': graph['interpreter_assets']})}


def validate_profile(profile):
    fields(profile, {'format', 'graph', 'initial', 'row_count', 'recipe', 'batches',
                     'numerical_profile', 'frozen'}, 'Invalid planner work prescription')
    if profile['format'] != FORMAT+'/prescription':
        raise ValueError('Unsupported planner work prescription')
    for name in ('graph', 'numerical_profile', 'frozen'):
        root(profile[name])
    initial = checkpoint(profile['initial'])
    validate_recipe(profile['recipe'], profile['row_count'])
    if (initial['step'] != 0 or initial['binding']['graph'] != profile['graph']
            or initial['binding']['recipe'] != identity(profile['recipe'])
            or not isinstance(profile['batches'], list) or len(profile['batches']) != profile['recipe']['steps']):
        raise ValueError('Planner work changed its initial state or complete schedule')
    for batch in profile['batches']:
        root(batch)
    return profile


def learning_rate(recipe, step):
    integer(step, 0, recipe['steps']-1)
    warmup = recipe['warmup_steps']
    progress = (step-warmup)/max(1, recipe['steps']-warmup-1)
    factor = ((step+1)/warmup if step < warmup else recipe['minimum_lr_ratio']+
              (1-recipe['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))
    return recipe['learning_rate']*factor


def work_identity(profile, before, batch):
    """Bind consumed numerical state, not a publisher's labels or job ID.

    Safetensor checkpoint files contain weights and Adam tensors without job
    metadata. This identifies that serialization and numerical profile; it is
    not a claim of equivalence across arbitrary encodings or implementations.
    """
    recipe = profile['recipe']
    return identity({'format': FORMAT+'/update', 'frozen': root(profile['frozen']),
        'state': root(before['sha256']), 'layout': before['binding']['layout'],
        'batch': root(batch), 'numerical_profile': root(profile['numerical_profile']),
        'optimizer': {'method': 'float32-adamw', 'betas': [.9, .999], 'eps': 1e-8,
            'learning_rate': learning_rate(recipe, before['step']), 'weight_decay': recipe['weight_decay'],
            'clip_norm': recipe['clip_norm'], 'microbatch': recipe['microbatch']}})


def validate(profile, current, window, paid=()):
    validate_profile(profile)
    checkpoint(current)
    fields(window, {'format', 'prescription', 'checkpoints', 'updates'}, 'Invalid complete planner window')
    states, updates = window['checkpoints'], window['updates']
    if (window['format'] != FORMAT or window['prescription'] != identity(profile)
            or not isinstance(updates, list) or not 1 <= len(updates) <= 16
            or not isinstance(states, list) or len(states) != len(updates)+1 or states[0] != current
            or current['binding'] != profile['initial']['binding']
            or current['step']+len(updates) > profile['recipe']['steps']):
        raise ValueError('Bind a bounded consecutive window to the current prescribed state')
    work = []
    for offset, (before, after, metrics) in enumerate(zip(states, states[1:], updates)):
        checkpoint(before)
        checkpoint(after)
        fields(metrics, {'step', 'loss', 'gradient_norm', 'planner', 'learning_rate'}, 'Invalid planner measurements')
        step = current['step']+offset
        if (before['binding'] != current['binding'] or after['binding'] != current['binding']
                or before['step'] != step or after['step'] != step+1
                or type(metrics['step']) is not int or metrics['step'] != after['step']
                or metrics['planner'] != after['fusion']):
            raise ValueError('A planner step changed its cursor, state or owned layout')
        for name in ('loss', 'gradient_norm', 'learning_rate'):
            value = metrics[name]
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Require finite nonnegative planner measurements')
        if metrics['learning_rate'] != learning_rate(profile['recipe'], step):
            raise ValueError('A planner step changed its prescribed effective learning rate')
        key = work_identity(profile, before, profile['batches'][step])
        if key in work or key in paid:
            raise ValueError('This numerical planner update has already been paid')
        work.append(key)
    return {'record_root': identity(window), 'work_ids': work, 'steps': len(updates),
            'input_state': current['sha256'], 'output_state': states[-1]['sha256'],
            'requires_complete_replay': True}
