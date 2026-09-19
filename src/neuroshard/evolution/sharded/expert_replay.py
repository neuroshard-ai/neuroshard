"""Replay an archived expert trajectory and emit bounded native work records."""
import time

from .. import expert_checkpoint as codec
from ..reference_data import identity, save
from . import features
from .expert_commitment import snapshot


def batch_identity(manifest_batch):
    # Tensor files contain actual prefix/reference/ID/mask/weight arrays.
    # Descriptive record IDs and local filenames do not alter the computation.
    return identity([{'sha256': spec['sha256'],
                      'reference_aliases_prefix': spec['reference_aliases_prefix']}
                     for spec in manifest_batch['files']])


def replay(shard, head, optimizer, parent, job, recipe, objective, bank, records,
           batches, schedule, expected_metrics, expected_checkpoints, numerical_profile,
           home, max_seconds):
    if len(schedule) != recipe['steps'] or len(expected_metrics) != len(schedule):
        raise ValueError('Replay every prescribed update and its observed trajectory')
    home.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    current = snapshot(shard, optimizer, parent, job, 0, recipe)
    if current['checkpoint'] != expected_checkpoints[0]:
        raise ValueError('Replay must start at the actual archived initial checkpoint')
    save(home / 'checkpoint-000000.json', current)
    window_start, steps, windows = current, [], []
    metrics = ('step', 'loss', 'response_loss', 'reference_kl', 'reference_margin',
               'gradient_norm', 'learning_rate', 'weighted_targets', 'anchor_targets')
    for index, batch_index in enumerate(schedule):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('Bounded replay deadline expired')
        rows = [records[i] for i in batches[batch_index]]
        packets = bank.batch(batch_index, rows, shard.device_name)
        inputs = batch_identity(bank.manifest['batches'][batch_index])
        work = codec.work_identity(parent, current, inputs, numerical_profile)
        result = features.train_step(shard, head, optimizer, packets, rows, recipe, index,
                                     bank.microbatch, **objective)
        del packets
        if any(result[key] != expected_metrics[index][key] for key in metrics):
            save(home / 'trajectory-mismatch.json', {'step': index + 1, 'actual': result,
                                                     'expected': expected_metrics[index]})
            raise ValueError('Replayed update differs from the observed numerical trajectory')
        after = snapshot(shard, optimizer, parent, job, index + 1, recipe)
        codec.transition(parent, current, after, 1)
        if index + 1 in expected_checkpoints and after['checkpoint'] != expected_checkpoints[index + 1]:
            save(home / 'checkpoint-mismatch.json', after)
            raise ValueError('Replayed weights or Adam state differ from the archived checkpoint')
        save(home / f'checkpoint-{index + 1:06d}.json', after)
        step = {'index': index, 'input_checkpoint': current['checkpoint'],
                'output_checkpoint': after['checkpoint'], 'batch': inputs, 'work_identity': work,
                'metrics': {key: result[key] for key in metrics}}
        steps.append(step)
        current = after
        if len(steps) == 4 or index + 1 == len(schedule):
            codec.transition(parent, window_start, current, 4)
            window = {'format': 'neuroshard-expert-replay-window-v1',
                      'input': window_start, 'output': current, 'steps': steps,
                      'numerical_profile': numerical_profile}
            save(home / f'window-{window_start["step"]:06d}.json', window)
            windows.append(identity(window))
            print({'event': 'replayed-window', 'step': index + 1, 'checkpoint': current['checkpoint']}, flush=True)
            window_start, steps = current, []
    result = {'passed': True, 'updates': len(schedule), 'windows': windows,
              'checkpoint': current['checkpoint'], 'seconds': time.monotonic() - started,
              'prefix_recomputed': False, 'intermediate_tensor_objects_persisted': False,
              'tokens_issued': 0, 'native_activated': False}
    save(home / 'result.json', result)
    return result
