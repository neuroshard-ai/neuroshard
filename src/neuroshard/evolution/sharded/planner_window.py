"""Produce or completely replay a bounded owned planner window.

Only the final assistant owner restores trainable tensors. The other assistant
owners execute their frozen partitions. An unavailable checkpoint is an error,
never a positive audit. Every replay starts from a fresh optimizer instance.
"""
import copy
import errno
import os
from pathlib import Path
import shutil
import tempfile

from .. import planner_window as window
from ..reference_data import identity, save, sha256
from ..schema import integer
from .planner_training import PlannerTraining


def restore(net, profile, current, rows, checkpoints):
    window.validate_profile(profile)
    window.checkpoint(current)
    if (profile != window.prescription(net.graph, rows, profile['recipe'], profile['initial'])
            or current['binding'] != profile['initial']['binding']
            or current['step'] >= profile['recipe']['steps']):
        raise ValueError('Replay requires the exact graph, source rows and unexhausted state')
    if net.all_owners.exchange(identity([profile, current])) != [identity([profile, current])]*net.world_size:
        raise ValueError('Owners disagree on the complete window inputs')
    binding = current['binding']
    # Complete current weights/Adam are sufficient: constructing the adapter
    # does not need to fetch or replay its earlier warm-start ancestors.
    training = PlannerTraining(net, rows, profile['recipe'],
        adapter_rank=binding['adapter_rank'], max_length=binding['max_length'])
    error = None
    if net.rank == 2:
        try:
            expected = {key: value for key, value in binding.items() if key != 'initial_weights'}
            if training.state.binding != expected:
                raise ValueError('Window state changed its numerical layout or prescription')
            training.state.binding = copy.deepcopy(binding)
            path = Path(checkpoints)/(current['sha256']+'.safetensors')
            if path.is_symlink():
                raise ValueError('A window state must be an owned regular object')
            training.state.restore(checkpoints, current)
        except (ValueError, OSError, KeyError, TypeError) as failure:
            error = type(failure).__name__
    # Finish this known collective boundary before reporting a local storage
    # failure, so the other owners cannot start an unmatched neural operation.
    errors = net.all_owners.exchange(error)
    if any(value is not None for value in errors):
        raise ValueError('Planner window checkpoint is unavailable or invalid')
    training.step = current['step']
    return training


def execute(net, profile, current, rows, checkpoints, home, stop):
    stop = integer(stop, current['step']+1, min(current['step']+16, profile['recipe']['steps']))
    if net.all_owners.exchange(stop) != [stop]*net.world_size:
        raise ValueError('Owners disagree on the prescribed window endpoint')
    training = restore(net, profile, current, rows, checkpoints)
    home = Path(home)
    home.mkdir(parents=True, exist_ok=False)
    states, updates = [copy.deepcopy(current)], []
    for _ in range(current['step'], stop):
        updates.append(training.advance())
        states.append(training.save(home/'checkpoints'))
    result = {'format': window.FORMAT, 'prescription': identity(profile),
              'checkpoints': states, 'updates': updates}
    window.validate(profile, current, result)
    if net.all_owners.exchange(identity(result)) != [identity(result)]*net.world_size:
        raise ValueError('Owners disagree on the complete planner window')
    error = None
    if net.rank == 2:
        try:
            for state in states[1:]:
                retain(home/'checkpoints', checkpoints, state)
        except (ValueError, OSError) as failure:
            error = type(failure).__name__
    if any(value is not None for value in net.all_owners.exchange(error)):
        raise ValueError('Complete planner output states could not be retained')
    save(home/'window.json', result)
    return result


def retain(produced, store, state):
    """Install an immutable numerical object for the next bounded window."""
    source = Path(produced)/(state['sha256']+'.safetensors')
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    target = store/source.name
    if source.is_symlink() or source.stat().st_size != state['bytes'] or sha256(source) != state['sha256']:
        raise ValueError('Produced planner object differs from its commitment')
    temporary = None
    try:
        try:
            os.link(source, target)
        except OSError as error:
            if error.errno == errno.EXDEV:
                with tempfile.NamedTemporaryFile(dir=store, prefix='.planner-', delete=False) as output:
                    temporary = Path(output.name)
                    with source.open('rb') as incoming:
                        shutil.copyfileobj(incoming, output)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            elif error.errno != errno.EEXIST:
                raise
        if target.is_symlink() or target.stat().st_size != state['bytes'] or sha256(target) != state['sha256']:
            raise ValueError('Retained planner object differs from its commitment')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replay(net, profile, current, expected, rows, checkpoints, home):
    window.validate(profile, current, expected)
    actual = execute(net, profile, current, rows, checkpoints, home,
                     current['step']+len(expected['updates']))
    return {'valid': actual == expected, 'record_root': identity(expected),
            'actual_root': identity(actual), 'steps': len(actual['updates'])}, actual


def audit_report(claim, profile, net, rows, checkpoints, home):
    """Execute every claimed update afresh before producing native coverage."""
    from .. import planner_work
    expected = claim['window']
    work = window.validate(profile, claim['input_checkpoint'], expected)
    if (claim['kind'] != planner_work.KIND or claim['prescription'] != profile
            or claim['record_root'] != identity(expected)
            or claim['input_checkpoint'] != expected['checkpoints'][0]
            or claim['output_checkpoint'] != expected['checkpoints'][-1]
            or claim['model_root'] != expected['checkpoints'][-1]['fusion']
            or claim['work_ids'] != work['work_ids'] or type(claim['stages']) is not int
            or claim['stages'] != work['steps']):
        raise ValueError('Audit claim differs from the installed complete planner prescription')
    verdict, actual = replay(net, profile, claim['input_checkpoint'], expected,
                            rows, checkpoints, home)
    stages = [{'stage': index, 'valid': (
        expected['checkpoints'][index:index+2] == actual['checkpoints'][index:index+2]
        and expected['updates'][index] == actual['updates'][index])}
        for index in range(verdict['steps'])]
    report = {'format': planner_work.FORMAT+'/replay', 'claim_id': claim['id'],
              'record_root': claim['record_root'], 'binding': planner_work.binding(claim), 'stages': stages}
    if planner_work.replay_report(claim, report)['valid'] != verdict['valid']:
        raise ValueError('Complete replay and native coverage disagree')
    return report, actual
