"""Recompute the local-window experiment from retained operated evidence."""
import math
import statistics

from . import cooperative as group
from . import cooperative_report as checked
from . import local_windows as windows
from . import reference as engine
from . import reference_data as data


def paired_interval(baseline, candidate):
    comparison = checked.paired_accuracy(baseline, candidate)
    differences = [int(after) - int(before) for before, after in zip(baseline, candidate)]
    if len(differences) < 2:
        raise ValueError('A paired interval needs at least two examples')
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return {**comparison, 'standard_error': standard_error,
            'normal_99pct_lower': comparison['accuracy_gain'] - 2.5758293035489004 * standard_error,
            'normal_99pct_upper': comparison['accuracy_gain'] + 2.5758293035489004 * standard_error,
            'interval_scope': 'Approximate paired normal interval on this generated distribution, one training run'}


def learning(prepared, selection, records, evaluations):
    plan = prepared['plan']
    arms = plan['arms']
    if selection['prepared'] != data.identity(prepared) or set(selection['candidates']) != set(arms):
        raise ValueError('Selection must bind every declared final candidate')
    if [row['id'] for row in records] != prepared['roles']['test']['ids']:
        raise ValueError('Final test differs from the committed partition')
    outcomes = {arm: checked.checked_outcomes(evaluations[arm], records, prepared, arm, selection)
                for arm in ('seed', *arms)}
    for arm, report in evaluations.items():
        if [row['id'] for row in report['retention']] != prepared['roles']['retention']['ids']:
            raise ValueError('Retention differs from its committed probes')
        expected_profile = selection['candidates']['single']['profile']
        if group.runtime_profile(report['runtime']) != expected_profile:
            raise ValueError('Evaluation changed the numerical profile')
    comparisons = {
        arm: {**paired_interval(outcomes['seed'], outcomes[arm]),
              'retention': engine.paired_summary(evaluations['seed']['retention'], evaluations[arm]['retention'])}
        for arm in arms
    }
    local_vs_ddp = paired_interval(outcomes['ddp-four'], outcomes['diloco-four'])
    contract = plan['quality_contract']
    primary = comparisons[contract['primary_arm']]
    conditions = {
        'minimum_accuracy_gain_vs_seed': primary['accuracy_gain'] >= contract['minimum_accuracy_gain_vs_seed'],
        'maximum_one_sided_p_vs_seed': primary['exact_one_sided_p'] <= contract['maximum_one_sided_p_vs_seed'],
        'retention_99pct_upper_max': primary['retention']['normal_99pct_upper'] <= contract['retention_99pct_upper_max'],
        'paired_99pct_lower_vs_ddp_min': local_vs_ddp['normal_99pct_lower'] >= contract['paired_99pct_lower_vs_ddp_min'],
    }
    breakdowns = {}
    for field in ('family', 'variant'):
        breakdowns[field] = {}
        for value in sorted({row['task'][field] for row in records}):
            indices = [i for i, row in enumerate(records) if row['task'][field] == value]
            breakdowns[field][str(value)] = {'documents': len(indices),
                **{arm: sum(values[i] for i in indices) for arm, values in outcomes.items()}}
    return {'prepared': data.identity(prepared), 'selection': data.identity(selection),
            'primary_arm': contract['primary_arm'], 'conditions': conditions,
            'quality_contract_passed': all(conditions.values()), 'comparisons_to_seed': comparisons,
            'local_vs_ddp': local_vs_ddp,
            'local_vs_single_descriptive': paired_interval(outcomes['single'], outcomes['diloco-four']),
            'breakdowns': breakdowns, 'scope': contract['scope'], 'serving_approved': False, 'tokens_issued': 0}


def training(prepared, selection, results):
    plan, identity = prepared['plan'], data.identity(prepared)
    recipe = plan['training']
    ids = prepared['roles']['train']['ids']
    schedule = [[ids[index] for index in batch]
                for batch in engine.schedule(len(ids), recipe['steps'], recipe['batch_documents'], recipe['seed'])]
    summaries = {}
    for arm, world in plan['arms'].items():
        ranks = results[arm]
        candidate = selection['candidates'][arm]
        if [row['rank'] for row in ranks] != list(range(world)):
            raise ValueError('Training must include every unique rank in order')
        measured = []
        for rank, result in enumerate(ranks):
            binding = windows.rank_binding(identity, result['runtime'], arm, rank, world)
            if (result['prepared'] != identity or result['arm'] != arm or result['world'] != world
                    or result['binding'] != binding or result['resume_step'] != 0
                    or result['parameter_digest'] != candidate['parameter_digest']
                    or result['group_manifest'] != candidate['group_manifest']
                    or group.runtime_profile(result['runtime']) != candidate['profile']
                    or [row['documents'] for row in result['steps']] != schedule
                    or [row['local_documents'] for row in result['steps']] != [batch[rank::world] for batch in schedule]
                    or [row['step'] for row in result['steps']] != list(range(1, recipe['steps'] + 1))):
                raise ValueError('Training identity, membership, state or assignment changed')
            if rank == 0 and (result['candidate'] != candidate['candidate'] or result['binding'] != candidate['binding']):
                raise ValueError('Rank-zero candidate differs from selection')
            synchronization = result['synchronizations']
            expected = list(range(recipe['local_steps'], recipe['steps'] + 1, recipe['local_steps'])) if arm == 'diloco-four' else []
            if [row['step'] for row in synchronization] != expected:
                raise ValueError('Synchronization schedule differs from the frozen windows')
            if arm == 'diloco-four':
                if ([row['round'] for row in synchronization] != list(range(1, len(expected) + 1))
                        or any(row['delta_payload_bytes_per_rank'] != plan['model']['parameters'] * 4 for row in synchronization)):
                    raise ValueError('Outer rounds or logical payload differ')
            if [row['step'] for row in result['checkpoint_measurements']] != plan['checkpoints'][arm]:
                raise ValueError('Checkpoint measurements are incomplete')
            intervals = [row['seconds'] for row in result['steps']] + [row['seconds'] for row in synchronization]
            intervals += [row['seconds'] for row in result['checkpoint_measurements']] + [result['seconds']]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in intervals):
                raise ValueError('Invalid timing measurement')
            network = {}
            if result['network_start'].keys() != result['network_end'].keys():
                raise ValueError('Host interfaces changed during measurement')
            for name, start in result['network_start'].items():
                network[name] = {direction: result['network_end'][name][direction] - start[direction] for direction in ('rx', 'tx')}
                if any(value < 0 for value in network[name].values()):
                    raise ValueError('Network counters went backwards')
            updates = sum(row['seconds'] for row in result['steps'])
            synchronization_seconds = sum(row['seconds'] for row in synchronization)
            measured.append({'rank': rank, 'update_seconds': updates, 'synchronization_seconds': synchronization_seconds,
                             'active_seconds': updates + synchronization_seconds,
                             'checkpoint_seconds': sum(row['seconds'] for row in result['checkpoint_measurements']),
                             'loop_seconds': result['seconds'], 'interface_bytes': network,
                             'sent_bytes': sum(value['tx'] for name, value in network.items() if name != 'lo')})
        if len({row['outer_digest'] for row in ranks}) != 1:
            raise ValueError('Ranks disagree on the outer optimizer')
        summaries[arm] = {'world': world, 'parameter_digest': candidate['parameter_digest'],
                          'max_rank_active_seconds': max(row['active_seconds'] for row in measured),
                          'max_rank_loop_seconds': max(row['loop_seconds'] for row in measured),
                          'allocated_gpu_loop_seconds': sum(row['loop_seconds'] for row in measured),
                          'total_sent_bytes': sum(row['sent_bytes'] for row in measured), 'ranks': measured}
    local, ddp, single = (summaries[key] for key in ('diloco-four', 'ddp-four', 'single'))
    if ddp['max_rank_active_seconds'] <= 0 or ddp['total_sent_bytes'] <= 0:
        raise ValueError('A measured nonzero DDP baseline is required')
    ratios = {'active_seconds_vs_ddp': local['max_rank_active_seconds'] / ddp['max_rank_active_seconds'],
              'sent_bytes_vs_ddp': local['total_sent_bytes'] / ddp['total_sent_bytes'],
              'active_seconds_vs_single': local['max_rank_active_seconds'] / single['max_rank_active_seconds'],
              'loop_seconds_vs_single': local['max_rank_loop_seconds'] / single['max_rank_loop_seconds']}
    contract = plan['efficiency_contract']
    conditions = {'active_time': ratios['active_seconds_vs_ddp'] <= contract['maximum_active_training_seconds_vs_ddp'],
                  'sent_bytes': ratios['sent_bytes_vs_ddp'] <= contract['maximum_sent_bytes_vs_ddp']}
    return {'arms': summaries, 'ratios': ratios, 'conditions': conditions,
            'efficiency_contract_passed': all(conditions.values()),
            'scope': contract['scope'],
            'timing_scope': 'Maximum over ranks of summed local update and sync durations. Checkpoint and loop durations are separate; initial model loading, process startup and idle allocation require the process and EC2 receipts.',
            'network_scope': 'Whole-host interface counters over each rank training loop, excluding loopback; includes checkpoint acknowledgments and SSH traffic, not isolated NCCL bytes'}


def serving(plan, selection, dev_records, phases):
    recipe = plan['inference']
    expected_digest = selection['candidates'][recipe['serving_arm']]['parameter_digest']
    if set(phases) != {'single', 'four', 'failure'}:
        raise ValueError('Retain all declared serving phases')
    if phases['four']['endpoints'] != phases['failure']['endpoints']:
        raise ValueError('Failure trial changed its provider set')
    if phases['single']['endpoints'] != phases['four']['endpoints'][:1]:
        raise ValueError('Single-provider control must use the same first provider')
    task_ids = [row['id'] for row in dev_records[:recipe['requests']]]
    baseline = {}
    summaries = {}
    conditions = {'all_requests_completed': True, 'identical_successful_output_tokens': True, 'no_cache_hits': True}
    for name in ('single', 'four', 'failure'):
        phase = phases[name]
        expected_count = 1 if name == 'single' else 4
        if (phase['model_digest'] != expected_digest or phase['requests'] != recipe['requests']
                or phase['concurrency'] != recipe['concurrency']
                or len(phase['endpoints']) != expected_count or len(set(phase['endpoints'])) != expected_count
                or len(phase['unavailable']) != (1 if name == 'failure' else 0)
                or [row['task_id'] for row in phase['results']] != task_ids
                or len({row['request_id'] for row in phase['results']}) != recipe['requests']):
            raise ValueError('Serving identity, workload, membership or outage differs from the fixed trial')
        if not math.isfinite(phase['seconds']) or phase['seconds'] <= 0:
            raise ValueError('Invalid serving duration')
        successes = 0
        for row in phase['results']:
            if (not 1 <= len(row['attempts']) <= recipe['maximum_attempts']
                    or any(attempt['endpoint'] not in phase['endpoints'] for attempt in row['attempts'])):
                raise ValueError('Serving exceeded its provider or retry bounds')
            if not row['success']:
                conditions['all_requests_completed'] = False
                continue
            answer = row['answer']
            if (answer['model_digest'] != expected_digest or answer['task_id'] != row['task_id']
                    or answer['request_id'] != row['request_id']):
                raise ValueError('Returned answer changed the requested identity')
            successes += 1
            conditions['no_cache_hits'] &= answer.get('cached') is False
            tokens = answer['generation']['output_ids']
            if row['task_id'] in baseline:
                conditions['identical_successful_output_tokens'] &= tokens == baseline[row['task_id']]
            else:
                baseline[row['task_id']] = tokens
        if phase['successful'] != successes or not math.isclose(phase['requests_per_second'], successes / phase['seconds']):
            raise ValueError('Serving summary differs from its retained requests')
        summaries[name] = {key: value for key, value in phase.items() if key != 'results'}
    one_rate = summaries['single']['requests_per_second']
    return {'arm': recipe['serving_arm'], 'parameter_digest': expected_digest, 'conditions': conditions,
            'serving_fixture_passed': all(conditions.values()),
            'four_throughput_ratio': summaries['four']['requests_per_second'] / one_rate if one_rate else None,
            'failure_retries': sum(len(row['attempts']) - 1 for row in phases['failure']['results']),
            'phases': summaries, 'scope': 'Fixed development tasks, equal client concurrency, one operator; no native payments or general chat approval'}


def recovery(prepared, selection, originals, resumed, comparisons, fault, failed_processes, restarted_processes):
    """Check the consistency of host-side recovery evidence, not remote proofs."""
    plan = prepared['plan']
    contract = plan['recovery']
    arm = contract['arm']
    world = plan['arms'][arm]
    if [row['rank'] for row in comparisons] != list(range(world)):
        raise ValueError('Recovery comparison must include every rank')
    if (fault['rank'] != contract['kill_rank'] or fault['signal'] != 'SIGKILL'
            or fault['last_observed_completed_step'] != contract['kill_after_step']
            or all(code == 0 for code in failed_processes['exit_codes'].values())
            or set(map(int, restarted_processes['exit_codes'])) != set(range(world))
            or any(code != 0 for code in restarted_processes['exit_codes'].values())
            or restarted_processes['started'] < failed_processes['finished']):
        raise ValueError('Recovery did not observe the declared failure and coordinated restart')
    for rank, (original, result, comparison) in enumerate(zip(originals, resumed, comparisons)):
        candidate = selection['candidates'][arm]
        expected_binding = windows.rank_binding(data.identity(prepared), candidate['profile'], arm, rank, world)
        if (result['rank'] != rank or result['world'] != world or result['arm'] != arm
                or result['prepared'] != data.identity(prepared) or result['binding'] != expected_binding
                or comparison['rank'] != rank or comparison['prepared'] != data.identity(prepared)
                or comparison['binding'] != expected_binding or not comparison['contents_exactly_equal']
                or comparison['uninterrupted_receipt'] != original['candidate']['receipt']
                or comparison['recovered_receipt'] != result['candidate']['receipt']
                or result['resume_step'] != contract['restore_step']
                or comparison['resume_step'] != contract['restore_step']
                or result['parameter_digest'] != candidate['parameter_digest']
                or comparison['parameter_digest'] != candidate['parameter_digest']
                or result['outer_digest'] != original['outer_digest']
                or comparison['outer_digest'] != original['outer_digest']):
            raise ValueError('Recovery rank, checkpoint or final state differs from the declared comparison')
        if (comparison['comparisons']['model']['tensor_elements'] != plan['model']['parameters']
                or comparison['comparisons']['outer.pt']['tensor_elements'] != plan['model']['parameters']
                or comparison['comparisons']['optimizer.pt']['tensor_elements'] < 2 * plan['model']['parameters']):
            raise ValueError('Recovery comparison omitted required tensors')
        numerical = lambda rows: [{key: value for key, value in row.items() if key != 'seconds'} for row in rows]
        if numerical(result['steps']) != numerical(original['steps']):
            raise ValueError('Recovery numerical trajectory differs from the uninterrupted run')
    if len(originals) != world or len(resumed) != world:
        raise ValueError('Missing recovery rank results')
    return {'restore_step': contract['restore_step'], 'killed_rank': contract['kill_rank'],
            'last_observed_step_before_kill': fault['last_observed_completed_step'],
            'all_rank_state_and_trajectory_comparisons_passed': True,
            'failed_group_seconds': failed_processes['seconds'], 'restarted_group_seconds': restarted_processes['seconds'],
            'rank_comparisons': comparisons,
            'scope': 'Consistency of retained host-side exact tensor/scalar/RNG comparisons under one operator; no elastic membership, Byzantine recovery or cryptographic execution proof'}
