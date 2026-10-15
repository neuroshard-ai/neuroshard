"""Growth on the ledger: units trained by contributors, segment by segment, checkable one step at a time.

A sponsor opens a growth job by escrowing a reward against a declared training recipe: the digest of its
data, spec, seed and schedule (``recipe_root``), the digest of the state training starts from
(``start_root``), and its length in optimizer steps. Contributors train it in order, one segment at a
time. A segment commit lists the digest of the training state after every step in it, chained from the
previous segment's last root, and locks a bond.

Training is deterministic within a pinned execution class (bit-exact across owners, A4), so a wrong step
is provable. Anyone may challenge one step of a commit with a deposit, naming the content address of its
proof bundle so validators can fetch it before the proof is submitted. The proof is that step re-executed
from the committed state before it: validators judge it as they judge serving fraud (``execute``), and a
validator that cannot replay gives no verdict. A proven commit loses its bond, half to the challenger
and half burned, and its segment reopens. A challenge whose proof does not arrive forfeits its deposit.

When every segment has passed its challenge window, the job settles. Contributors are paid per step
committed and keep their bonds, and the job's ``final_root`` is a trained unit owners can vote on
(``model_propose``). Nothing in this module judges a unit's quality.
"""

import copy

FIELDS = {'growth_open': {'recipe_root', 'start_root', 'steps', 'segment_steps', 'reward'},
          'growth_commit': {'job_id', 'segment', 'step_roots'},
          'growth_challenge': {'job_id', 'segment', 'step', 'proof_root'},
          'growth_prove': {'job_id', 'challenge_id'}}
SEAL_MARGIN = 20  # rounds (one minute) after settlement: the trained state is fixed well before the seal exists
PARAMS = {'growth_bond': 2_000_000, 'growth_challenge_deposit': 1_000_000, 'growth_window_blocks': 40,
          'growth_proof_blocks': 40, 'max_growth_jobs': 8, 'max_segment_steps': 64, 'max_growth_steps': 4096}


def params(state):
    return {**PARAMS, **{key: value for key, value in state['params'].items() if key in PARAMS}}


def jobs(state):
    return state.get('growth', {})


def segments(job):
    return -(-job['steps'] // job['segment_steps'])


def segment_length(job, index):
    return min(job['segment_steps'], job['steps'] - index * job['segment_steps'])


def previous_root(job, index):
    """The state a segment starts from: the job's start, or the last step root of the segment before it."""
    return job['start_root'] if index == 0 else job['commits'][index - 1]['step_roots'][-1]


def step_request(state, job_id, segment, step, proof_root):
    """What a growth proof must establish: re-executing ``step`` of ``segment`` from its committed input
    state, under the job's recipe, does not produce the committed output state."""
    job = state['growth'][job_id]
    commit = job['commits'][segment]
    before = previous_root(job, segment) if step == 0 else commit['step_roots'][step - 1]
    return {'kind': 'growth_step', 'chain_id': state['chain_id'], 'job_id': job_id, 'recipe_root': job['recipe_root'],
            'segment': segment, 'step': segment * job['segment_steps'] + step, 'before_root': before,
            'claimed_root': commit['step_roots'][step], 'proof_root': proof_root}


def admit(ledger, previous, body, sender):
    """Cheap checks before state changes; returns the request a proof must establish, or None."""
    kind = body['kind']
    if kind == 'growth_prove':
        job = jobs(previous).get(body['job_id'])
        challenge = job['challenges'].get(ledger.hex_digest(body['challenge_id'])) if job else None
        if challenge is None:
            raise ValueError('No open growth challenge to prove')
        if challenge['opened'] >= previous['height']:
            raise ValueError('A proof must come in a later block than its challenge')
        return step_request(previous, body['job_id'], challenge['segment'], challenge['step'], challenge['proof_root'])
    return None


def apply(ledger, state, envelope, body, sender, debit, execute, request):
    """The state change of one growth transaction (fee and nonce already applied)."""
    kind, settings = body['kind'], params(state)
    growth = state.setdefault('growth', {})
    if kind == 'growth_open':
        steps = ledger.integer(body['steps'], 1, settings['max_growth_steps'])
        segment = ledger.integer(body['segment_steps'], 1, settings['max_segment_steps'])
        reward = ledger.integer(body['reward'], steps)
        if len(growth) >= settings['max_growth_jobs']:
            raise ValueError('Too many open growth jobs')
        debit(reward)
        growth[ledger.transaction_id(envelope)] = {
            'sponsor': sender, 'recipe_root': ledger.hex_digest(body['recipe_root']),
            'start_root': ledger.hex_digest(body['start_root']), 'steps': steps, 'segment_steps': segment,
            'reward': reward, 'opened': state['height'], 'commits': [], 'challenges': {}}
    elif kind == 'growth_commit':
        job = growth.get(body['job_id'])
        if job is None:
            raise ValueError('No such growth job')
        index = ledger.integer(body['segment'], 0, segments(job) - 1)
        if index != len(job['commits']):
            raise ValueError('Segments are committed once each, in order')
        roots = body['step_roots']
        if not isinstance(roots, list) or len(roots) != segment_length(job, index):
            raise ValueError('A segment commit names one state root per step')
        roots = [ledger.hex_digest(root) for root in roots]
        debit(settings['growth_bond'])
        job['commits'].append({'contributor': sender, 'step_roots': roots, 'bond': settings['growth_bond'],
                               'deadline': state['height'] + settings['growth_window_blocks']})
    elif kind == 'growth_challenge':
        job = growth.get(body['job_id'])
        index = body['segment']
        commit = job['commits'][index] if job and type(index) is int and 0 <= index < len(job['commits']) else None
        if commit is None or state['height'] > commit['deadline']:
            raise ValueError('No committed segment in an open window')
        step = ledger.integer(body['step'], 0, len(commit['step_roots']) - 1)
        if any(c['segment'] == index for c in job['challenges'].values()):
            raise ValueError('This segment is already challenged')
        debit(settings['growth_challenge_deposit'])
        job['challenges'][ledger.transaction_id(envelope)] = {
            'challenger': sender, 'segment': index, 'step': step, 'proof_root': ledger.hex_digest(body['proof_root']),
            'deposit': settings['growth_challenge_deposit'],
            'opened': state['height'], 'deadline': state['height'] + settings['growth_proof_blocks']}
        commit['deadline'] = max(commit['deadline'], state['height'] + settings['growth_proof_blocks'])
    elif kind == 'growth_prove':
        if execute is None or execute(state, request) is not True:
            raise ValueError('Growth proof does not verify')
        job = growth[body['job_id']]
        challenge = job['challenges'].pop(body['challenge_id'])
        commit = job['commits'][challenge['segment']]
        # The faulty segment and every later one depend on its roots: all reopen; their bonds are slashed or returned.
        reopened, job['commits'] = job['commits'][challenge['segment']:], job['commits'][:challenge['segment']]
        for offset, later in enumerate(reopened):
            if offset == 0:
                half = later['bond'] // 2
                ledger.account(state, challenge['challenger'])['balance'] += challenge['deposit'] + half
                state['burned'] += later['bond'] - half
            else:
                ledger.account(state, later['contributor'])['balance'] += later['bond']
        for other_id in [i for i, c in job['challenges'].items() if c['segment'] > challenge['segment']]:
            other = job['challenges'].pop(other_id)
            ledger.account(state, other['challenger'])['balance'] += other['deposit']
        state.setdefault('growth_faults', []).append({'job_id': body['job_id'], 'segment': challenge['segment'],
                                                      'step': challenge['step'], 'contributor': commit['contributor'],
                                                      'height': state['height']})
    else:
        raise ValueError('unsupported growth transaction')


def advance(ledger, state, time=None):
    """Block boundary: forfeit lapsed growth challenges and settle complete, unchallenged jobs.

    ``time`` is the block's BFT time (validator-agreed, as CometBFT computes it). A settled job records it
    and the first drand round after it plus ``SEAL_MARGIN``: that round's randomness seals the job's
    evaluation, and it did not exist when the trained state was fixed.
    """
    for job_id in sorted(jobs(state)):
        job = state['growth'][job_id]
        for challenge_id in sorted(job['challenges']):
            if state['height'] > job['challenges'][challenge_id]['deadline']:
                state['burned'] += job['challenges'].pop(challenge_id)['deposit']
        done = len(job['commits']) == segments(job) and not job['challenges'] and all(
            state['height'] > commit['deadline'] for commit in job['commits'])
        if not done:
            continue
        paid = 0
        for commit in job['commits']:
            share = job['reward'] * len(commit['step_roots']) // job['steps']
            ledger.account(state, commit['contributor'])['balance'] += share + commit['bond']
            paid += share
        ledger.account(state, job['sponsor'])['balance'] += job['reward'] - paid
        final = job['commits'][-1]['step_roots'][-1]
        record = {'status': 'grown', 'recipe_root': job['recipe_root'], 'start_root': job['start_root'],
                  'final_root': final, 'steps': job['steps'],
                  'contributors': sorted({c['contributor'] for c in job['commits']})}
        if time is not None:
            from neuroshard.inference import beacon

            record.update(settled_unix=int(time), seal_round=beacon.round_at(int(time)) + SEAL_MARGIN)
        ledger.remember(state, job_id, record)
        del state['growth'][job_id]
    if 'growth' in state and not state['growth']:
        del state['growth']


def escrow(state):
    """Growth funds held by the ledger: rewards, bonds and challenge deposits."""
    return sum(job['reward'] + sum(c['bond'] for c in job['commits'])
               + sum(c['deposit'] for c in job['challenges'].values()) for job in jobs(state).values())


def copy_state(state):
    return copy.deepcopy(state)


def open_proof(state, root):
    """Whether an open growth challenge names this proof bundle (so a proof store may accept it)."""
    return any(c['proof_root'] == root for job in jobs(state).values() for c in job['challenges'].values())
