"""Pass/fail rules of the growth-proof quorum gate (scripts/growth_quorum_gate.py), kept free of I/O so they are tested.

**What a validator should conclude.** For each validator configuration, the expected verdict on each case:
- a pilot-2-rc2 validator holding the parent: honest step → no fraud; arbitrary forgery → fraud;
  legacy 8-thread result → fraud;
- a pilot-1 validator holding the parent: honest → fraud (the thread bug); forgery → fraud;
  legacy 8-thread → no fraud (the bug);
- a validator without the model, or without the proof bundle: no verdict, so it rejects any proposal carrying the
  proof;
- an offline validator: nothing.

**When a penalty must commit.** Exactly when at least ``QUORUM`` of the five equal-weight validators find fraud
(more than two thirds of the voting power).

**How a scenario is judged.**
- **Pass:** every observation matches what is expected, and the decisive event was observed:
  - the penalty committed before the challenge deadline; or
  - the challenge lapsed after its full window and the job then settled.
- **Fail:** an observation contradicts the expectation. For example: a wrong verdict; a penalty that should not
  have committed (or one missing); wrong balances; a recovery that disagrees with its peers.
- **Inconclusive:** the evidence is incomplete. For example: no decisive event before the timeout; an unreadable
  block; a recovery that timed out.

The gate passes only if every scenario passes.
"""

QUORUM = 4  # of 5 equal-weight validators: more than two thirds of the voting power
FEE, BOND, DEPOSIT, OPEN_DEPOSIT, REWARD, START = 1000, 2_000_000, 1_000_000, 1_000_000, 1_000_000, 100_000_000

VERDICT = {  # (flavour, case) -> what a validator holding the model and the bundle concludes
    ('rc2', 'honest'): 'False', ('rc2', 'forged'): 'True', ('rc2', 'legacy8'): 'True',
    ('pilot1', 'honest'): 'True', ('pilot1', 'forged'): 'True', ('pilot1', 'legacy8'): 'False',
}


def expected_verdicts(validators, case, offline=(), missing_store=(), late_store=()):
    """Per validator index: 'True', 'False', 'no-verdict', 'offline', or 'X|no-verdict'.

    The last form is for a validator whose store comes up late: it judges only if the proof is still open when its
    material arrives, so either answer is correct. ``validators`` is a list of {'flavour', 'model'}."""
    out = {}
    for i, v in enumerate(validators):
        if i in offline:
            out[i] = 'offline'
        elif not v['model'] or i in missing_store:
            out[i] = 'no-verdict'
        elif i in late_store:
            out[i] = VERDICT[(v['flavour'], case)] + '|no-verdict'
        else:
            out[i] = VERDICT[(v['flavour'], case)]
    return out


def expect_penalty(verdicts):
    """True if a penalty must commit, False if it must not, None if it depends on timing (late material).

    A penalty commits exactly when enough validators find fraud to accept a block carrying the proof."""
    certain = sum(1 for v in verdicts.values() if v == 'True')
    possible = certain + sum(1 for v in verdicts.values() if v.startswith('True|'))
    if certain >= QUORUM:
        return True
    return False if possible < QUORUM else None


def verdict_matches(expected, observed):
    if '|' in expected:
        return any(verdict_matches(option, observed) for option in expected.split('|'))
    if expected == 'no-verdict':
        return observed.startswith('no-verdict') or observed == 'not-judged'
    if expected == 'offline':
        return observed in ('offline', 'not-judged')
    return observed == expected


def expected_balances(penalty, nonces):
    """Balances and burned supply after the decisive event, from each account's transaction count.

    ``nonces`` = {'sponsor', 'contributor', 'challenger', 'bystander'}: committed transaction counts. Every
    transaction burns its fee. The bystander only sends itself transfers, so it pays fees and nothing else."""
    fees = FEE * sum(nonces.values())
    if penalty:  # the bond is slashed: half to the challenger with its deposit back, half burned; the job reopens
        balances = {'sponsor': START - FEE * nonces['sponsor'] - REWARD - OPEN_DEPOSIT,
                    'contributor': START - FEE * nonces['contributor'] - BOND,
                    'challenger': START - FEE * nonces['challenger'] + BOND // 2}
        burned = fees + (BOND - BOND // 2)
    else:  # the challenge lapses (deposit burned); the one-step job settles: reward and bond paid, deposit returned
        balances = {'sponsor': START - FEE * nonces['sponsor'] - REWARD,
                    'contributor': START - FEE * nonces['contributor'] + REWARD,
                    'challenger': START - FEE * nonces['challenger'] - DEPOSIT}
        burned = fees + DEPOSIT
    balances['bystander'] = START - FEE * nonces['bystander']
    return balances, burned


def judge_scenario(obs):
    """'pass', 'fail' or 'inconclusive', with every reason.

    ``obs`` holds: expected_verdicts, verdicts, outcome ('penalty-committed' | 'lapsed-and-settled' | 'undecided'),
    fault_height, deadline, balances, burned, nonces, blocks_complete, and optionally recovery.
    """
    fail, unsure = [], []
    for i, want in obs['expected_verdicts'].items():
        got = obs['verdicts'].get(i, 'not-judged')
        if not verdict_matches(want, got):
            # A validator that could judge but never did is incomplete evidence, not a contradiction.
            (unsure if got == 'not-judged' else fail).append(f'validator {i}: expected {want}, observed {got}')
    want_penalty = expect_penalty(obs['expected_verdicts'])
    if want_penalty is None:  # timing decides it: the observed verdicts must explain the observed outcome
        want_penalty = sum(1 for v in obs['verdicts'].values() if v == 'True') >= QUORUM
    outcome = obs['outcome']
    if outcome == 'undecided':
        unsure.append('no decisive event (penalty, or lapse then settlement) before the timeout')
    elif (outcome == 'penalty-committed') != want_penalty:
        fail.append(f'outcome {outcome}, but a penalty was {"" if want_penalty else "not "}expected')
    elif outcome == 'penalty-committed' and not obs['fault_height'] <= obs['deadline']:
        fail.append(f'penalty at height {obs["fault_height"]} after the deadline {obs["deadline"]}')
    if outcome != 'undecided':
        balances, burned = expected_balances(outcome == 'penalty-committed', obs['nonces'])
        for name, value in balances.items():
            if obs['balances'].get(name) != value:
                fail.append(f'{name} balance {obs["balances"].get(name)} != expected {value}')
        if obs['burned'] != burned:
            fail.append(f'burned {obs["burned"]} != expected {burned}')
    if not obs['blocks_complete']:
        unsure.append('some blocks of the scenario could not be read')
    recovery = obs.get('recovery')
    if recovery is not None:
        if recovery['result'] == 'disagreement':
            fail.append('recovered validator disagrees with its peers: ' + recovery.get('detail', ''))
        elif recovery['result'] != 'recovered':
            unsure.append('recovery not established: ' + recovery.get('detail', recovery['result']))
    return ('fail' if fail else 'inconclusive' if unsure else 'pass'), fail + unsure


def recovery_decision(samples, address, signed_needed=5, lag_allowed=2):
    """Whether a restarted validator has recovered, from observations of it and of its peers.

    ``samples`` holds:
    - mine: {height, catching_up};
    - peers: [{height}];
    - common: {height, mine: (block_hash, app_hash), peers: [(block_hash, app_hash)]}, read from
      different nodes at the same height;
    - signatures: {height: [validator addresses that signed that height's commit]}, from peers.

    Recovered only if all of these hold:
    - it is not catching up;
    - it is within ``lag_allowed`` of the *current* highest peer;
    - its block and app hash at a common height equal every peer's;
    - its address is in ``signed_needed`` consecutive fresh commits.
    """
    mine, peers, common = samples['mine'], samples['peers'], samples.get('common')
    if common and any(peer != common['mine'] for peer in common['peers']):
        return 'disagreement', f'at height {common["height"]} its block/app hash differs from a peer'
    if mine['catching_up']:
        return 'pending', 'still catching up'
    top = max(p['height'] for p in peers)
    if mine['height'] < top - lag_allowed:
        return 'pending', f'height {mine["height"]} behind peers at {top}'
    if not common or common['height'] > mine['height']:
        return 'pending', 'no common height compared yet'
    heights = sorted(samples.get('signatures', {}))
    streak = best = 0
    for h in heights:
        streak = streak + 1 if address in samples['signatures'][h] else 0
        best = max(best, streak)
    if best < signed_needed:
        return 'pending', f'signed {best} consecutive fresh commits of {signed_needed}'
    return 'recovered', f'at {mine["height"]} (peers {top}), hashes equal at {common["height"]}, signed {best}'


def block_gaps(times):
    """Gaps between consecutive committed blocks: ``times`` is {height: unix}. Returns max and the largest few."""
    heights = sorted(times)
    gaps = [(round(times[b] - times[a], 3), a, b) for a, b in zip(heights, heights[1:]) if b == a + 1]
    gaps.sort(reverse=True)
    return {'max_seconds': gaps[0][0] if gaps else None, 'largest': [{'seconds': s, 'from': a, 'to': b}
                                                                      for s, a, b in gaps[:5]],
            'blocks': len(heights)}
