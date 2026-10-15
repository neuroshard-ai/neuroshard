"""The autonomous growth loop: chain state in, signed transactions out, no person in between.

Anyone can run ``Agent`` with their own keys; it does only what the chain's rules allow and what it can
check itself. One ``tick`` looks at the ledger and takes the next permitted step:

- **contribute**: for every open growth job whose recipe it holds, train the next uncommitted segment from
  the committed state and commit its step roots (``growth_commit``);
- **check**: re-execute a random step of every committed segment in its window; if the committed root is
  wrong, challenge it with a proof bundle (``growth_challenge``), deliver the bundle, then ``growth_prove``;
- **evaluate**: when a job has settled and its ``seal_round`` has passed, fetch and verify that drand beacon,
  derive the sealed evaluation, run the grown unit on it, and publish the result digest (``seal_report``);
- **propose**: if the sealed result clears the recipe's declared minimums and beats the serving version on
  the same cases, propose the grown version to owners (``model_propose``) with that digest as evidence;
- **vote**: as an owner, vote yes on a proposal only if it recomputes the proposal's evidence itself and it
  passes; otherwise vote no. It never votes on judgment.

Every external effect is a ledger transaction or a content-addressed bundle; ``state`` (a local JSON file)
holds only what the agent has done, so a restart resumes. Steps that need a model run through the
injected ``trainer``/``evaluator`` so the loop itself stays testable without one.
"""

import hashlib
import json
from pathlib import Path
import random
import time


class Agent:
    def __init__(self, chain, account, home, *, recipes=None, trainer=None, evaluator=None, owner_keys=None,
                 beacon=None, deliver=None, rng=None, clock=time.time):
        self.chain, self.account, self.home = chain, account, Path(home)
        self.recipes = recipes or {}          # recipe_root -> recipe (published)
        self.trainer, self.evaluator = trainer, evaluator
        self.owner_keys = owner_keys or {}    # log_key -> owner account, for voting
        self.beacon, self.deliver = beacon, deliver
        self.rng, self.clock = rng or random.Random(), clock
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / 'agent-state.json'
        self.memory = json.loads(self.path.read_text()) if self.path.exists() else {'actions': [], 'checked': [],
                                                                                    'evaluated': {}, 'voted': []}

    def remember(self, action):
        self.memory['actions'].append({**action, 'unix': self.clock()})
        partial = self.path.with_suffix('.partial')
        partial.write_text(json.dumps(self.memory, indent=1, sort_keys=True))
        partial.replace(self.path)

    def tick(self):
        """One pass over the ledger; returns the actions taken."""
        state, taken = self.chain.state(), []
        for job_id, job in sorted(state.get('growth', {}).items()):
            # Check before building: never train on top of a segment this agent cannot reproduce.
            taken += self.check(state, job_id, job)
            taken += self.contribute(job_id, self.chain.state().get('growth', {}).get(job_id, job))
        for job_id, result in sorted(state.get('results', {}).items()):
            if result.get('status') == 'grown':
                taken += self.evaluate(job_id, result)
        taken += self.vote(state)
        return taken

    def contribute(self, job_id, job):
        from neuroshard.inference import growth

        recipe = self.recipes.get(job['recipe_root'])
        if recipe is None or self.trainer is None:
            return []
        index = len(job['commits'])
        if index >= growth.segments(job) or any(c['segment'] == index - 1 for c in job['challenges'].values()):
            return []
        start = growth.previous_root(job, index)
        first = index * job['segment_steps']
        try:
            roots = self.trainer.segment(recipe, job_id, start, first, first + growth.segment_length(job, index))
        except (AssertionError, ValueError, FileNotFoundError):
            return []  # the committed start state does not reproduce here, or its bytes are not available yet
        self.chain.submit(self.account, 'growth_commit', job_id=job_id, segment=index, step_roots=roots)
        action = {'kind': 'commit', 'job_id': job_id, 'segment': index}
        self.remember(action)
        return [action]

    def check(self, state, job_id, job):
        """Spot-check one random step of each segment it has not checked; challenge a wrong one."""
        recipe = self.recipes.get(job['recipe_root'])
        if recipe is None or self.trainer is None:
            return []
        out = []
        for index, commit in enumerate(job['commits']):
            # Keyed by the commit's content: a reopened and recommitted segment is a new commit and is checked again.
            key = f"{job_id}/{index}/" + hashlib.sha256(json.dumps(commit['step_roots']).encode()).hexdigest()[:16]
            if key in self.memory['checked'] or commit['contributor'] == self.account.public:
                continue
            step = self.rng.randrange(len(commit['step_roots']))
            correct, bundle = self.trainer.replay(recipe, job, index, step)
            self.memory['checked'].append(key)
            if correct == commit['step_roots'][step]:
                action = {'kind': 'checked', 'job_id': job_id, 'segment': index, 'step': step}
                self.remember(action)
                out.append(action)
                continue
            # A proof starts from the committed root before the disputed step, which must be right: so dispute the
            # first wrong step, found by replaying from the segment's start up to the sampled one.
            for first in range(step + 1):
                correct, bundle = self.trainer.replay(recipe, job, index, first)
                if correct != commit['step_roots'][first]:
                    step = first
                    break
            root = hashlib.sha256(bundle).hexdigest()
            if self.deliver:
                self.deliver(root, bundle)
            envelope = self.chain.submit(self.account, 'growth_challenge', job_id=job_id, segment=index, step=step,
                                         proof_root=root)
            from neuroshard.inference import optimistic as ledger
            action = {'kind': 'challenge', 'job_id': job_id, 'segment': index, 'step': step,
                      'challenge_id': ledger.transaction_id(envelope)}
            self.remember(action)
            out.append(action)
        for action in [a for a in self.memory['actions'] if a['kind'] == 'challenge' and not a.get('proven')]:
            if action['job_id'] == job_id and action['challenge_id'] in job['challenges']:
                opened = job['challenges'][action['challenge_id']]['opened']
                if state['height'] > opened:
                    self.chain.submit(self.account, 'growth_prove', job_id=job_id, challenge_id=action['challenge_id'])
                    action['proven'] = True
                    self.remember({'kind': 'prove', 'job_id': job_id, 'challenge_id': action['challenge_id']})
                    out.append({'kind': 'prove', 'challenge_id': action['challenge_id']})
        return out

    def evaluate(self, job_id, result):
        """Sealed evaluation once the job's seal round exists; propose the unit if it earns it."""
        if job_id in self.memory['evaluated'] or self.evaluator is None or self.owner_keys:
            return []  # owners do not propose what they will judge; any other agent may
        computed = self.sealed_report(result)
        if computed is None:
            return []
        digest, report = computed
        self.memory['evaluated'][job_id] = {'digest': digest, 'passed': report['passed'],
                                            'model_root': report.get('model_root')}
        actions = [{'kind': 'evaluated', 'job_id': job_id, 'digest': digest, 'passed': report['passed']}]
        (self.home / f'sealed-{job_id}.json').write_text(json.dumps(report, indent=1, sort_keys=True))
        if report['passed'] and report.get('model_root'):
            self.chain.submit(self.account, 'model_propose', model_root=report['model_root'], quality_root=digest)
            actions.append({'kind': 'propose', 'job_id': job_id, 'model_root': report['model_root']})
        for action in actions:
            self.remember(action)
        return actions

    def sealed_report(self, result):
        """This agent's own sealed evaluation of a grown result: (digest, report), or None before the seal exists."""
        from neuroshard.evolution import sealed_evaluation as sealed
        from neuroshard.inference import beacon as drand

        if 'seal_round' not in result or self.clock() < drand.time_of(result['seal_round']):
            return None
        round_number, signature, randomness = (self.beacon or drand.fetch)(result['seal_round'])
        if round_number != result['seal_round'] or not drand.verify(round_number, signature, randomness):
            return None
        report = self.evaluator.evaluate(result, sealed.sealed(round_number, signature, randomness))
        return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(), report

    def vote(self, state):
        """Vote as each owner held here: yes only if the proposal's evidence recomputes and passes."""
        if not self.owner_keys:
            return []
        out = []
        for proposal_id, proposal in sorted(state.get('governance', {}).get('proposals', {}).items()):
            if proposal.get('kind') == 'validator' or proposal_id in self.memory['voted']:
                continue
            verdict = self.verdict(proposal, state)
            if verdict is None:
                continue
            for key, owner in self.owner_keys.items():
                if key in proposal['weights'] and key not in proposal['votes']:
                    self.chain.submit(owner, 'model_vote', proposal_id=proposal_id, log_key=key, approve=verdict)
            self.memory['voted'].append(proposal_id)
            action = {'kind': 'vote', 'proposal_id': proposal_id, 'approve': verdict}
            self.remember(action)
            out.append(action)
        return out

    def verdict(self, proposal, state):
        """True if this agent's own sealed evaluation of a grown result reproduces the proposal's evidence digest
        and model root and passes; False if it recomputes and disagrees; None if it cannot evaluate yet.

        The proposer's report is never read: the owner derives the sealed cases from the beacon and runs them."""
        if self.evaluator is None:
            return None
        grown = [r for r in state.get('results', {}).values() if r.get('status') == 'grown']
        undecided = False
        for result in grown:
            computed = self.sealed_report(result)
            if computed is None:
                undecided = True
                continue
            digest, report = computed
            if digest == proposal['quality_root'] and report.get('model_root') == proposal['model_root']:
                return bool(report['passed'])
        return None if undecided else False
