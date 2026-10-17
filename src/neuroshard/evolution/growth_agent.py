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
                 beacon=None, deliver=None, rng=None, clock=time.time, planner=None, probe_every=None):
        self.chain, self.account, self.home = chain, account, Path(home)
        self.recipes = recipes or {}          # recipe_root -> recipe (published)
        self.trainer, self.evaluator = trainer, evaluator
        self.owner_keys = owner_keys or {}    # log_key -> owner account, for voting
        self.beacon, self.deliver = beacon, deliver
        self.rng, self.clock = rng or random.Random(), clock
        # planner(randomness) -> recipe or None: probes the serving version on a sealed set and, if a family is weak,
        # returns a verified recipe for it. Planning runs every ``probe_every`` seconds at most.
        self.planner, self.probe_every = planner, probe_every
        self.recipe_stores = []  # HTTP stores where proposed recipes are published, by recipe root
        self.home.mkdir(parents=True, exist_ok=True)
        from neuroshard.evolution import growth_replay
        for path in sorted((self.home / 'recipes').glob('*.json')) if (self.home / 'recipes').exists() else []:
            try:  # recipes this agent proposed or fetched before a restart, each checked against its root
                recipe = json.loads(path.read_text())
                if growth_replay.recipe_root(recipe, recipe.get('model_root')) == path.stem:
                    self.recipes.setdefault(path.stem, recipe)
            except (OSError, ValueError, KeyError):
                continue
        self.path = self.home / 'agent-state.json'
        self.memory = json.loads(self.path.read_text()) if self.path.exists() else {'actions': [], 'checked': [],
                                                                                    'evaluated': {}, 'voted': []}
        self.recipe_check = None  # recipe -> bool: an owner's own re-verification of a proposed recipe
        self.contributing = True  # False: check and judge only (an owner keeping its CPUs for serving)
        self.memory.setdefault('reports', {})

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
        if state.get('growth') and self.contributing and self.trainer is not None:
            return taken  # open jobs first: an hours-long evaluation must not keep the network's training waiting
        # Newest results first: an old result's evaluation must not delay the one the network just grew.
        for job_id, result in sorted(state.get('results', {}).items(), key=lambda kv: -kv[1].get('settled_unix', 0)):
            if result.get('status') == 'grown':
                taken += self.evaluate(job_id, result)
                if taken:
                    break
        if self.owner_keys:
            taken += self.prepare(state)
        taken += self.plan(state)
        taken += self.vote(state)
        taken += self.enact(self.chain.state())
        return taken

    def prepare(self, state):
        """As an owner, run the sealed evaluation of each grown result as soon as its seal exists, and keep it.

        A sealed evaluation takes hours on CPU; doing it before any proposal exists means the vote is ready within
        the proposal's window. The cached report is this owner's own computation, never the proposer's."""
        out = []
        if self.evaluator is None:
            return out
        for result in state.get('results', {}).values():
            if result.get('status') != 'grown' or result['final_root'] in self.memory['reports']:
                continue
            if result['recipe_root'] not in self.recipes and self.fetch_recipe(result['recipe_root']) is None:
                continue
            computed = self.sealed_report(result)
            if computed is None:
                continue
            digest, report = computed
            self.memory['reports'][result['final_root']] = {'digest': digest, 'passed': bool(report['passed']),
                                                           'model_root': report.get('model_root')}
            (self.home / f"sealed-{result['final_root'][:16]}.json").write_text(json.dumps(report, sort_keys=True))
            action = {'kind': 'sealed', 'final_root': result['final_root'], 'digest': digest,
                      'passed': bool(report['passed'])}
            self.remember(action)
            out.append(action)
        return out

    def builds_on(self, proposal, state):
        """True if a grown version was trained on the version now served (enacting a stale one would undo a newer
        unit). Decided from this agent's own sealed reports and the grown result's recipe; unknown means no."""
        for result in state.get('results', {}).values():
            report = self.memory.get('reports', {}).get(result.get('final_root'))
            if result.get('status') == 'grown' and report and report['model_root'] == proposal['model_root']:
                recipe = self.recipes.get(result['recipe_root']) or self.fetch_recipe(result['recipe_root'])
                return bool(recipe) and recipe.get('model_root') == state['model_root']
        return False

    def enact(self, state):
        """Enact every proposal the owners have passed: anyone may, so the network never waits on a person."""
        from neuroshard.inference import optimistic as ledger

        governance, out = state.get('governance'), []
        if not governance:
            return out
        for proposal_id, proposal in sorted(governance.get('proposals', {}).items()):
            approving, total = ledger.tally(proposal)
            passed = total and approving * 1_000_000 >= governance['approval_ppm'] * total
            if not passed and state['height'] > proposal['deadline']:
                # A window that closed without passing: close it so the same version can be proposed again.
                try:
                    self.chain.submit(self.account, 'model_enact', proposal_id=proposal_id)
                except ValueError:
                    continue
                action = {'kind': 'closed', 'proposal_id': proposal_id}
                self.remember(action)
                out.append(action)
                continue
            if passed:
                if 'model_root' in proposal and (governance.get('pending_model')
                                                 or not self.builds_on(proposal, state)):
                    continue
                try:
                    self.chain.submit(self.account, 'model_enact', proposal_id=proposal_id)
                except ValueError:
                    continue  # another agent enacted it first
                action = {'kind': 'enact', 'proposal_id': proposal_id, 'what': proposal.get('kind', 'model')}
                self.remember(action)
                out.append(action)
                return out + self.enact(self.chain.state())  # decide the rest against the state this one produced
        return out

    def plan(self, state):
        """Propose a recipe for the serving version's weakest families, when none is already queued or open.

        Nothing here needs a person: a fresh drand round seeds the probe; a weak family becomes a recipe; owners'
        agents vote on it like any proposal; once admitted, the growth pool opens it when usage has paid for it.
        """
        from neuroshard.evolution import growth_replay
        from neuroshard.inference import beacon as drand

        if self.planner is None or 'governance' not in state:
            return []
        last = self.memory.get('planned_unix', 0)
        if self.probe_every is not None and self.clock() - last < self.probe_every:
            return []
        pending = [p for p in state['governance']['proposals'].values() if p.get('kind') == 'recipe']
        if pending or state.get('growth_queue') or state.get('growth'):
            return []  # one growth round at a time: the next probe measures the version that round produces
        round_number = drand.round_at(self.clock()) - 1
        round_number, signature, randomness = (self.beacon or drand.fetch)(round_number)
        if not drand.verify(round_number, signature, randomness):
            return []
        self.memory['planned_unix'] = self.clock()
        planned = self.planner(randomness)
        # A planner returns one recipe or several: one per weak family, each trained as its own job, so a growth round
        # uses as many contributor nodes in parallel as it has weaknesses.
        recipes = [] if planned is None else planned if isinstance(planned, list) else [planned]
        if not recipes:
            self.remember({'kind': 'probed', 'round': round_number, 'weak': []})
            return [{'kind': 'probed', 'round': round_number, 'weak': []}]
        actions = []
        for recipe in recipes:
            root = growth_replay.recipe_root(recipe, recipe['model_root'])
            self.recipes[root] = recipe
            (self.home / 'recipes').mkdir(exist_ok=True)
            (self.home / 'recipes' / f'{root}.json').write_text(json.dumps(recipe, sort_keys=True))
            self.chain.submit(self.account, 'recipe_propose', recipe_root=root, start_root=recipe['start_root'],
                              steps=recipe['steps'], segment_steps=recipe['segment_steps'], reward=recipe['reward'])
            if self.deliver:  # publish once the proposal funds the store's acceptance
                try:
                    self.deliver(root, json.dumps(recipe, sort_keys=True).encode())
                except Exception:  # noqa: BLE001 - a store being down does not stop the proposal; others can serve it
                    pass
            action = {'kind': 'recipe', 'recipe_root': root, 'families': recipe.get('families'), 'round': round_number}
            self.remember(action)
            actions.append(action)
        return actions

    def contribute(self, job_id, job):
        from neuroshard.inference import growth

        if not self.contributing:
            return []
        recipe = self.recipes.get(job['recipe_root']) or self.fetch_recipe(job['recipe_root'])
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
        try:
            self.chain.submit(self.account, 'growth_commit', job_id=job_id, segment=index, step_roots=roots)
        except ValueError:
            return []  # another contributor committed this segment first; continue from theirs next pass
        publish = getattr(self.trainer, 'state_bytes', None)
        if self.deliver and publish:  # once committed, the state is what the next contributor continues from
            try:
                self.deliver(roots[-1], publish(recipe, roots[-1]))
            except Exception:  # noqa: BLE001 - another store, or this agent's next pass, can serve it
                pass
        action = {'kind': 'commit', 'job_id': job_id, 'segment': index}
        self.remember(action)
        return [action]

    def check(self, state, job_id, job):
        """Spot-check one random step of each segment it has not checked; challenge a wrong one."""
        recipe = self.recipes.get(job['recipe_root']) or self.fetch_recipe(job['recipe_root'])
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
        if self.evaluator is None or self.owner_keys:
            return []  # owners do not propose what they will judge; any other agent may
        if job_id in self.memory['evaluated']:
            return self.repropose(job_id)
        computed = self.sealed_report(result)
        if computed is None:
            return []
        digest, report = computed
        self.memory['evaluated'][job_id] = {'digest': digest, 'passed': report['passed'],
                                            'model_root': report.get('model_root')}  # kept to propose again
        actions = [{'kind': 'evaluated', 'job_id': job_id, 'digest': digest, 'passed': report['passed']}]
        (self.home / f'sealed-{job_id}.json').write_text(json.dumps(report, indent=1, sort_keys=True))
        if report['passed'] and report.get('model_root'):
            open_already = any(p.get('model_root') == report['model_root']
                               for p in self.chain.state().get('governance', {}).get('proposals', {}).values())
            if not open_already:
                try:
                    self.chain.submit(self.account, 'model_propose', model_root=report['model_root'], quality_root=digest)
                    actions.append({'kind': 'propose', 'job_id': job_id, 'model_root': report['model_root']})
                except ValueError:
                    pass  # another agent proposed the same version first: the outcome is the same
        for action in actions:
            self.remember(action)
        return actions

    def repropose(self, job_id):
        """Propose a passed version again if its proposal closed unvoted (owners' own evaluation can outlast a vote
        window); never one already serving, still open, or that a later version has replaced."""
        done = self.memory['evaluated'][job_id]
        state = self.chain.state()
        root = done.get('model_root')
        if not done.get('passed') or not root or root == state['model_root']:
            return []
        if any(p.get('model_root') == root for p in state.get('governance', {}).get('proposals', {}).values()):
            return []
        if any(h['model_root'] == root for h in state.get('model_history', [])):
            return []
        recipe_root = state.get('results', {}).get(job_id, {}).get('recipe_root')
        recipe = self.recipes.get(recipe_root) or (self.fetch_recipe(recipe_root) if recipe_root else None)
        if recipe is None or recipe.get('model_root') != state['model_root']:
            return []  # grown on a version no longer served
        try:
            self.chain.submit(self.account, 'model_propose', model_root=root, quality_root=done['digest'])
        except ValueError:
            return []
        action = {'kind': 'propose', 'job_id': job_id, 'model_root': root, 'again': True}
        self.remember(action)
        return [action]

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
            if proposal.get('kind') in ('validator', 'levy') or proposal_id in self.memory['voted']:
                continue
            verdict = (self.recipe_verdict(proposal) if proposal.get('kind') == 'recipe'
                       else self.verdict(proposal, state))
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

    def fetch_recipe(self, root):
        """A proposed recipe from any store, accepted only if it hashes to the root the chain names."""
        import urllib.request

        from neuroshard.evolution import growth_replay

        for base in self.recipe_stores:
            try:
                with urllib.request.urlopen(f"{base.rstrip('/')}/{root}.json", timeout=120) as response:
                    recipe = json.loads(response.read())
            except (OSError, ValueError):
                continue
            if growth_replay.recipe_root(recipe, recipe.get('model_root')) == root:
                self.recipes[root] = recipe
                return recipe
        return None

    def recipe_verdict(self, proposal):
        """Yes to a recipe only if this agent holds it, it hashes to the proposal, its start state is the recipe's
        own, and every demonstration in it re-verifies here; ``recipe_check`` is the owner's own check."""
        from neuroshard.evolution import growth_replay

        recipe = self.recipes.get(proposal['recipe_root']) or self.fetch_recipe(proposal['recipe_root'])
        if recipe is None:
            return None
        if growth_replay.recipe_root(recipe, recipe['model_root']) != proposal['recipe_root']:
            return False
        if recipe.get('start_root') != proposal['start_root'] or recipe.get('steps') != proposal['steps']:
            return False
        check = getattr(self, 'recipe_check', None)
        return bool(check(recipe)) if check else True

    def verdict(self, proposal, state):
        """True if this agent's own sealed evaluation of a grown result reproduces the proposal's evidence digest
        and model root and passes; False if it recomputes and disagrees; None if it cannot evaluate yet.

        The proposer's report is never read: the owner derives the sealed cases from the beacon and runs them."""
        if self.evaluator is None:
            return None
        grown = [r for r in state.get('results', {}).values() if r.get('status') == 'grown']
        undecided = False
        for result in grown:
            cached = self.memory.get('reports', {}).get(result['final_root'])
            if cached is None:
                computed = self.sealed_report(result)
                if computed is None:
                    undecided = True
                    continue
                digest, report = computed
                cached = {'digest': digest, 'passed': bool(report['passed']), 'model_root': report.get('model_root')}
                self.memory.setdefault('reports', {})[result['final_root']] = cached
            if cached['digest'] == proposal['quality_root'] and cached['model_root'] == proposal['model_root']:
                return cached['passed']
        return None if undecided else False
