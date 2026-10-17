"""Real adapters for the growth agent: the protocol trainer on a recipe's model, and a sealed evaluator.

``StateTrainer`` trains and replays segments with ``growth_replay`` on the recipe's model and keeps each
committed segment's state as a content-addressed file (``<state root>.state``). Those files are what other
contributors and checkers fetch to continue or check a job: anything is accepted only if it hashes to the
root committed on chain. Agents publish each committed segment's state to the network's stores; ``StateTrainer.fetch`` takes it from any
store, and only if its bytes recompute to the committed root (``growth_replay.bytes_root``).

``RouteEvaluator`` runs a grown unit on a sealed evaluation of one general-route kind (documents or chat)
under its policy, beside the serving model on the same cases, and passes it only if it meets the declared
minimums and solves more cases than the serving model. The report names the candidate version's model root
(the serving descriptor with the grown unit's files pinned in place of the route's current unit), so a
proposal and every owner's recomputation agree on exactly what would be served.
"""

import hashlib
import json
from pathlib import Path
import urllib.request

from neuroshard.evolution import growth_replay


class StateTrainer:
    def __init__(self, store, make_model, stores=()):
        self.store, self.make_model, self.stores = Path(store), make_model, list(stores)
        self.store.mkdir(parents=True, exist_ok=True)

    def path(self, root):
        return self.store / f'{root}.state'

    def fetch(self, root):
        """The committed state bytes for ``root``, local or from a store, accepted only if they hash to it."""
        local = self.path(root)
        if local.exists():
            return local.read_bytes()
        for base in self.stores:
            try:
                with urllib.request.urlopen(f'{base.rstrip("/")}/{root}.state', timeout=120) as response:
                    payload = response.read()
            except OSError:
                continue
            try:
                if growth_replay.bytes_root(payload) != root:
                    continue
            except Exception:  # noqa: BLE001 - bytes that do not even parse are just a bad source
                continue
            local.write_bytes(payload)
            return payload
        raise FileNotFoundError(f'no store holds the state for {root}')

    def state(self, recipe, root):
        model = self.make_model()
        if root is None:
            trainable, optimizer = growth_replay.start(model, recipe)
        else:
            trainable, optimizer = growth_replay.load(model, recipe, self.fetch(root))
        return model, trainable, optimizer

    def state_bytes(self, recipe, root):
        return self.fetch(root)

    def segment(self, recipe, job_id, start, first, last):
        model, trainable, optimizer = self.state(recipe, None if first == 0 else start)
        if growth_replay.state_root(trainable, optimizer) != start:
            raise ValueError('local start state differs from the committed root')
        roots = growth_replay.train_segment(model, trainable, optimizer, recipe, first, last)
        self.path(roots[-1]).write_bytes(growth_replay.save(trainable, optimizer))
        return roots

    def replay(self, recipe, job, index, step):
        from neuroshard.inference import growth

        start = growth.previous_root(job, index)
        model, trainable, optimizer = self.state(recipe, None if index == 0 else start)
        first = index * job['segment_steps']
        if step:
            growth_replay.train_segment(model, trainable, optimizer, recipe, first, first + step)
        bundle = growth_replay.bundle(growth_replay.save(trainable, optimizer), first + step)
        growth_replay.step(model, trainable, optimizer, recipe, first + step)
        return growth_replay.state_root(trainable, optimizer), bundle


def unit_files(state_bytes, recipe, final_root):
    """A grown unit's served files from a committed training state: its manifest and its tensors.

    Built from the state bytes alone (no model is loaded), so every owner derives the identical files."""
    import io

    import torch
    from safetensors.torch import save as save_tensors

    saved = torch.load(io.BytesIO(state_bytes), weights_only=True)['trainable']
    payload = save_tensors({name: value.detach().contiguous() for name, value in sorted(saved.items())})
    manifest = {'format': 'neuroshard-assistant-experience-checkpoint/1', 'arm': recipe['arm'],
                'trainable_sha256': hashlib.sha256(payload).hexdigest(), 'growth_final_root': final_root,
                'spec': recipe['spec']}
    return json.dumps(manifest, sort_keys=True).encode(), payload


def candidate(descriptor, unit_name, final_root, manifest, payload):
    """The serving descriptor with a grown unit's files pinned in place of ``unit_name``'s."""
    files = dict(descriptor['files'])
    for name, data in (('manifest.json', manifest), ('trainable.safetensors', payload)):
        files[f'{unit_name}/{name}'] = {'asset': f'{unit_name}-{final_root[:16]}-{name}',
                                        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return {**descriptor, 'files': files}


def upgrade(descriptor, state, recipes, state_bytes):
    """The version the chain now serves, rebuilt locally after an owner vote enacted a grown unit.

    For each grown result whose recipe this machine holds, derive the candidate descriptor from the committed final
    state (``state_bytes(recipe, root) -> bytes``, verified by the caller against the root) and keep the one whose
    model root is the chain's. Returns (descriptor, {file name: bytes}), or None if no grown result explains it.
    Nobody publishes the new version: every owner rebuilds it from what the chain committed.
    """
    from neuroshard.assistant import network

    target = state['model_root']
    if network.model_root(descriptor) == target:
        return descriptor, {}
    grown = sorted((r for r in state.get('results', {}).values() if r.get('status') == 'grown'),
                   key=lambda r: -r.get('settled_unix', 0))
    for result in grown:
        recipe = recipes.get(result['recipe_root'])
        route = (recipe or {}).get('route')
        if route not in (descriptor.get('routing') or {}):
            continue
        unit = descriptor['routing'][route]['addition']
        try:
            manifest, payload = unit_files(state_bytes(recipe, result['final_root']), recipe, result['final_root'])
        except (OSError, ValueError, KeyError):
            continue
        upgraded = candidate(descriptor, unit, result['final_root'], manifest, payload)
        if network.model_root(upgraded) == target:
            return upgraded, {f'{unit}/manifest.json': manifest, f'{unit}/trainable.safetensors': payload}
    return None


class Evaluators:
    """The agent's evaluator for any grown result: the route evaluator of the result's own recipe."""

    def __init__(self, recipes, serving, trainer, policies, descriptor, minimum=0.9):
        self.recipes, self.serving, self.trainer = recipes, serving, trainer
        self.policies, self.descriptor, self.minimum = policies, descriptor, minimum

    def evaluate(self, result, cases):
        recipe = self.recipes[result['recipe_root']]
        route = recipe['route']
        unit = self.descriptor['routing'][route]['addition']
        def serving():
            return self.serving(route)

        serving.tokenizer = self.serving.tokenizer
        evaluator = RouteEvaluator(recipe, self.policies[route], serving, self.trainer.trainer(recipe), self.minimum,
                                   unit, self.descriptor)
        return evaluator.evaluate(result, cases)


class RouteEvaluator:
    """Sealed evaluation of a grown unit on one route against the serving model, on the same sealed cases."""

    def __init__(self, recipe, policy, serving, trainer, minimum, unit_name, descriptor):
        self.recipe, self.policy, self.serving, self.trainer = recipe, policy, serving, trainer
        self.minimum, self.unit_name, self.descriptor = minimum, unit_name, descriptor

    def unit_files(self, final_root):
        """The grown unit's manifest and tensors from the committed final state (pinned by their digests)."""
        self.trainer.state(self.recipe, final_root)  # fetches and verifies the committed bytes
        return unit_files(self.trainer.path(final_root).read_bytes(), self.recipe, final_root)

    def candidate_root(self, final_root):
        from neuroshard.assistant import network

        return network.model_root(candidate(self.descriptor, self.unit_name, final_root, *self.unit_files(final_root)))

    def evaluate(self, result, cases):
        from neuroshard.evolution import assistant_general as general
        from neuroshard.evolution.assistant_serving import cached_responder

        kind = self.recipe['route']
        run = general.documents_episode if kind == 'documents' else general.chat_episode
        rows = {}
        for name in ('candidate', 'serving'):  # one model at a time: an owner's machine also serves its shard
            model = self.grown(result['final_root']) if name == 'candidate' else self.serving()
            tokenizer = self.serving.tokenizer
            rows[name] = [run(case, cached_responder(model, tokenizer, self.policy), self.policy)['passed']
                          for case in cases['sets'][kind]]
            del model
        candidate, serving = sum(rows['candidate']), sum(rows['serving'])
        total = len(rows['candidate'])
        passed = candidate / total >= self.minimum and candidate > serving
        real = cases['sets'].get(f'{kind}-real', [])
        report = {}
        if real:
            # Non-regression on realistic requests: template families can rise while real use gets worse (L9 did).
            real_rows = {}
            for name in ('candidate', 'serving'):
                model = self.grown(result['final_root']) if name == 'candidate' else self.serving()
                real_rows[name] = [run(case, cached_responder(model, self.serving.tokenizer, self.policy),
                                       self.policy)['passed'] for case in real]
                del model
            report = {'real_candidate': sum(real_rows['candidate']), 'real_serving': sum(real_rows['serving']),
                      'real_cases': len(real), 'real_outcomes': real_rows}
            # Real requests decide; template families are a floor. A unit that only lifts the templates (L9 did, and
            # was worse than no unit on real requests) never passes, and one that helps real use is not blocked by
            # matching, rather than beating, the served version on templates.
            passed = (candidate / total >= self.minimum and report['real_candidate'] > report['real_serving'])
        return {'format': 'neuroshard-growth-sealed-report/1', 'round': cases['round'], 'cases_sha256': cases['sha256'],
                'route': kind, 'final_root': result['final_root'], 'candidate': candidate, 'serving': serving,
                'cases': total, 'minimum': self.minimum, 'passed': passed, **report,
                'outcomes': rows, 'model_root': self.candidate_root(result['final_root']) if passed else None}

    def grown(self, final_root):
        model, trainable, _ = self.trainer.state(self.recipe, final_root)
        return model.eval()


class RoutePlanner:
    """The agent's planner and recipe check for the general routes, on the served model.

    ``serving(route)`` returns the served model of a route (parent plus its current unit); ``units`` maps a route
    to its unit's ``trainable_sha256``, so a recipe continues from the unit it would replace. Called with a
    verified drand randomness, it probes every route on that beacon's sealed set and returns one recipe per
    route whose weakest families are below target. ``check(recipe)`` is the owner's re-verification.
    """

    def __init__(self, serving, make_parent, tokenizer, policies, spec, model_root, units, *, target=0.9, per_family=48,
                 replay_per_family=8, reward_per_step=40_000, segment_steps=8, attempts=3, routes=('documents', 'chat')):
        self.serving, self.make_parent, self.tokenizer = serving, make_parent, tokenizer
        self.policies, self.spec = policies, spec
        self.model_root, self.units, self.target = model_root, units, target
        self.per_family, self.replay_per_family, self.attempts = per_family, replay_per_family, attempts
        self.reward_per_step, self.segment_steps, self.routes = reward_per_step, segment_steps, routes

    def encoder(self, route):
        from neuroshard.evolution import assistant_documents as documents
        from neuroshard.evolution import assistant_experience_train as trainer
        from neuroshard.evolution import assistant_general as general

        policy = self.policies[route]
        if route == 'documents':
            tools = documents.tools(general.version_of(policy))
        else:
            tools = ((general.ChatTools if policy.get('interface') == 'chat-tools/1' else general.Calculator).TOOLS
                     if policy.get('interface') in ('chat-calculator/1', 'chat-tools/1') else None)
        return lambda item: trainer.encode(self.tokenizer, item, tools)

    def responders(self, route):
        from neuroshard.evolution.assistant_serving import cached_responder

        model, policy = self.serving(route), self.policies[route]
        return [lambda case, a=attempt: cached_responder(model, self.tokenizer, policy,
                                                         None if a == 0 else (self.spec['seed'] + a, 0.7))
                for attempt in range(self.attempts)]

    def probe(self, route, randomness):
        from neuroshard.evolution import assistant_general as general
        from neuroshard.evolution import growth_planner as planner
        from neuroshard.evolution import sealed_evaluation as sealed

        cases = {'round': None, 'sha256': hashlib.sha256(randomness.encode()).hexdigest(),
                 'sets': {route: sealed.documents(randomness) if route == 'documents' else sealed.chat(randomness)}}
        run = general.documents_episode if route == 'documents' else general.chat_episode
        greedy = self.responders(route)[0]
        report = planner.probe(lambda case: run(case, greedy(case), self.policies[route])['passed'], cases, route)
        return {**report, 'round': randomness[:16]}

    def __call__(self, randomness):
        from neuroshard.evolution import growth_planner as planner

        out = []
        for route in self.routes:
            report = self.probe(route, randomness)
            encode = self.encoder(route)
            if route == 'documents':
                recipe = planner.recipe(report, randomness, self.policies[route], self.spec, self.model_root, encode,
                                        self.per_family, self.replay_per_family, self.target)
            else:
                recipe = planner.chat_recipe(report, randomness, self.policies[route], self.spec, self.model_root,
                                             encode, self.responders(route), self.per_family, self.replay_per_family,
                                             self.target)
            if recipe is None:
                continue
            recipe['continue_from'] = self.units.get(route)
            steps = self.spec['steps']
            start = growth_replay.state_root(*growth_replay.start(self.make_parent(), recipe))
            out.append({**recipe, 'start_root': start, 'steps': steps, 'segment_steps': self.segment_steps,
                        'reward': steps * self.reward_per_step})
        return out

    def check(self, recipe):
        """True only if every training sequence in the recipe re-derives here from the beacon and its checks."""
        from neuroshard.evolution import growth_planner as planner

        route = recipe.get('route')
        spec = recipe.get('spec') or {}
        # The trainer settings must be this network's; the number of steps is the recipe's own (the reward pays it).
        if (route not in self.routes or recipe.get('model_root') != self.model_root
                or {k: v for k, v in spec.items() if k != 'steps'} != {k: v for k, v in self.spec.items() if k != 'steps'}
                or not 1 <= spec.get('steps', 0) <= 1024 or spec['steps'] != recipe.get('steps')):
            return False
        if recipe.get('continue_from') != self.units.get(route):
            return False
        if recipe.get('kind') == 'realistic-corpus' and route == 'documents':
            from neuroshard.evolution import realistic_training

            def replay_check(sequences):  # the replay must be template demonstrations this owner rebuilds itself
                families = sorted(planner.sealed.templates('documents'))
                per = len(sequences) // len(families) if families else 0
                return per > 0 and sequences == planner.recipe_from(families, 'f0' * 32, self.policies[route],
                                                                    self.encoder(route), per, 'replay')
            return realistic_training.verify(recipe, self.policies[route], self.encoder(route), replay_check)
        verify = planner.verify_documents if route == 'documents' else planner.verify_chat
        return verify(recipe, self.policies[route], self.encoder(route))


def fetch_recipe(stores, root):
    """A recipe from any store, accepted only if it hashes to ``root``."""
    for base in stores:
        try:
            with urllib.request.urlopen(f"{base.rstrip('/')}/{root}.json", timeout=120) as response:
                recipe = json.loads(response.read())
        except (OSError, ValueError):
            continue
        if isinstance(recipe, dict) and growth_replay.recipe_root(recipe, recipe.get('model_root')) == root:
            return recipe
    return None


def follow(descriptor, state, home, stores, progress=print):
    """The descriptor of the version the chain serves, following owner-enacted growth with no published release.

    Versions this machine already rebuilt are kept under ``home/versions/<model root>/``: the descriptor and the
    grown unit's files. A new one is rebuilt from the chain's grown results, the recipes and the committed final
    states in the network's stores (each checked against its hash), and its unit files are written where
    ``fetch_stage`` looks, so no download of them is ever needed.
    """
    from neuroshard.assistant import network

    target = state['model_root']
    if network.model_root(descriptor) == target:
        return descriptor
    versions = Path(home) / 'versions'
    saved = versions / target / 'network.json'
    if saved.exists():
        known = json.loads(saved.read_text())
        if network.model_root(known) == target:
            install(known, versions / target)
            return known
    recipes = {}
    for result in state.get('results', {}).values():
        if result.get('status') == 'grown' and result['recipe_root'] not in recipes:
            recipe = fetch_recipe(stores, result['recipe_root'])
            if recipe is not None:
                recipes[result['recipe_root']] = recipe

    def state_bytes(recipe, root):
        for base in stores:
            try:
                with urllib.request.urlopen(f"{base.rstrip('/')}/{root}.state", timeout=300) as response:
                    payload = response.read()
            except OSError:
                continue
            if growth_replay.bytes_root(payload) == root:
                return payload
        raise FileNotFoundError(root)

    rebuilt = upgrade(descriptor, state, recipes, state_bytes)
    if rebuilt is None:
        raise ValueError('the chain serves a version this machine cannot rebuild from committed growth yet')
    upgraded, files = rebuilt
    (versions / target).mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = network.local_path(versions / target / 'files', name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    saved.write_text(json.dumps(upgraded, indent=2, sort_keys=True))
    install(upgraded, versions / target)
    progress(f'Following the owner-enacted version {target[:16]} (grown unit rebuilt from committed state)')
    return upgraded



def install(descriptor, directory):
    """Make a rebuilt version's grown files a local source for ``network.fetch`` (pinned, so always safe)."""
    from neuroshard.assistant import network

    files = str(Path(directory) / 'files')
    if files not in network.LOCAL_SOURCES:
        network.LOCAL_SOURCES.append(files)


class Served:
    """The served general routes on one machine: the pinned parent plus each route's current unit.

    Planning, sealed evaluation and owner votes need whole route models (serving is sharded across owners, but
    a planner or voter holds the parent locally: ``NEUROSHARD_MODELS``). Units come from the descriptor's pinned
    files; ``units[route]`` is the unit's ``trainable_sha256``, what a recipe continues from.
    """

    def __init__(self, descriptor, home, spec, routes=('documents', 'chat'), make_parent=None, tokenizer=None):
        from neuroshard.assistant import network

        from neuroshard.evolution.growth_model import model_factory

        self.descriptor, self.home, self.spec, self.routes = descriptor, Path(home), spec, routes
        self.make_parent = make_parent or model_factory({'base': 'granite-parent'})
        self.policies = network.version(descriptor)[0].policies
        self.directories, self.units, self.models = {}, {}, {}
        for route in routes:
            unit = descriptor['routing'][route]['addition']
            directory = self.home / 'units' / unit
            for name in ('manifest.json', 'trainable.safetensors'):
                path = directory / name
                if not network.matches(path, descriptor['files'][f'{unit}/{name}']):
                    network.fetch(descriptor, f'{unit}/{name}', path)
            self.directories[route] = directory
            self.units[route] = json.loads((directory / 'manifest.json').read_text())['trainable_sha256']
        growth_replay.UNIT_STORE = self.home / 'units'
        if tokenizer is None:
            import os

            from transformers import AutoTokenizer

            from neuroshard.evolution.modular_reference_execution import ROOT

            tokenizer = AutoTokenizer.from_pretrained(
                Path(os.environ.get('NEUROSHARD_MODELS', ROOT / '.models')) / 'baseline', local_files_only=True)
        self.tokenizer = tokenizer
        self.keep = True  # False: load a route's model per use (an owner's machine also serves its shard)

    def __call__(self, route):
        if not self.keep:
            from neuroshard.evolution import assistant_experience_train as trainer

            model = self.make_parent()
            trainer.load_trainable(model, 'addition', self.spec, self.directories[route])
            return model.eval()
        if route not in self.models:
            from neuroshard.evolution import assistant_experience_train as trainer

            model = self.make_parent()
            trainer.load_trainable(model, 'addition', self.spec, self.directories[route])
            self.models[route] = model.eval()
        return self.models[route]


def growth_spec(steps=64):
    """The training settings every planner and owner uses for a regrown general unit (L9's, with fewer steps)."""
    from neuroshard.evolution import assistant_general_l4 as l4

    return {**l4.spec(l4.declared('assistant-general-l9-gpu')), 'steps': steps}
