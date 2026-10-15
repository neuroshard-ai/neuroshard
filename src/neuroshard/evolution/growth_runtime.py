"""Real adapters for the growth agent: the protocol trainer on a recipe's model, and a sealed evaluator.

``StateTrainer`` trains and replays segments with ``growth_replay`` on the recipe's model and keeps each
committed segment's state as a content-addressed file (``<state root>.state``). Those files are what other
contributors and checkers fetch to continue or check a job: anything is accepted only if it hashes to the
root committed on chain. ``StateTrainer.publish``/``fetch`` move them through any list of HTTP stores.

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
            model = self.make_model()
            trainable, optimizer = growth_replay.load(model, self.recipe_for_check, payload)
            if growth_replay.state_root(trainable, optimizer) == root:
                local.write_bytes(payload)
                return payload
        raise FileNotFoundError(f'no store holds the state for {root}')

    def state(self, recipe, root):
        model = self.make_model()
        if root is None:
            trainable, optimizer = growth_replay.start(model, recipe)
        else:
            self.recipe_for_check = recipe
            trainable, optimizer = growth_replay.load(model, recipe, self.fetch(root))
        return model, trainable, optimizer

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


class RouteEvaluator:
    """Sealed evaluation of a grown unit on one route against the serving model, on the same sealed cases."""

    def __init__(self, recipe, policy, serving, trainer, minimum, unit_name, descriptor):
        self.recipe, self.policy, self.serving, self.trainer = recipe, policy, serving, trainer
        self.minimum, self.unit_name, self.descriptor = minimum, unit_name, descriptor

    def unit_files(self, final_root):
        """The grown unit's manifest and tensors from the committed final state (pinned by their digests)."""
        from safetensors.torch import save as save_tensors

        _, trainable, _ = self.trainer.state(self.recipe, final_root)
        tensors = {name: value.detach().contiguous() for name, value in sorted(trainable.items())}
        payload = save_tensors(tensors)
        manifest = {'format': 'neuroshard-assistant-experience-checkpoint/1', 'arm': self.recipe['arm'],
                    'trainable_sha256': hashlib.sha256(payload).hexdigest(), 'growth_final_root': final_root,
                    'spec': self.recipe['spec']}
        return json.dumps(manifest, sort_keys=True).encode(), payload

    def candidate_root(self, final_root):
        from neuroshard.assistant import network

        manifest, payload = self.unit_files(final_root)
        files = dict(self.descriptor['files'])
        for name, data in (('manifest.json', manifest), ('trainable.safetensors', payload)):
            files[f'{self.unit_name}/{name}'] = {'asset': f'{self.unit_name}-{final_root[:16]}-{name}',
                                                 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return network.model_root({**self.descriptor, 'files': files})

    def evaluate(self, result, cases):
        from neuroshard.evolution import assistant_general as general
        from neuroshard.evolution.assistant_serving import cached_responder

        kind = self.recipe['route']
        run = general.documents_episode if kind == 'documents' else general.chat_episode
        rows = {}
        for name, model in (('candidate', self.grown(result['final_root'])), ('serving', self.serving())):
            tokenizer = self.serving.tokenizer
            rows[name] = [run(case, cached_responder(model, tokenizer, self.policy), self.policy)['passed']
                          for case in cases['sets'][kind]]
        candidate, serving = sum(rows['candidate']), sum(rows['serving'])
        total = len(rows['candidate'])
        passed = candidate / total >= self.minimum and candidate > serving
        return {'format': 'neuroshard-growth-sealed-report/1', 'round': cases['round'], 'cases_sha256': cases['sha256'],
                'route': kind, 'final_root': result['final_root'], 'candidate': candidate, 'serving': serving,
                'cases': total, 'minimum': self.minimum, 'passed': passed,
                'outcomes': rows, 'model_root': self.candidate_root(result['final_root']) if passed else None}

    def grown(self, final_root):
        model, trainable, _ = self.trainer.state(self.recipe, final_root)
        return model.eval()
