"""Train an offline ``workspace-documents/4`` documents unit with the growth protocol trainer, continuing a served unit.

Usage (on an AMX machine with the pinned live runtime and the parent under NEUROSHARD_MODELS):
  python -m neuroshard.evolution.revision_train CORPUS [CORPUS ...] OUT --continue-dir DIR [--steps 128]

This is ``realistic_train`` adapted to ``/4``: experience is every case of the given corpora (``revision_training``
demonstrations, and G2 realistic conversations re-rendered as ``/4``), each encoded with the v4 tools and the /4
policy's system prompt exactly as serving renders the conversation; replay is the template families' solver
demonstrations, the same cases G2 replayed, re-rendered as ``/4`` conversations (``revision_training.template_replay``).
Training runs ``growth_replay`` in segments of 8 steps: the code, schedule and state roots a growth job commits.
The result is an offline development candidate, not a grown or enacted unit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main(argv=None):
    from neuroshard.client import live_runtime

    live_runtime.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('corpora', type=Path, nargs='+')
    parser.add_argument('out', type=Path)
    parser.add_argument('--continue-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--replay-per-family', type=int, default=16)
    parser.add_argument('--policy', type=Path, default=Path('config/experiments/assistant-general-policy-documents7.json'))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--time-steps', type=int, default=0, help='only time this many steps, then stop')
    args = parser.parse_args(argv)

    import os

    import torch
    from safetensors.torch import save as save_tensors
    from transformers import AutoTokenizer

    from neuroshard.evolution import assistant_documents as documents
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import growth_replay, growth_runtime
    from neuroshard.evolution import revision_training as corpus_module
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.evolution.modular_reference_execution import ROOT, identity

    torch.set_num_threads(args.threads)
    policy = json.loads(args.policy.read_text())
    if policy['interface'] != 'workspace-documents/4':
        raise ValueError('the /4 trainer needs a policy that declares workspace-documents/4')
    models = Path(os.environ.get('NEUROSHARD_MODELS', ROOT / '.models'))
    tokenizer = AutoTokenizer.from_pretrained(models / 'baseline', local_files_only=True)
    tools = documents.tools(4)
    cases, corpora = [], []
    for path in args.corpora:
        corpus = json.loads(path.read_text())
        ok, problems = corpus_module.verify(corpus, policy)
        if not ok:
            raise ValueError(f'{path} does not verify: {problems[:5]}')
        corpora.append({'path': str(path), 'sha256': corpus['sha256'], 'cases': len(corpus['cases'])})
        cases += corpus['cases']
    over = [c['id'] for c in cases if not corpus_module.fits(c, policy, tokenizer)]
    if over:
        raise ValueError(f'{len(over)} conversations exceed the token budget under the pinned tokenizer: {over[:5]}')
    experience = [trainer.encode(tokenizer, corpus_module.trajectory(case, policy), tools) for case in cases]
    replay = [trainer.encode(tokenizer, item, tools) for item in corpus_module.template_replay(policy,
                                                                                               args.replay_per_family)]
    manifest = json.loads((args.continue_dir / 'manifest.json').read_text())
    growth_replay.UNIT_STORE = args.continue_dir.parent
    recipe = {'arm': 'addition', 'spec': growth_runtime.growth_spec(args.steps), 'base': 'granite-parent',
              'model_root': 'revision-offline', 'continue_from': manifest['trainable_sha256'],
              'sources': {'experience': experience, 'replay': replay}}
    sources_root = identity([[e['sha256'] for e in experience], [r['sha256'] for r in replay]])
    print(json.dumps({'experience': len(experience), 'max_tokens': max(len(e['input_ids']) for e in experience),
                      'replay': len(replay), 'steps': args.steps, 'sources_root': sources_root,
                      'continue_from': manifest['trainable_sha256'], 'corpora': corpora}), flush=True)
    model = model_factory(recipe)()
    trainable, optimizer = growth_replay.start(model, recipe)
    started, roots = time.monotonic(), []
    if args.time_steps:
        schedule = growth_replay.updates(recipe)
        for index in range(args.time_steps):
            began = time.monotonic()
            loss = growth_replay.step(model, trainable, optimizer, recipe, index, schedule)
            print(f'step {index + 1} loss {loss:.4f} {time.monotonic() - began:.1f}s', flush=True)
        return
    log = args.out.parent / (args.out.name + '-roots.jsonl')
    for first in range(0, args.steps, 8):
        last = min(first + 8, args.steps)
        segment = growth_replay.train_segment(model, trainable, optimizer, recipe, first, last)
        roots += segment
        with log.open('a') as handle:
            handle.write(json.dumps({'first': first, 'last': last, 'roots': segment,
                                     'seconds': round(time.monotonic() - started, 1)}) + '\n')
        print(f'step {last}/{args.steps} {time.monotonic() - started:.0f}s root {roots[-1][:16]}', flush=True)
    tensors = {name: value.detach().contiguous() for name, value in sorted(trainable.items())}
    payload = save_tensors(tensors)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'trainable.safetensors').write_bytes(payload)
    (args.out / 'manifest.json').write_text(json.dumps({
        'format': 'neuroshard-assistant-experience-checkpoint/1', 'arm': 'addition',
        'trainable_sha256': hashlib.sha256(payload).hexdigest(), 'growth_final_root': roots[-1],
        'continued_from': manifest['trainable_sha256'], 'corpora': corpora, 'sources_root': sources_root,
        'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(), 'steps': args.steps,
        'seconds': round(time.monotonic() - started, 1), 'spec': recipe['spec']}, indent=1, sort_keys=True))
    print('saved', args.out, hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
