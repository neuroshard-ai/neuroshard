"""Train a documents unit on realistic demonstrations with the growth protocol trainer, continuing a served unit.

Usage (on a machine with the pinned parent):
  python -m neuroshard.evolution.realistic_train CORPUS OUT --continue-dir DIR [--steps 128]

Experience is the realistic corpus (``realistic_training``), each conversation encoded exactly as serving sees
it; replay is solver-written template demonstrations of every documents family, so earlier behaviour is kept.
Training runs ``growth_replay`` step by step: the same code, schedule and state roots a growth job commits, so
the result could be reproduced and checked by any contributor. The unit is written as a served unit directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time


def trajectory(case, policy):
    """System, user, then each scripted reply with the workspace's tool results: the conversation serving sees."""
    from neuroshard.evolution import assistant_documents as documents

    world = documents.Workspace(case['world'], 2)
    messages = [{'role': 'system', 'content': policy['system_instruction']},
                {'role': 'user', 'content': case['turns'][0]['user']}]
    for text in case['replies']:
        messages.append({'role': 'assistant', 'content': text})
        for call in documents.parse_calls(text, 2):
            messages.append({'role': 'tool', 'content': json.dumps(world.execute(call), ensure_ascii=False,
                                                                     sort_keys=True)})
    return {'case_id': case['id'], 'family': case['family'], 'messages': messages,
            'trainable': [m['role'] == 'assistant' for m in messages]}


def main(argv=None):
    from neuroshard.evolution import granite_shard_execution as runtime

    runtime.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--continue-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--replay-per-family', type=int, default=16)
    parser.add_argument('--policy', type=Path, default=Path('config/experiments/assistant-general-policy-documents5.json'))
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import save as save_tensors
    from neuroshard.evolution import granite_tokenizer

    from neuroshard.evolution import assistant_documents as documents
    from neuroshard.evolution import assistant_experience_train as trainer
    from neuroshard.evolution import growth_planner, growth_replay, growth_runtime
    from neuroshard.evolution.growth_model import model_factory
    from neuroshard.evolution.modular_reference_execution import ROOT

    policy = json.loads(args.policy.read_text())
    tokenizer, _ = granite_tokenizer.load(ROOT / '.models' / 'baseline')  # the served splitting
    tools = documents.tools(2)
    corpus = json.loads(args.corpus.read_text())
    experience = [trainer.encode(tokenizer, trajectory(case, policy), tools) for case in corpus['cases']]
    families = sorted(growth_planner.sealed.templates('documents'))
    replay = growth_planner.recipe_from(families, 'f0' * 32, policy, lambda item: trainer.encode(tokenizer, item, tools),
                                        args.replay_per_family, 'replay')
    manifest = json.loads((args.continue_dir / 'manifest.json').read_text())
    growth_replay.UNIT_STORE = args.continue_dir.parent
    recipe = {'arm': 'addition', 'spec': growth_runtime.growth_spec(args.steps), 'base': 'granite-parent',
              'model_root': 'realistic-offline', 'continue_from': manifest['trainable_sha256'],
              'sources': {'experience': experience, 'replay': replay}}
    print(f"experience {len(experience)} (max {max(len(e['input_ids']) for e in experience)} tokens), "
          f"replay {len(replay)}, steps {args.steps}", flush=True)
    model = model_factory(recipe)()
    trainable, optimizer = growth_replay.start(model, recipe)
    started, roots = time.monotonic(), []
    for first in range(0, args.steps, 8):
        roots += growth_replay.train_segment(model, trainable, optimizer, recipe, first, min(first + 8, args.steps))
        print(f'step {first + 8}/{args.steps} {time.monotonic() - started:.0f}s root {roots[-1][:16]}', flush=True)
    tensors = {name: value.detach().contiguous() for name, value in sorted(trainable.items())}
    payload = save_tensors(tensors)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'trainable.safetensors').write_bytes(payload)
    (args.out / 'manifest.json').write_text(json.dumps({
        'format': 'neuroshard-assistant-experience-checkpoint/1', 'arm': 'addition',
        'trainable_sha256': hashlib.sha256(payload).hexdigest(), 'growth_final_root': roots[-1],
        'continued_from': manifest['trainable_sha256'], 'corpus_sha256': corpus['sha256'],
        'steps': args.steps, 'spec': recipe['spec']}, indent=1, sort_keys=True))
    print('saved', args.out, hashlib.sha256(payload).hexdigest()[:16], flush=True)
    del torch


if __name__ == '__main__':
    main()
