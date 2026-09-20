"""Prepare a pinned coding trial and owned tensors; never launch or train."""
import argparse
import json
from pathlib import Path
import subprocess

from neuroshard.evolution import programming_expert as experiment
from neuroshard.evolution.reference_data import identity, save, sha256
from programming_sandbox import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('config/experiments/programming-expert.json'))
    parser.add_argument('--freeze', action='store_true', help='Bind committed execution source after preparation')
    args = parser.parse_args()
    if args.freeze:
        names = subprocess.check_output(['git', 'ls-files', 'src/neuroshard',
                    'scripts/prepare_programming_expert.py', 'scripts/run_programming_expert.py',
                    'scripts/programming_sandbox.py', 'docs/learning-reference-requirements.txt',
                    'config/experiments/programming-expert.json',
                    'config/experiments/programming-expert-selection.json']).decode().splitlines()
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', *names], check=True)
        plan = json.loads(args.plan.read_bytes())
        selection = json.loads(Path('config/experiments/programming-expert-selection.json').read_bytes())
        freeze = {'format': experiment.FORMAT + '/freeze',
                  'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  'plan': identity(plan), 'selection': identity(selection),
                  'sources': {name: sha256(name) for name in names}}
        path = Path('config/experiments/programming-expert-freeze.json')
        if path.exists() and json.loads(path.read_bytes()) != freeze:
            raise ValueError('Preserve the earlier freeze; this candidate cannot be silently redefined')
        save(path, freeze)
        print('Commit ' + str(path) + ' before training.')
        return
    from transformers import AutoTokenizer, LlamaConfig
    from safetensors import safe_open
    from safetensors.torch import save_file
    from neuroshard.evolution.sharded.model import owner
    from neuroshard.evolution.sharded.portable import configuration, shapes
    plan = json.loads(args.plan.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(args.seed, local_files_only=True)
    inputs = args.home / 'inputs'
    if not inputs.exists():
        prepared = experiment.prepare(plan, args.home / 'mbpp.jsonl', args.home / 'smoltalk.parquet',
                                      tokenizer, inputs, check)
    else:
        prepared = json.loads((inputs / 'prepared.json').read_bytes())
        if prepared['plan'] != identity(plan):
            raise ValueError('Preserve previous inputs; choose a new study directory')
        for role in prepared['roles']:
            experiment.read_role(inputs, prepared, role)
    weights = args.seed / 'model.safetensors'
    if sha256(weights) != plan['model']['sha256']:
        raise ValueError('The instruction seed differs')
    config = LlamaConfig.from_pretrained(args.seed, local_files_only=True)
    manifests = [{'tensors': {}} for _ in range(4)]
    objects = args.home / 'objects'
    objects.mkdir(exist_ok=True)
    with safe_open(weights, framework='pt', device='cpu') as source:
        if set(source.keys()) != set(shapes(config)):
            raise ValueError('Seed tensor coverage differs')
        for name in sorted(source.keys()):
            path = objects / 'tensor.pending'
            value = source.get_tensor(name)
            save_file({'weight': value.contiguous()}, path)
            digest = sha256(path)
            target = objects / (digest + '.safetensors')
            path.replace(target)
            spec = {'file': target.name, 'sha256': digest, 'bytes': target.stat().st_size,
                    'shape': list(value.shape)}
            manifests[owner(name, plan['parent_layout'])]['tensors'][name] = spec
            if name.startswith('model.layers.') and int(name.split('.')[2]) >= plan['split']:
                manifests[3]['tensors'][name] = spec
    for rank, manifest in enumerate(manifests):
        save(args.home / f'owner-{rank}.json', manifest)
    head = {name: manifests[0]['tensors'][name] for name in ('model.embed_tokens.weight', 'model.norm.weight')}
    selection = {'format': experiment.FORMAT + '/selection', 'plan': identity(plan),
                 'prepared': identity(prepared), 'config': configuration(config),
                 'owners': manifests, 'head': head, 'roles': prepared['roles']}
    destination = Path('config/experiments/programming-expert-selection.json')
    save(destination, selection)
    print(json.dumps({'selection': identity(selection), 'counts':
          {k: {x: v[x] for x in ('code', 'general')} for k, v in prepared['roles'].items()},
          'excluded': len(prepared['excluded'])}))


if __name__ == '__main__':
    main()
