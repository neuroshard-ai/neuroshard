"""CPU elect-sign merge of the frozen parent, incumbent and added tails.

Does not trim. Does not train, decode, or open the original 128-task final.
Writes named safetensors and a manifest whose hashes must match the GPU worker.
"""
import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from neuroshard.evolution import programming_growth_elect as elect
from neuroshard.evolution import programming_growth_ties as ties
from neuroshard.evolution.reference_data import identity, save, sha256


def load_parent(objects, selection):
    owner = selection['owners'][3]
    return elect.load_named(objects, owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--added', type=Path, required=True)
    parser.add_argument('--selection', type=Path,
                        default=Path('config/experiments/programming-expert-selection.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_bytes())
    incumbent_manifest = json.loads((args.incumbent / 'manifest.json').read_bytes())
    added_manifest = json.loads((args.added / 'manifest.json').read_bytes())
    if identity(incumbent_manifest) != elect.INCUMBENT_EXPERT:
        raise ValueError('Incumbent checkpoint is not the leftover fallback tail')
    if identity(added_manifest) != elect.ADDED_EXPERT:
        raise ValueError('Added checkpoint is not the failed merge tail')
    parent = load_parent(args.objects, selection)
    incumbent = elect.load_named(args.incumbent, incumbent_manifest)
    added = elect.load_named(args.added, added_manifest)
    merged = elect.merge_named(parent, incumbent, added)
    args.out.mkdir(parents=True, exist_ok=False)
    files = {}
    conflicts = []
    for name, tensor in merged.items():
        path = args.out / (name + '.safetensors')
        save_file({'weight': tensor.contiguous()}, path)
        files[name] = {'file': path.name, 'sha256': sha256(path)}
        conflicts.append(ties.conflict_fraction(parent[name], incumbent[name], added[name]))
    manifest = {
        'format': elect.FORMAT + '/expert',
        'rule': 'elect-sign-disjoint-mean',
        'trim': False,
        'scale': elect.LAMBDA,
        'parent_owner': 3,
        'incumbent_expert': elect.INCUMBENT_EXPERT,
        'added_expert': elect.ADDED_EXPERT,
        'growth_plan': elect.GROWTH_PLAN,
        'tensors': files,
        'mean_raw_sign_conflict': sum(conflicts) / len(conflicts),
        'train': False,
        'admission_evidence': False,
    }
    save(args.out / 'manifest.json', manifest)
    print(json.dumps({
        'manifest': identity(manifest),
        'tensors': len(files),
        'mean_raw_sign_conflict': manifest['mean_raw_sign_conflict'],
        'trim': False,
        'admission_evidence': False,
    }), flush=True)


if __name__ == '__main__':
    main()
