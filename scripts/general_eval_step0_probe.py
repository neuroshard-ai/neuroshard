"""Every repository file the step-0 worker imports or opens, found by running its logic with scripted models.

    PYTHONPATH=src python scripts/general_eval_step0_probe.py

Prints a sorted JSON list of repository paths. ``test_general_eval_step0`` requires each of them to be in the
execution contract's ``sources``, which is all the canonical host checks out. Model loading itself is not
exercised here; its files are the ones the accepted l9-confirmation worker declared for the same units.
"""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
opened = set()


def audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve() if not isinstance(args[0], int) else None
        if path is not None and ROOT in path.parents:
            opened.add(str(path.relative_to(ROOT)))


sys.addaudithook(audit)


def main():
    import tempfile

    from neuroshard.evolution.general_eval import development_run as step0
    from neuroshard.evolution.general_eval import items as schema, served

    policies, network = served.a4_policies()
    served.route_names(network)
    work = step0.reproduction_work(1)
    split = schema.load_split('development')

    def scripted(model, tokenizer, policy):
        return lambda messages, tools: {'text': 'Answer: 1', 'terminated': True}

    def route(text, opening):
        return 'documents' if 'doc-' in text or 'document' in text.lower() else 'chat'

    for name in ('drafting', 'scheduling'):
        try:  # building the calendar interfaces imports their modules
            step0.served_system(policies, lambda t, o, n=name: n, {}, None, scripted).converse(
                {'documents': []}, ['plan my week'])
        except Exception:  # noqa: BLE001 - only the imports and opens matter here
            pass
    system = step0.served_system(policies, route, {'documents': 'L9', 'chat': 'C1'}, None, scripted)
    with tempfile.TemporaryDirectory() as home:
        step0.execute(system, home, work=work[:1], items=split[:1], split_digest='x', cost={}, deadline=float('inf'))
        system = step0.served_system(policies, route, {'documents': 'L9', 'chat': 'C1'}, None, scripted)
        step0.reproduce(system, work)
        from neuroshard.evolution.general_eval import runner

        runner.run(split, system, split='development')
    modules = {str(Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values())
               if getattr(m, '__file__', None) and ROOT in Path(m.__file__).resolve().parents}
    names = sorted(name for name in opened | modules if '__pycache__' not in name and not name.startswith('/tmp'))
    print(json.dumps(names, indent=1))


if __name__ == '__main__':
    main()
