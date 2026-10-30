"""Smoke-run the general-capability harness end to end on a few development items. Not a baseline.

    PYTHONPATH=src /tmp/tf5venv/bin/python scripts/smoke_general_eval.py [--per-area 3] [--out PATH]

Three systems, all local, CPU only, nothing downloaded and nothing spent:

1. ``oracle``: a scripted system that replies with each item's reference (answer, code, plan) or a hand-made
   correct reply, so every scorer, the sandbox and the tool loop are exercised on a passing path.
2. ``tiny-model``: the test suite's tiny, randomly initialised Granite-architecture model (32 hidden units)
   behind the real ``assistant_serving.cached_responder``, run through ``ModelSystem``. Its replies are noise.
3. ``tiny-served``: the same tiny model behind ``ServedSystem`` with ``a4-general``'s pinned route policies
   (chat-tools/1, documents5) and a fixed keyword stand-in for the router (the real router needs the
   Granite parent). It proves that the served-session path runs; it says nothing about a4-general.

The report states ``baseline_claim: false`` and the per-system numbers are pipeline checks only.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'evolution'))

from neuroshard.evolution.general_eval import items, runner, served  # noqa: E402


def call(name, **arguments):
    return '<tool_call>\n' + json.dumps({'name': name, 'arguments': arguments}) + '\n</tool_call>'


ORACLE_TEXT = {  # hand-written correct replies where an item has no machine reference
    'writing': {
        'writing-dev-001': 'The kitchen tap has been leaking since Monday.\n\nCould you let me know when it will be '
                           'repaired?\n\nI am at home most evenings if that helps.',
        'writing-dev-002': 'The library east wing closes 3 to 21 March for a new roof; children move to the '
                           'community hall and Thursday evenings pause.',
        'writing-dev-003': '- East wing closed 3 to 21 March\n- Children\'s section at the community hall, Bridge '
                           'Street\n- No Thursday evening opening until the work ends'},
    'factual': {'factual-dev-001': 'Answer: Canberra', 'factual-dev-002': 'Answer: W',
                'factual-dev-003': 'Answer: Jane Austen'},
    'tool_use': {'tool_use-dev-001': [call('get_order', order_id='O-5002'), 'Order O-5002 has shipped.'],
                 'tool_use-dev-002': [call('find_customer', query='jordan.lee@example.com'),
                                      call('list_orders', customer_id='C-101'),
                                      call('issue_refund', order_id='O-5001', amount_cents=4599, reason='broken'),
                                      'Refunded 45.99 for order O-5001.'],
                 'tool_use-dev-003': [call('get_order', order_id='O-5002'),
                                      'Order O-5002 has shipped but not been delivered, so it cannot be refunded '
                                      'yet.']},
}


def oracle_replies(item):
    spec = item['scoring']
    if item['area'] == 'reasoning':
        answer = spec['answer']
        return ['Answer: ' + str(answer['value'] if answer['type'] != 'text' else answer['accepted'][0])]
    if item['area'] == 'coding':
        return ['```python\n' + spec['reference'] + '```']
    if item['area'] == 'planning':
        return ['```json\n' + json.dumps(spec['reference']) + '\n```']
    value = ORACLE_TEXT[item['area']][item['id']]
    return value if isinstance(value, list) else [value]


def oracle(chosen):
    queues = {item['id']: iter(oracle_replies(item)) for item in chosen}
    current = {}

    class Oracle(runner.ModelSystem):
        def episode(self, item):
            current['id'] = item['id']
            return super().episode(item)

    return Oracle(lambda messages, tools: next(queues[current['id']]), 'oracle (references)')


def tiny_parts():
    import torch
    from test_assistant_experience import TEMPLATE, tiny_model
    from test_granite_tokenizer import granite_like, load_tiny
    import tempfile

    torch.set_num_threads(4)
    directory = granite_like(Path(tempfile.mkdtemp(prefix='general-eval-smoke-')) / 'granite')
    (directory / 'chat_template.jinja').write_text(TEMPLATE)
    tokenizer = load_tiny(directory)[0]
    return tiny_model(tokenizer, seed=1), tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--per-area', type=int, default=3)
    parser.add_argument('--out', type=Path, default=ROOT / 'config/experiments/general-capability-smoke-report.json')
    args = parser.parse_args()
    from neuroshard.evolution.assistant_serving import cached_responder

    split = items.load_split('development')
    chosen = [item for area in items.AREAS for item in [i for i in split if i['area'] == area][:args.per_area]]
    digest = items.split_digest(split)
    out = {'format': 'neuroshard-general-capability-smoke/1', 'baseline_claim': False,
           'purpose': 'pipeline check only; no number here describes any served or candidate model',
           'split': 'development', 'split_sha256': digest, 'items': [i['id'] for i in chosen], 'systems': {}}

    def record(name, system):
        started = time.monotonic()
        report, transcripts = runner.run(chosen, system, split='development', split_digest=digest,
                                         cost={'host': 'local 4 vCPU development box', 'usd_per_hour': 0.0})
        report['rescored_passed'] = runner.rescore(chosen, transcripts)
        out['systems'][name] = {k: report[k] for k in ('system', 'areas', 'overall', 'cost', 'sandbox_isolation',
                                                       'rescored_passed')}
        print(f'{name}: {report["overall"]} in {time.monotonic() - started:.1f} s', flush=True)

    record('oracle', oracle(chosen))
    model, tokenizer = tiny_parts()
    generation = {'generation': {'max_input_tokens': 8192, 'max_new_tokens': 24}}
    record('tiny-model', runner.ModelSystem(
        lambda messages, tools: cached_responder(model, tokenizer, generation)(messages, tools),
        'tiny random Granite-architecture model (test fixture)', max_steps=2))
    policies, _ = served.a4_policies()
    bounded = {name: {**policy, 'generation': {**policy['generation'], 'max_new_tokens': 24},
                      'limits': {**policy['limits'], 'model_turns_per_user_turn': 2}}
               for name, policy in policies.items()}

    def stand_in_route(text, opening):
        return 'documents' if any(w in opening.lower() for w in ('document', 'insurance', 'bill', 'flight')) else 'chat'

    system = runner.ServedSystem(bounded, stand_in_route, lambda route: cached_responder(model, tokenizer,
                                                                                        bounded[route]),
                                 name='tiny model under a4-general route policies (stand-in router)',
                                 modules=('tiny',))
    record('tiny-served', system)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print('wrote', args.out.relative_to(ROOT))


if __name__ == '__main__':
    main()
