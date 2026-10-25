#!/usr/bin/env python3
"""Offline: how many tokens per ring pass would a small draft model on the user's device get accepted?

**Method.**
1. Real replies come from the served model (parent plus route unit, whole, as the owners compute it), through
   ``public.Session`` on the documents and chat routes.
2. Each generation's exact input and output token IDs are kept.
3. For each output position, a proposer guesses the next ``k`` tokens. A pass accepts the longest prefix of
   guesses equal to the real output, plus the one token the model itself produces.

**Compared:**
- ``lookup``: today's prompt-lookup guesses (``granite_serving.lookup``);
- ``draft``: a small model's greedy continuation (same tokenizer).

**Reported:** tokens per pass (fewer passes means fewer round trips), and the drafter's own time per proposed
token on this CPU. Nothing touches the network; the replies are what the network would produce, token for token.

    NEUROSHARD_MODELS=... python scripts/eval_draft_speedup.py --network NET.json --home H --draft DIR --out R.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHAT = [
    'Can you help me write a short, friendly email to my landlord asking when the heating will be fixed?',
    'What are three good ways to fall asleep faster?',
    'Summarise the main idea of compound interest in two sentences.',
    'I have eggs, spinach and cheese. What can I cook for dinner in 15 minutes?',
    'Explain the difference between a virus and a bacterium simply.',
    'Give me a packing list for a two-day business trip.',
    'How do I politely decline a meeting invitation?',
    'What is 15% of 240, and how do you work it out?',
    'Write a two-line birthday message for my colleague Sam.',
    'Suggest a 20-minute workout I can do at home without equipment.',
]


def accepted_per_pass(context, output, propose, k):
    """Passes needed to produce ``output`` when each pass verifies ``k`` guesses from ``propose``."""
    passes, i = 0, 0
    while i < len(output):
        guess = propose(context + output[:i], k) if i else []  # the first token of a reply comes from a plain pass
        accepted = 0
        while accepted < len(guess) and i + accepted < len(output) and guess[accepted] == output[i + accepted]:
            accepted += 1
        i += accepted + 1
        passes += 1
    return passes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--network', type=Path)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True, help='directory of the draft model (same tokenizer)')
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-generations', type=int, default=24, help='cap on target generations measured')
    parser.add_argument('--budget-seconds', type=float, default=2400, help='stop measuring after this much wall time')
    args = parser.parse_args()
    import fcntl
    lock = open(str(args.out) + '.lock', 'w')
    try:  # one run per output; a second invocation exits instead of competing for the CPU
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'another run holds {args.out}.lock')
    deadline = time.monotonic() + args.budget_seconds
    progress = Path(str(args.out) + '.jsonl')

    def save(kind, **fields):
        with progress.open('a') as f:
            f.write(json.dumps({'kind': kind, 'unix': round(time.time(), 2), **fields}) + '\n')

    from neuroshard.client import live_runtime
    live_runtime.configure()
    import torch
    from transformers import AutoModelForCausalLM

    import soak
    from neuroshard.assistant import network
    from neuroshard.assistant.public import Session
    from neuroshard.client.assistant_cli import workspace
    from neuroshard.evolution import growth_runtime
    from neuroshard.evolution.assistant_serving import cached_responder
    from neuroshard.evolution.sharded.granite_serving import lookup

    torch.set_num_threads(8)
    descriptor = network.descriptor(args.network)
    chain = network.chain(descriptor, args.home)
    descriptor = growth_runtime.follow(descriptor, chain.state(), args.home, descriptor.get('proof_endpoints', []))
    version = network.version(descriptor)[0]
    served = growth_runtime.Served(descriptor, args.home, version.spec, routes=('documents', 'chat'))
    world, _ = workspace(0, None)

    generations = []
    for route, prompts in (('documents', [q for q, _, _ in soak.QUESTIONS]), ('chat', CHAT)):
        model, policy = served(route), served.policies[route]
        for text in prompts:
            session = Session(policy, world, name=version.name, route_policies=served.policies)
            records = []
            inner = cached_responder(model, served.tokenizer, policy)

            def respond(messages, tools, inner=inner, records=records):
                out = inner(messages, tools)
                records.append(out)
                return out
            session.turn(text, respond, route=route)
            for r in records:
                generations.append({'route': route, 'prompt': text, 'input': r['input_token_ids'], 'output': r['token_ids'],
                                    'target_seconds': r.get('seconds')})
                save('generation', route=route, prompt=text[:60], input_tokens=len(r['input_token_ids']),
                     output_tokens=len(r['token_ids']), seconds=round(r.get('seconds') or 0, 2))
            if len(generations) >= args.max_generations or time.monotonic() > deadline - 600:
                break
        if len(generations) >= args.max_generations or time.monotonic() > deadline - 600:
            del model
            served.models.clear()
            break
        del model
        served.models.clear()

    from transformers import AutoTokenizer
    drafter_tokenizer = AutoTokenizer.from_pretrained(args.draft, local_files_only=True)
    if drafter_tokenizer.get_vocab() != served.tokenizer.get_vocab() or \
            drafter_tokenizer.all_special_ids != served.tokenizer.all_special_ids:
        raise SystemExit('the drafter tokenizer differs from the served one: token IDs would not mean the same')
    save('tokenizer', identical=True, vocab=len(drafter_tokenizer))
    import resource
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16, local_files_only=True).eval()
    timing = {'proposed': 0, 'seconds': 0.0, 'prefill_seconds': 0.0, 'prefill_tokens': 0}
    state = {'ids': [], 'cache': None}

    def forward(ids):
        out = draft(input_ids=torch.tensor([ids]), past_key_values=state['cache'], use_cache=True)
        state['cache'] = out.past_key_values
        return out.logits[0, -1]

    def drafted(context, k):
        """Greedy k-token proposal with the drafter's KV cache kept across passes (as a client would keep it)."""
        with torch.inference_mode():
            began = time.monotonic()
            common = 0
            for a, b in zip(state['ids'], context):
                if a != b:
                    break
                common += 1
            if common < len(state['ids']) and state['cache'] is not None:
                try:  # attention caches crop; a hybrid model's recurrent (Mamba) state cannot, so it re-prefills
                    state['cache'].crop(common)
                    state['ids'] = state['ids'][:common]
                except Exception:  # noqa: BLE001 - measured: the re-prefill counts as drafter time
                    state['cache'], state['ids'] = None, []
                    timing['recomputes'] = timing.get('recomputes', 0) + 1
            if common == 0:
                state['cache'], state['ids'] = None, []
            fresh = context[len(state['ids']):]
            if len(fresh) > 1:
                p0 = time.monotonic()
                forward(fresh[:-1])
                timing['prefill_seconds'] += time.monotonic() - p0
                timing['prefill_tokens'] += len(fresh) - 1
                state['ids'] += fresh[:-1]
            logits = forward([context[-1]])
            state['ids'].append(context[-1])
            out = []
            for _ in range(k):
                token = int(logits.argmax())
                out.append(token)
                logits = forward([token])
                state['ids'].append(token)
            timing['seconds'] += time.monotonic() - began
            timing['proposed'] += len(out)
        return out

    save('drafter-loaded', rss_mb_increase=round((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before) / 1024))
    rows = []
    for g in generations:
        if time.monotonic() > deadline:
            save('budget-exhausted', measured=len(rows), of=len(generations))
            break
        lk = accepted_per_pass(g['input'], g['output'], lambda c, k: lookup(c, k), args.k)
        before = dict(timing)
        state['ids'], state['cache'] = [], None  # each generation starts the drafter fresh, as a new turn would
        dr = accepted_per_pass(g['input'], g['output'], drafted, args.k)
        rows.append({'route': g['route'], 'prompt': g['prompt'][:60], 'tokens': len(g['output']),
                     'passes_plain': len(g['output']), 'passes_lookup': lk, 'passes_draft': dr,
                     'draft_seconds': round(timing['seconds'] - before['seconds'], 2),
                     'draft_prefill_seconds': round(timing['prefill_seconds'] - before['prefill_seconds'], 2)})
        save('row', **rows[-1])
        print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for route in ('documents', 'chat', 'all'):
        part = [r for r in rows if route == 'all' or r['route'] == route]
        tokens = sum(r['tokens'] for r in part)
        summary[route] = {'generations': len(part), 'tokens': tokens,
                          **{f'tokens_per_pass_{m}': round(tokens / max(1, sum(r[f'passes_{m}'] for r in part)), 2)
                             for m in ('plain', 'lookup', 'draft')}}
    decode_rows = list(rows)
    summary['draft_seconds_total'] = round(sum(r['draft_seconds'] for r in decode_rows), 1)
    summary['draft_prefill_seconds_total'] = round(sum(r['draft_prefill_seconds'] for r in decode_rows), 1)
    summary['draft_seconds_per_proposed_token'] = round(
        (timing['seconds'] - timing['prefill_seconds']) / max(1, timing['proposed']), 4)
    summary['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    summary['drafter_cache_recomputes'] = timing.get('recomputes', 0)
    summary['drafter_prefill_tokens'] = timing['prefill_tokens']
    # Break-even, per route: the drafter pays off only if the ring time saved by fewer passes exceeds its own time.
    # pass_seconds (one ring pass on the live fleet, measured from owners' timing records) is an input, not measured here.
    for route in ('documents', 'chat', 'all'):
        part = [r for r in decode_rows if route == 'all' or r['route'] == route]
        if part:
            saved = sum(r['passes_lookup'] - r['passes_draft'] for r in part)
            summary[route]['passes_saved_vs_lookup'] = saved
            summary[route]['draft_seconds'] = round(sum(r['draft_seconds'] for r in part), 1)
            summary[route]['break_even_pass_seconds'] = round(sum(r['draft_seconds'] for r in part) / saved, 3) if saved > 0 else None
    summary['note'] = ('The drafter keeps its KV cache across passes (prefill once per generation, one step per proposed '
                       'token). Net gain needs: saved passes x live pass time > drafter time, and the larger verify '
                       'batch must not raise the live pass time; neither is measured here.')
    args.out.write_text(json.dumps({'summary': summary, 'rows': rows, 'model_root': network.model_root(descriptor)}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
