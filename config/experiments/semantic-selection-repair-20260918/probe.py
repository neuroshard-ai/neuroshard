"""Question-equivalence development probe; no answers or neural training.

Run only on opened development queries. The input contains the nearest eight
intent candidates (their four training questions) and eight earlier-question
negatives. A duplicate-question classifier scores meaning jointly, unlike the
independent sentence embeddings used for retrieval. No evaluation labels enter
the model. This script does not change a serving graph or issue a transaction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


FILES = ('config.json', 'model.safetensors', 'merges.txt', 'vocab.json',
         'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md')
MODEL = 'cross-encoder/quora-distilroberta-base'
REVISION = 'f62e7a4b20b97195c2868e53ec59126df5eac743'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def run(home):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    plan = json.loads((home/'plan.json').read_bytes())
    raw = (home/'questions.json').read_bytes()
    if (plan['inputs_sha256'] != digest(raw)
            or plan['driver_sha256'] != digest(Path(__file__).read_bytes())
            or plan['model'] != MODEL or plan['revision'] != REVISION):
        raise ValueError('The development probe differs from its commitment')
    rows = json.loads(raw)
    for row in rows:
        if set(row) != {'id', 'question', 'previous_intent', 'candidates'}:
            raise ValueError('Questions and candidate identities only')
        if not 1 <= len(row['candidates']) <= 40:
            raise ValueError('Bound the candidate inventory')
        for candidate in row['candidates']:
            if set(candidate) != {'id', 'intent', 'question'}:
                raise ValueError('No candidate answers or evaluation labels')
    model_home = home/'model'
    model_home.mkdir(exist_ok=True)
    inventory = {}
    for name in FILES:
        path = model_home/name
        if not path.exists():
            url = 'https://huggingface.co/'+MODEL+'/resolve/'+REVISION+'/'+name
            with urllib.request.urlopen(url, timeout=120) as response, path.with_suffix('.part').open('wb') as target:
                total = 0
                while block := response.read(1024**2):
                    total += len(block)
                    if total > 512*1024**2:
                        raise ValueError('Model file exceeds the download bound')
                    target.write(block)
            path.with_suffix('.part').replace(path)
        inventory[name] = {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}
    (home/'model-inventory.json').write_text(json.dumps(inventory, sort_keys=True)+'\n')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = plan['device']
    if device not in ('cpu', 'cuda'):
        raise ValueError('Require an explicit supported execution device')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForSequenceClassification.from_pretrained(
        model_home, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation='eager').to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_home, local_files_only=True)
    results = []
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            candidates = row['candidates']
            scores = []
            for start in range(0, len(candidates), 16):
                batch = candidates[start:start+16]
                inputs = tokenizer([row['question']]*len(batch), [r['question'] for r in batch],
                                   return_tensors='pt', padding=True, truncation=False)
                if inputs['input_ids'].shape[1] > 512:
                    raise ValueError('Do not truncate a question to obtain a match')
                logits = model(**inputs.to(device)).logits.flatten()
                scores.extend(round(float(value)*65536) for value in logits)
            ranked = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]['id']))
            best, score = ranked[0]
            intent = best['intent'] if score > 0 else 'parent'
            results.append({'id': row['id'], 'question': row['question'],
                'previous_intent': row['previous_intent'], 'intent': intent,
                'scores': [{'id': candidate['id'], 'intent': candidate['intent'], 'logit_q16': score}
                           for candidate, score in ranked]})
            (home/'progress.json').write_text(json.dumps({'done': len(results), 'count': len(rows)})+'\n')
    result = {'model': MODEL, 'revision': REVISION, 'files': inventory,
        'parameters': sum(p.numel() for p in model.parameters()), 'device': device,
        'torch': torch.__version__, 'seconds': time.monotonic()-started,
        'answers_received': False, 'training': False, 'new_final': False, 'results': results}
    (home/'result.json').write_text(json.dumps(result, sort_keys=True)+'\n')
    print(json.dumps({'queries': len(results), 'seconds': result['seconds'], 'parameters': result['parameters']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', required=True, type=Path)
    run(parser.parse_args().home)
