"""Inference-only question matching on an already opened cohort.

The model receives question pairs only. Targets remain with the local scorer.
This probe cannot admit a model or count as a prospective learning pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.input_sha256:
        raise ValueError('Question-only probe inputs changed')
    data = json.loads(raw)
    if set(data) != {'questions'} or not 1 <= len(data['questions']) <= 128:
        raise ValueError('Bound the probe and exclude scoring targets')
    for row in data['questions']:
        if set(row) != {'id','question','candidates'} or not 2 <= len(row['candidates']) <= 32:
            raise ValueError('Require only a bounded question and candidate questions')
        if any(set(candidate) != {'id','question'} for candidate in row['candidates']):
            raise ValueError('Candidate answers are forbidden')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from huggingface_hub import snapshot_download
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = Path(snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['*.json','model.safetensors','sentencepiece.bpe.model','README.md']))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for row in data['questions']:
            scores = []
            for start in range(0,len(row['candidates']),8):
                candidates = row['candidates'][start:start+8]
                encoded = tokenizer([[row['question'], candidate['question']] for candidate in candidates],
                    padding=True, truncation=False, return_tensors='pt')
                if encoded['input_ids'].shape[1] > 256:
                    raise ValueError('Do not silently truncate distinguishing words')
                logits = model(**{k:v.to('cuda') for k,v in encoded.items()}).logits.flatten()
                scores.extend(torch.round(logits*1024).to(torch.int32).cpu().tolist())
            ordered = sorted(zip(row['candidates'],scores), key=lambda pair:(-pair[1],pair[0]['id']))
            records.append({'id':row['id'], 'selected':ordered[0][0]['id'],
                'scores':{candidate['id']:score for candidate,score in ordered}})
    result={'model':MODEL,'revision':REVISION,'input_sha256':args.input_sha256,
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.monotonic()-started,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
                 if p.is_file() and p.name != '.gitattributes'},'records':records,
        'training':False,'new_final':False,'native_promotion':False}
    args.output.write_text(json.dumps(result,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','files')}),flush=True)


if __name__ == '__main__':
    main()
