"""Preserve a pinned reranker in bounded public safetensor files."""
import hashlib
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from transformers import AutoModelForSequenceClassification
import torch

REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
SOURCE = Path('/home/ubuntu/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots')/REVISION
DESTINATION = Path('/home/ubuntu/native-expert-live/diagnostics/question-reranker-20260919/model')


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda:source.read(8*1024**2),b''):value.update(part)
    return value.hexdigest()


def main():
    torch.set_num_threads(2)
    DESTINATION.mkdir(parents=True,exist_ok=False)
    model=AutoModelForSequenceClassification.from_pretrained(SOURCE,local_files_only=True,
        trust_remote_code=False,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager')
    model.save_pretrained(DESTINATION,safe_serialization=True,max_shard_size='1GB')
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','README.md'):
        shutil.copyfile(SOURCE/name,DESTINATION/name)
    del model
    index=json.loads((DESTINATION/'model.safetensors.index.json').read_bytes())['weight_map']
    checked=0
    with safe_open(SOURCE/'model.safetensors',framework='pt',device='cpu') as original:
        assert set(original.keys())==set(index)
        for name in sorted(set(index.values())):
            with safe_open(DESTINATION/name,framework='pt',device='cpu') as shard:
                for key in shard.keys():
                    assert index[key]==name and torch.equal(original.get_tensor(key),shard.get_tensor(key))
                    checked+=1
    inventory={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(DESTINATION.iterdir()) if p.is_file()}
    result={'upstream_sha256':digest(SOURCE/'model.safetensors'),'tensor_count':checked,'all_tensors_equal':True,
        'files':inventory,'path':str(DESTINATION)}
    (DESTINATION.parent/'repack-result.json').write_text(json.dumps(result,sort_keys=True)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
