"""Encode only fresh training questions; no final questions or answer bytes."""
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer
from neuroshard.evolution import reference
from neuroshard.evolution.reference_data import identity, save, sha256
from neuroshard.evolution.sharded.semantic_features import SemanticFeatures
from neuroshard.evolution.sharded.router_features import EmbeddingFeatures

home = Path(sys.argv[1])
read = lambda name: json.loads((home/name).read_bytes())
freeze, questions, encoder, profile = [read(name+'.json') for name in ('freeze','questions','encoder','feature-profile')]
assert freeze['driver'] == sha256(__file__)
assert freeze['questions'] == identity(questions) and freeze['encoder'] == identity(encoder)
assert freeze['profile'] == identity(profile)
runtime = reference.configure('cuda', 2)
base = Path('/home/ubuntu/native-expert-live')
semantic = SemanticFeatures(encoder, base/'objects/policies', 'cuda')
coarse = EmbeddingFeatures(base/'interpreter'/(profile['embedding_sha256']+'.safetensors'),
    profile['embedding_sha256'], AutoTokenizer.from_pretrained(base/'seed',local_files_only=True),
    profile['tokenizer_root'],max_tokens=profile['max_tokens'])
assert coarse.profile == profile
rows = []
for row in questions:
    assert set(row) == {'id','question'}
    rows.append({'id':row['id'],'semantic':semantic(row['question'])['features'],'coarse':coarse(row['question'])})
save(home/'features.json',rows)
save(home/'result.json',{'plan':identity(freeze),'features':identity(rows),'count':len(rows),
    'answers_received':False,'finals_received':False,'runtime':runtime})
print(json.dumps({'count':len(rows),'features':identity(rows)}),flush=True)
