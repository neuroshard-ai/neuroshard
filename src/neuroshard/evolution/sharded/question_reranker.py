"""Hash-bound FP32 question-pair model on one declared shard owner."""
import json
from pathlib import Path

from .. import question_reranking
from ..reference_data import identity, sha256


class QuestionReranker:
    def __init__(self, profile, home, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        question_reranking.validate_model(profile)
        self.profile, self.root = profile, identity(profile)
        self.home = Path(home) / 'question-rerankers' / self.root
        for name, spec in profile['files'].items():
            path = self.home / name
            if (not path.is_file() or path.is_symlink() or path.stat().st_size != spec['bytes']
                    or sha256(path) != spec['sha256']):
                raise ValueError('Question reranker bytes are unavailable or changed')
        index = json.loads((self.home / 'model.safetensors.index.json').read_bytes())
        shards = set(index['weight_map'].values())
        if shards != set(profile['files']) - question_reranking.MODEL_FILES:
            raise ValueError('The reranker tensor index differs from its declared shard inventory')
        self.tokenizer = AutoTokenizer.from_pretrained(self.home, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.home,
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
            attn_implementation='eager', torch_dtype=torch.float32).eval().to(device)
        if (self.model.config.model_type != 'xlm-roberta'
                or sum(p.numel() for p in self.model.parameters()) != profile['parameters']):
            raise ValueError('Question reranker architecture differs from its commitment')
        self.model.requires_grad_(False)
        self.versions = [p._version for p in self.model.parameters()]
        self.torch, self.device = torch, device

    def __call__(self, question, candidates):
        if (not isinstance(question, str) or not question.strip() or len(question.encode()) > 2048
                or not isinstance(candidates, list) or not 1 <= len(candidates) <= 32):
            raise ValueError('Bound the complete question reranking request')
        scores = {}
        with self.torch.inference_mode():
            for start in range(0, len(candidates), self.profile['batch_size']):
                batch = candidates[start:start+self.profile['batch_size']]
                encoded = self.tokenizer([[question, row['question']] for row in batch],
                    padding=True, truncation=False, return_tensors='pt')
                if encoded['input_ids'].shape[1] > self.profile['max_tokens']:
                    raise ValueError('Question-family reranking cannot truncate qualifiers')
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits.flatten()
                scaled = logits.float().cpu().mul(self.profile['scale']).round()
                if (not bool(self.torch.isfinite(scaled).all())
                        or bool((scaled < -(2**31)).any()) or bool((scaled > 2**31-1).any())):
                    raise ValueError('Reranker scores exceed their committed integer bound')
                scores.update((row['id'], score) for row, score in
                              zip(batch, scaled.to(self.torch.int32).tolist()))
        if self.versions != [p._version for p in self.model.parameters()]:
            raise ValueError('The frozen question reranker changed')
        return {'profile': self.root,
                'inputs_root': identity({'question': question, 'candidates': candidates}),
                'scores': scores}
