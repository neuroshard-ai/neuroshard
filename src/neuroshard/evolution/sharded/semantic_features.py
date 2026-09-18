"""Pinned small semantic encoder on the existing first partition owner."""
from pathlib import Path

from .. import expert_router, semantic_questions
from ..reference_data import identity, sha256


class SemanticFeatures:
    def __init__(self, profile, home, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.profile = profile
        self.root = identity(profile)
        self.home = Path(home)/'semantic-encoders'/self.root
        for name, digest in profile['files'].items():
            path = self.home/name
            if not path.is_file() or path.is_symlink() or sha256(path) != digest:
                raise ValueError('Semantic encoder artifact is unavailable or changed')
        self.tokenizer = AutoTokenizer.from_pretrained(self.home, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(self.home, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, attn_implementation='eager', torch_dtype=torch.float32).eval().to(device)
        if (self.model.config.model_type != 'bert' or self.model.config.hidden_size != semantic_questions.DIMENSIONS
                or sum(value.numel() for value in self.model.parameters()) != semantic_questions.PARAMETERS):
            raise ValueError('Semantic encoder architecture differs from its declared inventory')
        self.model.requires_grad_(False)
        self.versions = [value._version for value in self.model.parameters()]
        self.torch, self.device = torch, device

    def __call__(self, question):
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 2048:
            raise ValueError('Bound the complete semantic routing question')
        encoded = self.tokenizer(question, truncation=False, return_tensors='pt')
        ids = encoded['input_ids'][0].tolist()
        if len(ids) > self.profile['max_tokens']:
            raise ValueError('The complete question exceeds the encoder context')
        with self.torch.inference_mode():
            hidden = self.model(**{key:value.to(self.device) for key,value in encoded.items()}).last_hidden_state[0,0]
            rounded = hidden.float().cpu().mul(4096).round()
            if not bool(self.torch.isfinite(rounded).all()) or bool((rounded.abs() > 2**47).any()):
                raise ValueError('Semantic features exceed the integer quantization bound')
            features = expert_router.normalize(rounded.to(self.torch.int64).tolist())
        if self.versions != [value._version for value in self.model.parameters()]:
            raise ValueError('The immutable semantic encoder changed')
        return {'profile': self.root, 'input_ids': ids, 'features': features}
