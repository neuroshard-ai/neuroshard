"""Input-only router features from the existing frozen embedding owner."""
from pathlib import Path

from .. import expert_router
from ..reference_data import identity, sha256
from ..schema import integer, root

FORMAT = 'neuroshard-frozen-router-features-v1'
QUANTIZATION = 4096


class EmbeddingFeatures:
    def __init__(self, path, embedding_sha256, tokenizer, tokenizer_root, *, max_tokens=4096):
        import numpy as np
        import torch
        from safetensors.torch import load_file

        root(embedding_sha256)
        root(tokenizer_root)
        integer(max_tokens, 1, 4096)
        if sha256(Path(path)) != embedding_sha256:
            raise ValueError('Router embedding bytes differ from their commitment')
        tensors = load_file(path, device='cpu')
        if set(tensors) != {'weight'}:
            raise ValueError('The router needs only the immutable input embedding')
        embedding = tensors['weight']
        if (embedding.ndim != 2 or not 1 <= embedding.shape[0] <= 262144
                or not 1 <= embedding.shape[1] <= expert_router.MAX_DIMENSIONS
                or embedding.dtype not in (torch.bfloat16, torch.float32)):
            raise ValueError('Unsupported router embedding shape or dtype')
        self.table = np.empty(tuple(embedding.shape), dtype=np.int16)
        for start in range(0, len(embedding), 4096):
            quantized = embedding[start:start + 4096].float().mul(QUANTIZATION).round()
            if (not bool(torch.isfinite(quantized).all()) or quantized.min().item() < -32768
                    or quantized.max().item() > 32767):
                raise ValueError('Frozen embeddings exceed the declared int16 quantization')
            self.table[start:start + len(quantized)] = quantized.to(torch.int16).numpy()
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.profile = {'format': FORMAT, 'embedding_sha256': embedding_sha256,
                        'tokenizer_root': tokenizer_root, 'quantization': QUANTIZATION,
                        'rounding': 'nearest-ties-even', 'pooling': 'sum-user-token-embeddings',
                        'normalization': 'integer-unit-16384', 'dimensions': self.table.shape[1],
                        'vocabulary': self.table.shape[0], 'max_tokens': max_tokens}
        self.root = identity(self.profile)
        self.numpy = np

    def __call__(self, question):
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 32768:
            raise ValueError('Require bounded raw user text for routing')
        ids = self.tokenizer.encode(question, add_special_tokens=False)
        if not 1 <= len(ids) <= self.max_tokens:
            raise ValueError('Router input exceeds the declared token bound')
        for token in ids:
            integer(token, 0, self.table.shape[0] - 1)
        pooled = self.table[ids].sum(axis=0, dtype=self.numpy.int64).tolist()
        return expert_router.normalize(pooled)
