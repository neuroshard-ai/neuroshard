# neuroshard/core/model/__init__.py
"""
Model components for NeuroShard.

- dynamic: DynamicNeuroNode, DynamicLayerPool, DynamicNeuroLLM
- llm: NeuroLLMModel, NeuroDecoderLayer
- tokenizer: NeuroTokenizer, get_neuro_tokenizer
- registry: TokenizerConfig
- scaler: ArchitectureScaler
"""

from neuroshard.core.model.dynamic import (
    DynamicNeuroNode,
    DynamicLayerPool,
    DynamicNeuroLLM,
    create_dynamic_node,
)

# Lazy imports to avoid circular dependencies
def get_tokenizer(*args, **kwargs):
    from neuroshard.core.model.tokenizer import get_neuro_tokenizer
    return get_neuro_tokenizer(*args, **kwargs)

__all__ = [
    'DynamicNeuroNode',
    'DynamicLayerPool', 
    'DynamicNeuroLLM',
    'create_dynamic_node',
    'get_tokenizer',
]
