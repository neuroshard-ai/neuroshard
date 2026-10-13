"""The model a growth recipe trains, rebuilt identically on every contributor and validator.

``base`` names how: ``granite-parent`` loads the pinned Granite 4.1 3B parent from the local verified
artifact store (``NEUROSHARD_MODELS``), and ``tiny-granite`` builds the small seeded test model the
protocol tests use. The recipe's ``model_root`` names the served version the unit grows.
"""

import os
from pathlib import Path


def model_factory(recipe):
    base = recipe.get('base', 'granite-parent')
    if base == 'tiny-granite':
        def make():
            import torch
            from transformers import GraniteConfig, GraniteForCausalLM

            torch.manual_seed(recipe.get('base_seed', 0))
            config = GraniteConfig(**recipe['base_config'])
            return GraniteForCausalLM(config).eval()
        return make
    if base == 'granite-parent':
        def make():
            import torch
            from transformers import AutoModelForCausalLM

            from neuroshard.evolution import assistant_growth_run as growth
            from neuroshard.evolution.modular_reference_execution import ROOT, read, verify_artifacts

            inventory = read(ROOT / growth.ARTIFACTS)['models']['baseline']
            directory = Path(os.environ.get('NEUROSHARD_MODELS', ROOT / '.models')) / 'baseline'
            verify_artifacts(directory, inventory, download=False)
            return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True).eval()
        return make
    raise ValueError(f'unknown growth base {base}')
