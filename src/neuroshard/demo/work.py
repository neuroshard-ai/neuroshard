"""Deterministic CPU training tasks using the repository's two-layer NeuroLLM.

The wire format is bounded JSON/base64 float32, never pickle. Each pipeline
stage receives only its own parameters. The verifier independently executes
the complete model and checks the boundary tensors and every gradient.
"""

import base64
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neuroshard.core.model.llm import NeuroLLMConfig, NeuroLLMForCausalLM


CONFIG = dict(vocab_size=256, hidden_dim=32, num_layers=2, num_heads=4,
              num_kv_heads=2, intermediate_dim=64, max_seq_len=64,
              tie_word_embeddings=False, dropout=0.0, attention_dropout=0.0)
BATCH_SIZE, SEQUENCE_LENGTH, LEARNING_RATE = 4, 32, 0.1
REWARD = 1_000_000  # one development NEURO, six decimal places
MAX_REWARDED_TASKS = 1000
LEASE_BLOCKS = 60
MAX_MESSAGE_BYTES = 2_000_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def make_model():
    # Initialization must not depend on other requests or process scheduling.
    with torch.random.fork_rng():
        torch.manual_seed(20260910)
        return NeuroLLMForCausalLM(NeuroLLMConfig(**CONFIG)).cpu()


def encode_tensor(tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    return {"shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii")}


def decode_tensor(value, expected_shape):
    if not isinstance(value, dict) or set(value) != {"shape", "data"}:
        raise ValueError("Invalid tensor envelope")
    if value["shape"] != list(expected_shape):
        raise ValueError("Tensor shape mismatch")
    if any(type(d) is not int for d in value["shape"]):
        raise ValueError("Invalid tensor dimensions")
    size = math.prod(expected_shape) * 4
    if not isinstance(value["data"], str) or len(value["data"]) > 4 * ((size + 2) // 3):
        raise ValueError("Tensor payload too large")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != size:
        raise ValueError("Tensor byte length mismatch")
    tensor = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(expected_shape))
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite tensor")
    return tensor


def encode_weights(model):
    return {name: encode_tensor(p) for name, p in model.named_parameters()}


def load_weights(model, weights):
    expected = dict(model.named_parameters())
    if set(weights) != set(expected):
        raise ValueError("Parameter set mismatch")
    with torch.no_grad():
        for name, p in expected.items():
            p.copy_(decode_tensor(weights[name], p.shape))


def stage_for(name):
    return 0 if name.startswith(("model.embed_tokens.", "model.layers.0.")) else 1


def stage_weights(weights, index):
    return {name: value for name, value in weights.items() if stage_for(name) == index}


def read_data(path):
    data = Path(path).read_bytes()
    if len(data) < 10000:
        raise ValueError("The demo requires the recorded TinyShakespeare corpus")
    return data


def manifest(data):
    from neuroshard.core.model import llm
    sources = [Path(llm.__file__), *sorted(Path(__file__).parent.glob("*.py"))]
    return {"protocol": "neuroshard-demo-v1", "model": CONFIG,
            "dataset_sha256": hashlib.sha256(data).hexdigest(),
            "initial_model_root": digest(encode_weights(make_model())),
            "numerics": f"torch-{torch.__version__}-cpu-fp32-single-thread-no-mkldnn",
            "numpy_version": np.__version__, "architecture": platform.machine(),
            "program_sha256": digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sources}),
            "optimizer": "SGD-no-momentum", "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "sequence_length": SEQUENCE_LENGTH,
            "reward_atoms": REWARD, "max_rewarded_tasks": MAX_REWARDED_TASKS,
            "lease_blocks": LEASE_BLOCKS}


def batch(data, round_number, validation=False):
    split = int(len(data) * 0.95)
    part = data[split:] if validation else data[:split]
    seed = 987654 if validation else 12000 + round_number
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(part) - SEQUENCE_LENGTH, (BATCH_SIZE,), generator=generator)
    return torch.tensor([list(part[int(i):int(i) + SEQUENCE_LENGTH]) for i in starts],
                        dtype=torch.long)


def input_ids(value):
    if (not isinstance(value, list) or len(value) != BATCH_SIZE
            or any(not isinstance(row, list) or len(row) != SEQUENCE_LENGTH for row in value)
            or any(type(v) is not int or not 0 <= v < 256 for row in value for v in row)):
        raise ValueError("Invalid task token batch")
    return torch.tensor(value, dtype=torch.long)


def causal_mask(length):
    return torch.triu(torch.full((length, length), float("-inf")), diagonal=1)


def gradient_receipt(task_id, stage, ids, activation, adjoint, gradients, loss):
    return {"task_id": task_id, "stage": stage,
            "input_root": digest(ids.tolist()) if stage == 0 else digest(encode_tensor(activation)),
            "output_root": digest(encode_tensor(activation if stage == 0 else adjoint)),
            "gradient_root": digest(gradients), "loss_hex": float(loss).hex()}


def apply_gradients(weights, gradients):
    if set(weights) != set(gradients):
        raise ValueError("Gradient parameter set mismatch")
    result = {}
    for name, value in weights.items():
        parameter = decode_tensor(value, value["shape"])
        parameter.add_(decode_tensor(gradients[name], parameter.shape), alpha=-LEARNING_RATE)
        result[name] = encode_tensor(parameter)
    return result


def replay(weights, data, round_number, task_id):
    model = make_model()
    load_weights(model, weights)
    ids = batch(data, round_number)
    captured = {}

    def capture(_module, _inputs, output):
        captured["activation"] = output[0]
        output[0].retain_grad()

    hook = model.model.layers[0].register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    loss = model(ids, labels=ids)["loss"]
    loss.backward()
    hook.remove()
    activation = captured["activation"]
    gradients = {name: encode_tensor(p.grad) for name, p in model.named_parameters()}
    receipts = [gradient_receipt(task_id, i, ids, activation, activation.grad,
                                stage_weights(gradients, i), loss.item()) for i in (0, 1)]
    return {"weights": apply_gradients(weights, gradients), "receipts": receipts,
            "loss": loss.item(), "gradients": gradients}


@torch.no_grad()
def evaluate(weights, data):
    model = make_model()
    load_weights(model, weights)
    ids = batch(data, 0, validation=True)
    return model(ids, labels=ids)["loss"].item()


class Stage(nn.Module):
    """Only one stage's trainable parameters survive construction."""

    def __init__(self, index):
        super().__init__()
        if index not in (0, 1):
            raise ValueError("Stage must be zero or one")
        self.index = index
        full = make_model()
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(index): full.model.layers[index]})
        if index == 0:
            self.model.embed_tokens = full.model.embed_tokens
        else:
            self.model.norm = full.model.norm
            self.lm_head = full.lm_head

    def forward(self, value):
        hidden = self.model.embed_tokens(value) if self.index == 0 else value
        hidden, _ = self.model.layers[str(self.index)](
            hidden, attention_mask=causal_mask(hidden.shape[1]), use_cache=False)
        return hidden if self.index == 0 else self.lm_head(self.model.norm(hidden))

    def compute(self, request):
        load_weights(self, request["weights"])
        ids = input_ids(request["input_ids"])
        self.zero_grad(set_to_none=True)
        if self.index == 0:
            activation = self(ids)
            if request["operation"] == "forward":
                return {"activation": encode_tensor(activation)}
            adjoint = decode_tensor(request["adjoint"], activation.shape)
            activation.backward(adjoint)
            loss = float.fromhex(request["loss_hex"])
            if not math.isfinite(loss):
                raise ValueError("Invalid loss")
        else:
            activation = decode_tensor(request["activation"], (BATCH_SIZE, SEQUENCE_LENGTH, 32))
            activation.requires_grad_(True)
            logits = self(activation)
            loss_tensor = F.cross_entropy(logits[:, :-1].contiguous().view(-1, 256),
                                          ids[:, 1:].contiguous().view(-1))
            loss_tensor.backward()
            loss, adjoint = loss_tensor.item(), activation.grad
        gradients = {name: encode_tensor(p.grad) for name, p in self.named_parameters()}
        return {"gradients": gradients, "adjoint": encode_tensor(adjoint),
                "receipt": gradient_receipt(request["task_id"], self.index, ids,
                                            activation, adjoint, gradients, loss)}


@torch.no_grad()
def infer(weights, prompt, max_tokens=24):
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 64:
        raise ValueError("Prompt must contain 1 to 64 UTF-8 bytes")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 64:
        raise ValueError("Generate 1 to 64 tokens")
    model = make_model()
    load_weights(model, weights)
    tokens = list(prompt.encode())
    for _ in range(max_tokens):
        ids = torch.tensor([tokens[-64:]], dtype=torch.long)
        tokens.append(int(model(ids)["logits"][0, -1].argmax()))
    return {"text": bytes(tokens).decode("utf-8", errors="replace"),
            "token_ids": tokens, "model_root": digest(weights)}
