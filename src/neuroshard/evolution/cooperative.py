"""Trusted compute-group research primitives; not a native execution profile."""
import contextlib
import hashlib
import time

from . import reference as engine


def runtime_profile(runtime):
    """Record host identity separately; numerical settings must still match."""
    return {key: value for key, value in runtime.items() if key != "host"}


def rank_records(records, rank, world):
    if world < 1 or not 0 <= rank < world or len(records) % world:
        raise ValueError("The global document batch must split evenly across ranks")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Repeated document inside a global batch")
    return records[rank::world]


def step(model, optimizer, records, device, recipe, index, rank=0, world=1, check=lambda: None):
    import torch
    import torch.distributed as dist
    local = rank_records(records, rank, world)
    total_targets = sum(record["targets"] for record in records)
    weights=[record.get("loss_weight",1) for record in records]
    if any(type(weight) is not int or not 1<=weight<=16 for weight in weights):
        raise ValueError("Invalid declared target weight")
    weighted_targets = sum(record["targets"] * weight for record,weight in zip(records,weights))
    if not local or total_targets <= 0:
        raise ValueError("Empty global or local training batch")
    if world > 1 and (not dist.is_initialized() or dist.get_world_size() != world or dist.get_rank() != rank):
        raise ValueError("Rank declaration differs from initialized process group")
    rate = engine.learning_rate(recipe, index)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    started = time.monotonic()
    for position, record in enumerate(local):
        check()
        context = model.no_sync() if world > 1 and position + 1 < len(local) else contextlib.nullcontext()
        # The context includes forward as required by DDP. Its reduction
        # averages ranks, so multiply local summed loss by world_size.
        with context:
            loss = engine.response_loss(model, record, device)
            weight=record.get("loss_weight",1)
            loss_sum += float(loss.detach()) * weight
            (loss * world * weight / weighted_targets).backward()
    if world > 1:
        value = torch.tensor(loss_sum, dtype=torch.float64, device=device)
        dist.all_reduce(value)
        loss_sum = float(value)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["clip_norm"], error_if_nonfinite=True)
    check()
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return {"step": index + 1, "loss": loss_sum / weighted_targets, "targets": total_targets,
            "weighted_targets": weighted_targets,
            "learning_rate": rate, "gradient_norm": float(norm),
            "documents": [record["id"] for record in records],
            "seconds": time.monotonic() - started,
            "local_documents": [record["id"] for record in local]}


def parameter_digest(model):
    """Stream one CPU tensor at a time, avoiding a full extra model copy."""
    import torch
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode() + b"\0" + str(tuple(value.shape)).encode() + b"\0")
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def agree_digest(value, world, device):
    """Exchange fixed 32-byte values, without pickle/object deserialization."""
    import torch
    import torch.distributed as dist
    raw = bytes.fromhex(value)
    if len(raw) != 32:
        raise ValueError("Expected SHA256 identity")
    if world == 1:
        return [value]
    current = torch.tensor(list(raw), dtype=torch.uint8, device=device)
    outputs = [torch.empty_like(current) for _ in range(world)]
    dist.all_gather(outputs, current)
    values = [bytes(item.cpu().tolist()).hex() for item in outputs]
    if len(set(values)) != 1:
        raise ValueError("Compute group disagrees on inputs, profile or resulting parameters")
    return values
