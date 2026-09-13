"""Memory-bounded adaptation of the pinned PyTorch PowerSGD research hook."""


def _power_sgd(state, bucket, offload_errors):
    """Complete one bucket before dispatching another bucket's collectives."""
    from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook

    index = bucket.index()
    if index in state.error_dict:
        state.error_dict[index] = state.error_dict[index].to(bucket.buffer().device)
    pending = powerSGD_hook(state, bucket)
    pending.wait()
    if offload_errors and index in state.error_dict:
        state.error_dict[index] = state.error_dict[index].detach().cpu()
    return pending


def resident_power_sgd(state, bucket):
    """Numerical control with completed bucket reductions and GPU residuals."""
    return _power_sgd(state, bucket, False)


def offloaded_power_sgd(state, bucket):
    """Keep dense FP32 error feedback on CPU between bucket reductions.

    Restore a bucket immediately before the upstream hook consumes its error;
    copy the new error back before dispatching another bucket. Completing the
    future also prevents rank-dependent interleaving of multi-stage collectives.
    This sacrifices communication overlap. No residual quantization or extra
    arithmetic is introduced. Projection state stays on GPU.

    This does not define a checkpoint codec or a native verification rule.
    """
    return _power_sgd(state, bucket, True)
