# SPDX-License-Identifier: Apache-2.0

"""Shared token-logprob computation for Neuron training scripts.

Handles both a full LM head and a vocab-sharded (tensor-parallel) LM head;
in the sharded case the logsumexp / max / target-gather are reduced over the
TP group, which is mathematically identical to the single-device path.
"""

import torch


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor, vocab_size: int,
                   return_top1: bool = False):
    """Per-token logP(target) from next-token logits.

    logits  [B, T, V or V/tp] — predictions for the NEXT token
    targets [B, T]            — the actual next tokens

    Returns lp [B, T] (float32), and additionally top1 [B, T] (bool-ish
    float: 1.0 where the target token is the model argmax) if return_top1.
    """
    local_v = logits.size(-1)
    logits = logits.float()

    if local_v >= vocab_size:
        target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        max_logit = logits.max(dim=-1).values
        logsumexp = torch.logsumexp(logits, dim=-1)
        lp = target_logit - logsumexp
        if return_top1:
            top1 = (target_logit >= max_logit - 1e-6).float()
            return lp, top1
        return lp

    # Vocab-parallel path: distributed logsumexp + target gather over TP group
    from neuronx_distributed.parallel_layers import parallel_state
    group = parallel_state.get_tensor_model_parallel_group()
    rank = parallel_state.get_tensor_model_parallel_rank()

    vstart = rank * local_v
    local_targets = targets - vstart
    in_range = (local_targets >= 0) & (local_targets < local_v)
    safe = local_targets.clamp(0, local_v - 1)

    target_logit = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    target_logit = torch.where(in_range, target_logit, torch.zeros_like(target_logit))
    torch.distributed.all_reduce(target_logit, op=torch.distributed.ReduceOp.SUM, group=group)

    local_max = logits.max(dim=-1).values
    global_max = local_max.clone()
    torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=group)
    sum_exp = torch.exp(logits - global_max.unsqueeze(-1)).sum(dim=-1)
    torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=group)
    logsumexp = global_max + torch.log(sum_exp)

    lp = target_logit - logsumexp
    if return_top1:
        top1 = (target_logit >= global_max - 1e-6).float()
        return lp, top1
    return lp
