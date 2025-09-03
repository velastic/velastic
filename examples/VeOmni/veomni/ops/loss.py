from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.constants import IGNORE_INDEX
from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import reduce_sequence_parallel_loss
from ..utils import logging
from ..utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


fused_linear_cross_entropy = None

if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore

    fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="mean")


def causallm_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: Optional[int] = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # We don't use shift_labels in causallm
    assert shift_labels is None

    loss = None
    logits = None

    if labels is None:
        logits = F.linear(hidden_states, weight)
        return loss, logits

    sp_enabled = get_parallel_state().sp_enabled

    # Shift the labels and hidden_states so that tokens < n predict n
    if not sp_enabled:
        labels = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()

    # Flatten the labels and hidden_states
    labels = labels.view(-1)
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))

    # Calculate loss
    if fused_linear_cross_entropy is not None:  # use liger kernels
        loss = fused_linear_cross_entropy(weight, hidden_states, labels)
    else:
        logits = F.linear(hidden_states, weight).float()
        loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (labels != IGNORE_INDEX).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)

    return loss, logits
