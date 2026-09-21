"""Avoid projecting loss-masked prompt positions into Qwen's large vocabulary."""

from __future__ import annotations

from typing import Any


def completion_only_fused_loss(base_loss: Any) -> Any:
    """Wrap TRL's fused loss without changing the masked DPO objective.

    Both policy and reference still run on the complete multimodal context.
    Only hidden positions whose labels are ignored in every batch row are
    removed, immediately before the loss projects hidden states into logits.
    Gradients through the retained states still reach their entire context.
    """
    import torch

    class CompletionOnlyLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_loss = base_loss

        def forward(
            self,
            weight: Any,
            hidden: Any,
            labels: Any,
            bias: Any = None,
            ref_hidden: Any = None,
            ref_weight: Any = None,
            ref_bias: Any = None,
        ) -> Any:
            if hidden.shape[:2] != labels.shape or labels.ndim != 2:
                raise ValueError("fused DPO labels must match the full hidden sequence")
            valid = labels.ne(-100)
            if not valid.any(dim=1).all().item():
                raise ValueError(
                    "DPO candidate contains no supervised completion token"
                )
            keep = valid.any(dim=0)
            return self.base_loss(
                weight,
                hidden[:, keep].contiguous(),
                labels[:, keep].contiguous(),
                bias,
                ref_hidden[:, keep].contiguous() if ref_hidden is not None else None,
                ref_weight,
                ref_bias,
            )

    return CompletionOnlyLoss()
