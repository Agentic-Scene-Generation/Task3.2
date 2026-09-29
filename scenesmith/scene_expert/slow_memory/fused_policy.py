"""Avoid projecting loss-masked prompt positions into Qwen's large vocabulary."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


class FusedLossOffloadContext:
    """Keep model activations offloaded without enclosing torch.func loss AD.

    TRL wraps the entire training step in saved-tensor hooks, while its fused
    loss uses torch.func, which rejects active saved-tensor hooks. Pause only
    hook registration around that small completion-only loss; retain the
    offloader's tensor tracker and streams for the model's actual backward.
    """

    def __init__(self, context: Any) -> None:
        import torch

        if not isinstance(context, torch.autograd.graph.saved_tensors_hooks):
            raise TypeError("expected the pinned TRL saved-tensor offload context")
        self.context = context
        self.active = False

    def __enter__(self) -> FusedLossOffloadContext:
        if self.active:
            raise RuntimeError("offload training-step context cannot be nested")
        self.context.__enter__()
        self.active = True
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self.context.__exit__(*exc)
        finally:
            self.active = False

    @contextmanager
    def pause_for_fused_loss(self) -> Iterator[None]:
        import torch

        if not self.active:
            # Evaluation does not enter the training-step offload context.
            yield
            return
        hooks = torch.autograd.graph.saved_tensors_hooks
        # Call only the base hook-stack API. OffloadActivations.__enter__/exit
        # clear its tracker/stashes and must run just once per training step.
        hooks.__exit__(self.context, None, None, None)
        try:
            yield
        finally:
            hooks.__enter__(self.context)


def completion_only_fused_loss(
    base_loss: Any, *, offload_context: FusedLossOffloadContext | None = None
) -> Any:
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
            with (
                offload_context.pause_for_fused_loss()
                if offload_context is not None
                else nullcontext()
            ):
                return self.base_loss(
                    weight,
                    hidden[:, keep].contiguous(),
                    labels[:, keep].contiguous(),
                    bias,
                    (
                        ref_hidden[:, keep].contiguous()
                        if ref_hidden is not None
                        else None
                    ),
                    ref_weight,
                    ref_bias,
                )

    return CompletionOnlyLoss()
