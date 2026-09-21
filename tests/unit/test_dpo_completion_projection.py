"""Completion projection preserves loss and gradients through the entire prompt."""

from __future__ import annotations

import pytest

from scenesmith.scene_expert.slow_memory.fused_policy import completion_only_fused_loss

torch = pytest.importorskip("torch")


class DenseDPOLoss(torch.nn.Module):
    def forward(self, weight, hidden, labels, bias, ref_hidden, ref_weight, ref_bias):
        self.projected_positions = labels.shape[1]
        valid = labels.ne(-100)
        target = labels.clamp_min(0)

        def logps(states, matrix):
            logits = states @ matrix.T
            values = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            return (values * valid).sum(-1)

        ratios = logps(hidden, weight) - logps(ref_hidden, ref_weight)
        chosen, rejected = ratios.chunk(2)
        return -torch.nn.functional.logsigmoid(0.1 * (chosen - rejected)).mean()


def test_projection_is_exact_with_different_answer_lengths_and_prompt_gradients():
    torch.manual_seed(42)
    inputs = torch.randn(2, 9, 6, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    weight = torch.randn(11, 6)
    reference = torch.randn(2, 9, 6)
    labels = torch.tensor([[-100] * 5 + [1, 3, 4, 7], [-100] * 5 + [2, 1, -100, -100]])
    full_loss = DenseDPOLoss()
    compact_loss = DenseDPOLoss()
    # A causal prefix contributes to every later hidden state. Its gradients
    # must survive dropping only loss-masked vocabulary projections.
    expected = full_loss(
        weight, inputs.cumsum(1), labels, None, reference, weight, None
    )
    actual = completion_only_fused_loss(compact_loss)(
        weight, other.cumsum(1), labels, None, reference, weight, None
    )
    expected.backward()
    actual.backward()
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(inputs.grad, other.grad, atol=1e-6)
    assert inputs.grad[:, :5].abs().sum() > 0
    assert compact_loss.projected_positions == 4 < full_loss.projected_positions


def test_empty_candidate_target_is_rejected():
    loss = completion_only_fused_loss(DenseDPOLoss())
    with pytest.raises(ValueError, match="no supervised"):
        loss(torch.zeros(3, 2), torch.zeros(2, 4, 2), torch.full((2, 4), -100))
