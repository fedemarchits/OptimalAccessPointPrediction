"""
Gradient Reversal Layer (GRL) for domain-adversarial training.

The GRL is an identity function in the forward pass but negates the gradient
during backpropagation, multiplied by a scaling factor lambda_.

Usage:
    from models.grad_reverse import grad_reverse
    reversed_feats = grad_reverse(features, lambda_=0.5)
"""

import torch
from torch.autograd import Function


class _GRL(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """
    Apply gradient reversal to tensor x with scaling factor lambda_.

    Forward pass: identity (x is returned unchanged).
    Backward pass: gradient is negated and scaled by lambda_.

    Args:
        x:       Input tensor.
        lambda_: Reversal strength (0 = no reversal, 1 = full reversal).
    """
    return _GRL.apply(x, lambda_)
