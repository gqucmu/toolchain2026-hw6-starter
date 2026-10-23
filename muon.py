"""
Muon optimizer, computed in float32.

Usage, like any PyTorch optimizer:

    from muon import Muon
    optimizer = Muon(muon_params, lr=muon_lr, weight_decay=0.1)

where muon_params is the list of parameters Muon should train. Muon only supports 2D parameters
(weight matrices). The other arguments (momentum=0.95, nesterov=True, ns_steps=5) can be left at
their defaults.

This follows PyTorch's own Muon implementation, but computes in float32: PyTorch's version computes in
bfloat16, which is fast on GPUs but can be about 100x slower than float32 on laptop CPUs.
"""
import math

import torch


def zeropower_via_newtonschulz(grad, ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5, eps=1e-7):
    """Approximately orthogonalize the matrix grad: replace its singular values by (roughly) 1."""
    a, b, c = ns_coefficients
    X = grad
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)              # make sure the largest singular value is at most 1
    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True,
                 ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_coefficients=ns_coefficients, eps=eps, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters, got a parameter of shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)                        # B_t = mu * B_{t-1} + g_t
                update = grad.add(buf, alpha=mu) if group["nesterov"] else buf

                update = zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])

                p.mul_(1 - lr * group["weight_decay"])        # decoupled weight decay
                A, B = p.shape
                adjusted_lr = lr * math.sqrt(max(1, A / B))    # same scaling as torch.optim.Muon's default
                p.add_(update, alpha=-adjusted_lr)

        return loss
