"""
Preconditioned Momentum Optimizer

Extends momentum to use key-value associative memory mapping.
The preconditioning matrix P provides more meaningful mappings.

Based on Section 2.3, Equations 19-20:
    min_m <m * grad_L^T, P_i>
    m_{i+1} = epsilon_i * m_i - eta_t * P_i * grad_L

where P can be based on Hessian information or other preconditioning.
"""

import torch
from torch.optim import Optimizer
import math


class PreconditionedMomentum(Optimizer):
    """
    Preconditioned Momentum optimizer.

    Uses a preconditioning matrix to provide better mappings between
    gradients and their corresponding values in the associative memory.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)
        precondition_type: Type of preconditioning ('diagonal', 'adam', 'none')
        eps: Term added to denominator for numerical stability (default: 1e-8)
        beta2: Exponential decay rate for second moment (default: 0.999)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        precondition_type='diagonal',
        eps=1e-8,
        beta2=0.999,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            precondition_type=precondition_type,
            eps=eps,
            beta2=beta2,
        )
        super(PreconditionedMomentum, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            precondition_type = group['precondition_type']
            eps = group['eps']
            beta2 = group['beta2']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    if precondition_type in ['diagonal', 'adam']:
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                buf = state['momentum_buffer']

                # Compute preconditioning matrix P
                if precondition_type == 'none':
                    # No preconditioning, just identity
                    preconditioned_grad = grad

                elif precondition_type == 'diagonal':
                    # Diagonal preconditioning based on gradient magnitude
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # P_i = 1 / sqrt(v_i + eps)
                    preconditioner = exp_avg_sq.sqrt().add_(eps).reciprocal()
                    preconditioned_grad = grad * preconditioner

                elif precondition_type == 'adam':
                    # Adam-style preconditioning with bias correction
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    bias_correction2 = 1 - beta2 ** state['step']

                    # P_i = 1 / sqrt(v_i / bias_correction + eps)
                    preconditioner = (exp_avg_sq / bias_correction2).sqrt().add_(eps).reciprocal()
                    preconditioned_grad = grad * preconditioner

                else:
                    raise ValueError(f"Unknown precondition_type: {precondition_type}")

                # Momentum update with preconditioning (Equation 20):
                # m_{i+1} = epsilon * m_i - lr * P_i * grad
                buf.mul_(momentum).add_(preconditioned_grad, alpha=-lr)

                # Update parameters
                p.add_(buf)

        return loss


class AdaptivePreconditionedMomentum(Optimizer):
    """
    Adaptive preconditioned momentum that learns the preconditioning.

    This variant uses a learned preconditioning matrix that adapts
    based on the gradient history.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        eps=1e-8,
        beta1=0.9,
        beta2=0.999,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            beta1=beta1,
            beta2=beta2,
        )
        super(AdaptivePreconditionedMomentum, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                # Initialize
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Adaptive preconditioning
                step_size = group['lr'] / bias_correction1
                preconditioner = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])

                # Update with preconditioned momentum
                # This combines momentum (exp_avg) with adaptive preconditioning
                p.addcdiv_(exp_avg, preconditioner, value=-step_size)

        return loss