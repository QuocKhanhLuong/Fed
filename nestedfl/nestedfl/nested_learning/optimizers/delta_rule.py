"""
Delta-Rule Momentum Optimizer

Uses L2 regression objective for momentum update instead of dot-product similarity.
This allows better capacity management for memorizing gradients.

Based on Section 2.3, Equations 21-22:
    W_{i+1} = W_i + m_{i+1}
    m_{i+1} = (epsilon_i * I - grad_L * grad_L^T) * m_i - eta_t * P_i * grad_L

The delta rule helps the memory better manage its limited capacity.
"""

import torch
from torch.optim import Optimizer


class DeltaRuleMomentum(Optimizer):
    """
    Momentum optimizer using delta-rule for updates.

    Instead of simple Hebbian-like updates, uses delta-rule which
    measures the error between current state and target, leading to
    better memory management.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient epsilon (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)
        precondition: Whether to use preconditioning matrix P (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        precondition=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            precondition=precondition,
        )
        super(DeltaRuleMomentum, self).__init__(params, defaults)

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
            precondition = group['precondition']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']

                # Flatten for matrix operations
                grad_flat = grad.flatten()
                buf_flat = buf.flatten()

                # Compute outer product term: grad * grad^T
                # For efficiency, we use the fact that (grad * grad^T) * buf = grad * (grad^T * buf)
                grad_dot_buf = torch.dot(grad_flat, buf_flat)
                outer_product_term = grad_flat * grad_dot_buf

                # Delta-rule update (Equation 22):
                # m_{i+1} = (epsilon * I - grad * grad^T) * m_i - lr * grad
                # = epsilon * m_i - grad * (grad^T * m_i) - lr * grad
                buf_flat_new = momentum * buf_flat - outer_product_term

                # Add gradient term
                if precondition:
                    # Use preconditioning (simplified version)
                    # In practice, P could be based on second moment or Hessian
                    if 'preconditioner' not in state:
                        state['preconditioner'] = torch.ones_like(grad_flat)
                    P = state['preconditioner']
                    buf_flat_new.add_(P * grad_flat, alpha=-lr)
                else:
                    buf_flat_new.add_(grad_flat, alpha=-lr)

                # Reshape and update buffer
                buf.copy_(buf_flat_new.view_as(buf))

                # Update parameters
                p.add_(buf, alpha=-1.0)

        return loss


class OjaRuleMomentum(Optimizer):
    """
    Momentum with Oja's rule for normalization.

    Oja's rule is a variant that includes self-normalization,
    preventing unbounded growth of the momentum buffer.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        oja_lr=0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            oja_lr=oja_lr,
        )
        super(OjaRuleMomentum, self).__init__(params, defaults)

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

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']

                # Oja's rule includes normalization term
                # m_{i+1} = m_i + eta * (grad - (grad^T * m_i) * m_i)
                grad_flat = grad.flatten()
                buf_flat = buf.flatten()

                projection = torch.dot(grad_flat, buf_flat)

                # Update: m = momentum * m + oja_lr * (grad - projection * m)
                buf.mul_(group['momentum'])
                buf.add_(grad, alpha=group['oja_lr'])
                buf.add_(buf, alpha=-group['oja_lr'] * projection)

                # Update parameters
                p.add_(buf, alpha=-group['lr'])

        return loss