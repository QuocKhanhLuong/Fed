"""
Nested Deep Momentum GD with Meta-Learning

An attempt at implementing the nested learning framework from the paper:
- Memory modules that can be trained via meta-learning
- Internal loss via `meta_step()` method
- Proper gradient flow through memory modules

STATUS: NEEDS VALIDATION
- Code structure attempts to follow paper's concepts
- Meta-learning loop is implemented but not thoroughly tested
- No experimental validation against paper's results
- May have bugs or deviations from paper's formulation

This is an improvement over deep_momentum.py (which uses static MLPs),
but should still be considered experimental/research-quality code.
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional, List, Callable


class LearnedMemoryModule(nn.Module):
    """
    MLP that learns to compress gradients.

    This is the core "memory" that replaces linear momentum.
    It will be trained via meta-learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(depth):
            if i == depth - 1:
                # Last layer
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))  # Add stability
                layers.append(nn.ReLU())
                current_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Initialize with small weights for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, grad: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """
        Process gradient and momentum to produce new momentum.

        Args:
            grad: Current gradient (flattened)
            momentum: Previous momentum (flattened)

        Returns:
            New momentum
        """
        # Concatenate gradient and momentum as input context
        x = torch.cat([grad, momentum], dim=-1)
        return self.network(x)


class NestedDeepMomentumGD(Optimizer):
    """
    Deep Momentum GD with proper nested optimization.

    This implements the full nested learning framework:
    - Outer loop: Optimize model parameters using learned memory
    - Inner loop: Optimize memory module parameters via meta-learning

    Args:
        params: Model parameters to optimize
        memory_optimizer: Optimizer for the memory modules (e.g., Adam)
        lr: Learning rate for model parameters
        memory_lr: Learning rate for memory module parameters
        momentum: Momentum coefficient
        memory_depth: Depth of memory MLP
        memory_hidden_dim: Hidden dimension of memory MLP
        meta_learning: Whether to enable meta-learning for memory
    """

    def __init__(
        self,
        params,
        memory_optimizer: Optional[Optimizer] = None,
        lr: float = 1e-3,
        memory_lr: float = 1e-4,
        momentum: float = 0.9,
        memory_depth: int = 2,
        memory_hidden_dim: int = 64,
        meta_learning: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            memory_depth=memory_depth,
            memory_hidden_dim=memory_hidden_dim,
            memory_lr=memory_lr,
        )
        super().__init__(params, defaults)

        self.meta_learning = meta_learning
        self.memory_modules = {}
        self.memory_optimizer = memory_optimizer

        # Initialize memory modules for each parameter
        self._init_memory_modules()

    def _init_memory_modules(self):
        """Initialize learned memory modules for each parameter group."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Initialize momentum buffer
                state['momentum_buffer'] = torch.zeros_like(p.data)

                # Get dimensions
                param_numel = p.numel()
                hidden_dim = group['memory_hidden_dim']

                # Create learned memory module
                # Input: [gradient, momentum] (concatenated)
                # Output: new momentum
                memory = LearnedMemoryModule(
                    input_dim=param_numel * 2,  # grad + momentum
                    hidden_dim=hidden_dim,
                    output_dim=param_numel,
                    depth=group['memory_depth'],
                )

                # Move to same device as parameter
                memory = memory.to(p.device)

                # Store in state and module dict
                state['memory_module'] = memory
                self.memory_modules[id(p)] = memory

        # Create optimizer for memory modules if meta-learning is enabled
        if self.meta_learning and self.memory_optimizer is None:
            memory_params = []
            for memory in self.memory_modules.values():
                memory_params.extend(memory.parameters())

            self.memory_optimizer = torch.optim.Adam(
                memory_params,
                lr=self.param_groups[0]['memory_lr'],
            )

    def get_memory_modules(self) -> List[nn.Module]:
        """Return list of all memory modules."""
        return list(self.memory_modules.values())

    def step(self, closure: Optional[Callable] = None, create_graph: bool = False):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for meta-learning.
            create_graph: If True, preserve computation graph for meta-learning.
                         This allows backprop through the optimization step.

        Note: For meta-learning to work, set create_graph=True. This keeps the
        computation graph through the memory modules so meta_step can compute
        gradients w.r.t. memory module parameters.

        The key insight: we store the memory outputs and use them in
        compute_meta_gradients() to create a gradient path back to memory modules.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Store outputs for meta-learning backward pass
        self._last_memory_outputs = []

        # Outer loop: Update model parameters using learned memory
        for group in self.param_groups:
            lr = group['lr']
            momentum_coef = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                memory_module = state['memory_module']

                # Detach gradient - we don't need higher order gradients through p.grad
                # We only need gradients through the memory module
                grad = p.grad.detach()

                # Flatten tensors
                grad_flat = grad.flatten()
                momentum_flat = state['momentum_buffer'].detach().flatten()

                # Process through learned memory module
                memory_module.train()
                new_momentum_flat = memory_module(
                    grad_flat.unsqueeze(0),
                    momentum_flat.unsqueeze(0),
                ).squeeze(0)

                # Store for meta-learning - this is what allows gradient flow
                if create_graph:
                    self._last_memory_outputs.append(new_momentum_flat)

                # Reshape back
                new_momentum = new_momentum_flat.view_as(p)

                # Compute full update with momentum (keep graph through new_momentum)
                combined_momentum = momentum_coef * state['momentum_buffer'].detach() + new_momentum

                # Update momentum buffer (detached copy - no in-place ops that affect graph)
                state['momentum_buffer'] = combined_momentum.detach().clone()

                # Update parameters (detached - actual parameter update)
                with torch.no_grad():
                    p.add_(combined_momentum.detach(), alpha=-lr)

        return loss

    def compute_meta_gradients(self, meta_loss: torch.Tensor):
        """
        Compute and accumulate gradients to memory modules from meta_loss.

        Because parameter updates happen in no_grad, the meta_loss doesn't
        directly connect to memory modules. This method creates that connection
        using a surrogate loss based on the stored memory outputs.

        The approach:
        1. We stored the memory module outputs during step()
        2. We compute a "surrogate" gradient target from meta_loss
        3. We backprop through the stored outputs to update memory modules

        Args:
            meta_loss: The meta loss computed after optimization steps
        """
        if not hasattr(self, '_last_memory_outputs') or not self._last_memory_outputs:
            return

        # Create a surrogate loss that connects meta_loss to memory outputs
        # The idea: we want memory outputs that would have led to lower meta_loss
        # We approximate this by backpropping meta_loss through model params,
        # then using those gradients as targets for the memory outputs

        # Get model parameters that were updated
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        # Get gradients of meta_loss w.r.t. current model parameters
        try:
            param_grads = torch.autograd.grad(
                meta_loss,
                params,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )
        except RuntimeError:
            # If meta_loss doesn't depend on params (shouldn't happen), skip
            return

        # Create surrogate loss: we want memory outputs to produce updates
        # that move parameters in the direction that reduces meta_loss
        # L_surrogate = sum_i (memory_output_i * grad_meta_i)
        # Minimizing this encourages memory to produce updates aligned with -grad_meta
        # Use meta_loss to get correct device/dtype
        surrogate_loss = meta_loss.new_zeros(())
        for i, (memory_out, grad) in enumerate(zip(self._last_memory_outputs, param_grads)):
            if grad is not None and memory_out.grad_fn is not None:
                # Reshape grad to match memory output
                grad_flat = grad.flatten()
                if grad_flat.shape == memory_out.shape:
                    # Want memory_out to point opposite to grad (for descent)
                    surrogate_loss = surrogate_loss + (memory_out * grad_flat).sum()

        # Backprop surrogate loss to memory modules
        if surrogate_loss.grad_fn is not None:
            surrogate_loss.backward()

    def meta_step(self, meta_loss: torch.Tensor):
        """
        Perform meta-learning step to improve memory modules.

        This implements the inner optimization loop L̃^(2) from the paper.

        The key challenge: standard backprop through meta_loss won't reach
        memory modules because parameter updates happen in no_grad().

        Solution: We use compute_meta_gradients() to create a surrogate loss
        that connects the meta_loss signal to memory module outputs.

        Args:
            meta_loss: Loss that depends on how well the optimizer performs
                      (e.g., validation loss after several optimization steps)

        Note: For this to work, you must call optimizer.step(create_graph=True)
        during the inner optimization loop to preserve the computation graph.
        """
        if not self.meta_learning:
            return

        if self.memory_optimizer is None:
            raise ValueError("Meta-learning enabled but no memory optimizer provided")

        # Zero gradients first
        self.memory_optimizer.zero_grad()

        # Check if we have stored memory outputs from create_graph=True steps
        if hasattr(self, '_last_memory_outputs') and self._last_memory_outputs:
            # Use surrogate loss approach to connect meta_loss to memory modules
            self.compute_meta_gradients(meta_loss)
        else:
            # Fallback: try direct backprop (may not work if graph not retained)
            try:
                meta_loss.backward(retain_graph=True)
            except RuntimeError:
                # Graph was not retained, memory modules won't be updated
                import warnings
                warnings.warn(
                    "meta_step called but computation graph not available. "
                    "Make sure to call step(create_graph=True) during inner loop."
                )
                return

        # Clip gradients for stability
        for memory in self.memory_modules.values():
            torch.nn.utils.clip_grad_norm_(memory.parameters(), max_norm=1.0)

        # Update memory module parameters
        self.memory_optimizer.step()

        # Clear stored outputs
        if hasattr(self, '_last_memory_outputs'):
            self._last_memory_outputs = []

    def state_dict(self):
        """
        Return state dict including memory modules.
        """
        state = super().state_dict()

        # Add memory modules
        memory_state = {}
        for param_id, memory in self.memory_modules.items():
            memory_state[param_id] = memory.state_dict()

        state['memory_modules'] = memory_state

        if self.memory_optimizer is not None:
            state['memory_optimizer'] = self.memory_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict):
        """
        Load state dict including memory modules.
        """
        # Load memory modules
        if 'memory_modules' in state_dict:
            memory_state = state_dict.pop('memory_modules')
            for param_id, memory_dict in memory_state.items():
                if param_id in self.memory_modules:
                    self.memory_modules[param_id].load_state_dict(memory_dict)

        # Load memory optimizer
        if 'memory_optimizer' in state_dict and self.memory_optimizer is not None:
            mem_opt_state = state_dict.pop('memory_optimizer')
            self.memory_optimizer.load_state_dict(mem_opt_state)

        # Load base optimizer state
        super().load_state_dict(state_dict)


def create_meta_learning_task_distribution(
    num_tasks: int = 100,
    input_dim: int = 10,
    output_dim: int = 1,
    task_type: str = 'regression',
) -> List[Callable]:
    """
    Create a distribution of tasks for meta-learning the optimizer.

    This is used to train the memory modules on a variety of optimization
    problems so they learn general optimization strategies.

    Args:
        num_tasks: Number of tasks in distribution
        input_dim: Input dimension for tasks
        output_dim: Output dimension for tasks
        task_type: Type of tasks ('regression', 'classification', 'mixed')

    Returns:
        List of task generator functions
    """
    tasks = []

    for _ in range(num_tasks):
        if task_type == 'regression' or (task_type == 'mixed' and torch.rand(1).item() < 0.5):
            # Random regression task
            true_weights = torch.randn(input_dim, output_dim)
            noise_level = torch.rand(1).item() * 0.5

            def task_fn():
                X = torch.randn(32, input_dim)
                y = X @ true_weights + noise_level * torch.randn(32, output_dim)
                return X, y

            tasks.append(task_fn)
        else:
            # Random classification task (binary)
            true_weights = torch.randn(input_dim, output_dim)

            def task_fn():
                X = torch.randn(32, input_dim)
                logits = X @ true_weights
                y = (logits > 0).float()
                return X, y

            tasks.append(task_fn)

    return tasks
