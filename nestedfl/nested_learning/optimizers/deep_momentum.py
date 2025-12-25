"""
Deep Momentum Gradient Descent (DMGD) with True Nested Optimization

Implements the core nested learning framework from the paper:
- Outer loop: Update model parameters using learned memory
- Inner loop: Train memory modules via internal loss L^(2)

Based on Section 2.3 of the paper, Equation 23:
    W_{i+1} = W_i + m_{i+1}(u_i)
    m_{i+1} = epsilon_{i+1} * m_i - eta_t * grad_L^(2)(m_i; u_i, I)

where u_i = grad_L(W_i; x_i) and m(·) is an MLP that IS trained via internal loss.

The key insight: Memory modules learn to compress gradients by optimizing a
reconstruction/prediction objective on the gradients they process.

Version 0.3.0 additions:
- Gradient checkpointing for memory efficiency
- Factorized memory modules for large parameter tensors
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional, Dict, List, Callable
import math
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class MemoryMLP(nn.Module):
    """
    Learned memory module that processes gradients.

    This MLP learns to transform gradients into effective update directions
    by optimizing an internal loss (gradient reconstruction + momentum prediction).

    Architecture follows the paper: takes [gradient, momentum] context and
    outputs a transformed gradient.
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 64,
        depth: int = 2,
        use_context: bool = True,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.use_context = use_context

        # Input: gradient (+ momentum if use_context)
        input_dim = param_dim * 2 if use_context else param_dim

        layers = []
        current_dim = input_dim

        for i in range(depth):
            if i == depth - 1:
                # Output layer
                layers.append(nn.Linear(current_dim, param_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())  # Smooth activation
                current_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Initialize for identity-like behavior at start
        self._init_weights()

    def _init_weights(self):
        """Initialize to approximate identity transformation initially."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small weights for stable start
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Make last layer even smaller
        last_linear = None
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight, gain=0.01)

    def forward(
        self,
        grad: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process gradient through learned memory.

        Args:
            grad: Current gradient (flattened)
            momentum: Previous momentum (flattened), optional

        Returns:
            Processed gradient
        """
        if self.use_context and momentum is not None:
            x = torch.cat([grad, momentum], dim=-1)
        else:
            x = grad

        return self.network(x)


class FactorizedMemoryMLP(nn.Module):
    """
    Low-rank factorized memory module for efficiency with large parameter tensors.

    Instead of: Linear(param_dim, hidden) -> Linear(hidden, param_dim)
    Use: Linear(param_dim, rank) -> core MLP -> Linear(rank, param_dim)

    This reduces parameters from O(param_dim * hidden) to O(param_dim * rank + rank * hidden)
    which is significant when param_dim >> hidden.

    For example, with param_dim=4096, hidden=64, rank=16:
    - Standard: 4096 * 64 * 2 = 524K params
    - Factorized: 4096 * 16 * 2 + 16 * 64 * 2 = 133K params (4x reduction)
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 64,
        rank: int = 16,
        depth: int = 2,
        use_context: bool = True,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.rank = rank
        self.use_context = use_context

        # Down projection: param_dim -> rank
        # If use_context, we project gradient and momentum separately
        if use_context:
            self.down_proj_grad = nn.Linear(param_dim, rank)
            self.down_proj_mom = nn.Linear(param_dim, rank)
            core_input_dim = rank * 2
        else:
            self.down_proj = nn.Linear(param_dim, rank)
            core_input_dim = rank

        # Core MLP in low-rank space
        core_layers = []
        current_dim = core_input_dim

        for i in range(depth):
            if i == depth - 1:
                core_layers.append(nn.Linear(current_dim, rank))
            else:
                core_layers.append(nn.Linear(current_dim, hidden_dim))
                core_layers.append(nn.LayerNorm(hidden_dim))
                core_layers.append(nn.SiLU())
                current_dim = hidden_dim

        self.core = nn.Sequential(*core_layers)

        # Up projection: rank -> param_dim
        self.up_proj = nn.Linear(rank, param_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize for near-identity behavior at start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Make up projection even smaller
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)

    def forward(
        self,
        grad: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process gradient through factorized memory.

        Args:
            grad: Current gradient [batch, param_dim] or [param_dim]
            momentum: Previous momentum, same shape as grad

        Returns:
            Processed gradient, same shape as input
        """
        squeeze_output = grad.dim() == 1
        if squeeze_output:
            grad = grad.unsqueeze(0)
            if momentum is not None:
                momentum = momentum.unsqueeze(0)

        if self.use_context and momentum is not None:
            # Project gradient and momentum separately to rank space
            g_low = self.down_proj_grad(grad)
            m_low = self.down_proj_mom(momentum)
            x = torch.cat([g_low, m_low], dim=-1)
        else:
            x = self.down_proj(grad) if hasattr(self, 'down_proj') else self.down_proj_grad(grad)

        # Process in low-rank space
        out_low = self.core(x)

        # Project back to full space
        output = self.up_proj(out_low)

        if squeeze_output:
            output = output.squeeze(0)

        return output


class SharedMemoryPool(nn.Module):
    """
    Shared memory modules for efficient parameter handling.

    Instead of creating one MLP per parameter (which doesn't scale),
    we create a pool of memory modules and assign parameters to them
    based on size buckets.

    Supports:
    - Standard MemoryMLP for smaller buckets
    - Factorized memory for larger buckets (memory efficient)
    - Gradient checkpointing for reduced memory during training
    """

    def __init__(
        self,
        bucket_sizes: List[int] = [64, 256, 1024, 4096],
        hidden_dim: int = 64,
        depth: int = 2,
        use_factorized: bool = False,
        factorized_rank: int = 16,
        factorize_threshold: int = 1024,
    ):
        """
        Args:
            bucket_sizes: List of bucket sizes for parameter grouping
            hidden_dim: Hidden dimension of memory MLPs
            depth: Depth of memory MLPs
            use_factorized: Use factorized memory for large buckets
            factorized_rank: Rank for factorized memory modules
            factorize_threshold: Use factorized memory for buckets >= this size
        """
        super().__init__()
        self.bucket_sizes = sorted(bucket_sizes)
        self.use_factorized = use_factorized
        self.factorize_threshold = factorize_threshold

        # Create one memory module per bucket
        self.memories = nn.ModuleDict()
        for size in self.bucket_sizes:
            if use_factorized and size >= factorize_threshold:
                self.memories[str(size)] = FactorizedMemoryMLP(
                    param_dim=size,
                    hidden_dim=hidden_dim,
                    rank=factorized_rank,
                    depth=depth,
                )
            else:
                self.memories[str(size)] = MemoryMLP(
                    param_dim=size,
                    hidden_dim=hidden_dim,
                    depth=depth,
                )

    def get_bucket(self, param_numel: int) -> int:
        """Get the appropriate bucket size for a parameter."""
        for size in self.bucket_sizes:
            if param_numel <= size:
                return size
        return self.bucket_sizes[-1]

    def get_memory(self, param_numel: int) -> MemoryMLP:
        """Get memory module for a parameter size."""
        bucket = self.get_bucket(param_numel)
        return self.memories[str(bucket)]

    def forward(
        self,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        param_numel: int,
    ) -> torch.Tensor:
        """Process gradient through appropriate memory module."""
        bucket = self.get_bucket(param_numel)
        memory = self.memories[str(bucket)]

        # Pad or truncate to bucket size
        if param_numel < bucket:
            # Pad with zeros
            grad_padded = torch.zeros(bucket, device=grad.device, dtype=grad.dtype)
            grad_padded[:param_numel] = grad
            mom_padded = torch.zeros(bucket, device=momentum.device, dtype=momentum.dtype)
            mom_padded[:param_numel] = momentum

            output = memory(grad_padded.unsqueeze(0), mom_padded.unsqueeze(0)).squeeze(0)
            return output[:param_numel]
        else:
            # Truncate (shouldn't happen if bucket_sizes are set correctly)
            return memory(
                grad[:bucket].unsqueeze(0),
                momentum[:bucket].unsqueeze(0),
            ).squeeze(0)


class DeepMomentumGD(Optimizer):
    """
    Deep Momentum Gradient Descent with TRUE nested optimization.

    This optimizer implements the full nested learning framework:
    - Memory modules process gradients to produce update directions
    - Memory modules are trained via an internal loss (gradient reconstruction)
    - The internal loss creates a self-supervised signal for learning

    Two internal loss modes are available:

    **Surrogate mode (default, `internal_loss_mode='surrogate'`):**
    Practical loss combining cosine similarity + magnitude preservation + temporal smoothness.
    This is conceptually aligned with the paper but not mathematically identical.

    **L2 regression mode (`internal_loss_mode='l2_regression'`):**
    Paper-exact L² regression loss (Equations 21-23):
        L^(2) = ||memory(k) - P @ k||^2
    where k is the gradient (key) and P is a learned projection matrix.
    The memory learns to approximate and improve upon a linear transformation.

    Args:
        params: Model parameters to optimize
        lr: Learning rate for model parameters
        momentum: Momentum coefficient
        memory_lr: Learning rate for memory module training
        memory_hidden_dim: Hidden dimension of memory MLP
        memory_depth: Depth of memory MLP
        memory_update_freq: How often to update memory (steps)
        use_shared_memory: Use shared memory pool (more efficient)
        bucket_sizes: Bucket sizes for shared memory pool
        weight_decay: Weight decay coefficient
        gradient_checkpointing: Use gradient checkpointing for memory efficiency
        use_factorized_memory: Use factorized memory modules for large tensors
        factorized_rank: Rank for factorized memory (if enabled)
        internal_loss_mode: 'surrogate' (default) or 'l2_regression' (paper-exact)
        l2_projection_lr: Learning rate for L2 projection matrix updates
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        memory_lr: float = 1e-4,
        memory_hidden_dim: int = 64,
        memory_depth: int = 2,
        memory_update_freq: int = 1,
        use_shared_memory: bool = True,
        bucket_sizes: Optional[List[int]] = None,
        weight_decay: float = 0.0,
        gradient_checkpointing: bool = False,
        use_factorized_memory: bool = False,
        factorized_rank: int = 16,
        internal_loss_mode: str = 'surrogate',
        l2_projection_lr: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if internal_loss_mode not in ('surrogate', 'l2_regression'):
            raise ValueError(f"internal_loss_mode must be 'surrogate' or 'l2_regression', got {internal_loss_mode}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            memory_lr=memory_lr,
            memory_hidden_dim=memory_hidden_dim,
            memory_depth=memory_depth,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.memory_update_freq = memory_update_freq
        self.use_shared_memory = use_shared_memory
        self.step_count = 0
        self.gradient_checkpointing = gradient_checkpointing
        self.internal_loss_mode = internal_loss_mode
        self.l2_projection_lr = l2_projection_lr

        # Memory output history for temporal smoothness in internal loss (surrogate mode)
        # Note: We store memory outputs, not raw gradients, to enforce smooth evolution
        self._output_history: Dict[int, List[torch.Tensor]] = {}
        self._history_size = 3  # Keep last N memory outputs

        # L2 projection matrices for paper-exact mode (per bucket size)
        # These implement the P matrix in L^(2) = ||memory(k) - P @ k||^2
        self._l2_projections: Dict[int, torch.Tensor] = {}

        # Initialize memory system
        if use_shared_memory:
            if bucket_sizes is None:
                bucket_sizes = [64, 256, 1024, 4096]
            self.shared_memory = SharedMemoryPool(
                bucket_sizes=bucket_sizes,
                hidden_dim=memory_hidden_dim,
                depth=memory_depth,
                use_factorized=use_factorized_memory,
                factorized_rank=factorized_rank,
            )
            self.memory_optimizer = torch.optim.Adam(
                self.shared_memory.parameters(),
                lr=memory_lr,
            )
        else:
            self.shared_memory = None
            self._init_individual_memories()

    def _init_individual_memories(self):
        """Initialize per-parameter memory modules (less efficient)."""
        self.memory_modules = {}
        memory_params = []

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)

                param_numel = min(p.numel(), 4096)  # Cap for efficiency
                memory = MemoryMLP(
                    param_dim=param_numel,
                    hidden_dim=group['memory_hidden_dim'],
                    depth=group['memory_depth'],
                ).to(p.device)

                state['memory_module'] = memory
                self.memory_modules[id(p)] = memory
                memory_params.extend(memory.parameters())

        self.memory_optimizer = torch.optim.Adam(
            memory_params,
            lr=self.param_groups[0]['memory_lr'],
        )

    def _compute_internal_loss(
        self,
        memory_output: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        param_id: int,
        bucket_size: int,
    ) -> torch.Tensor:
        """
        Compute internal loss L^(2) for memory module training.

        Dispatches to either surrogate or L2 regression mode based on config.

        Args:
            memory_output: Output from memory module
            grad: Current gradient
            momentum: Previous momentum
            param_id: Parameter identifier
            bucket_size: Size of the bucket (for L2 projection lookup)

        Returns:
            Internal loss value
        """
        if self.internal_loss_mode == 'l2_regression':
            return self._compute_l2_regression_loss(memory_output, grad, bucket_size)
        else:
            return self._compute_surrogate_loss(memory_output, grad, param_id)

    def _compute_surrogate_loss(
        self,
        memory_output: torch.Tensor,
        grad: torch.Tensor,
        param_id: int,
    ) -> torch.Tensor:
        """
        Compute surrogate internal loss (default mode).

        Combines cosine similarity, magnitude preservation, and temporal smoothness.
        This is conceptually aligned with the paper but not mathematically identical.

        Args:
            memory_output: Output from memory module
            grad: Current gradient
            param_id: Parameter identifier

        Returns:
            Internal loss value
        """
        # Component 1: Reconstruction loss
        # The memory output should capture the essential gradient direction
        # We use cosine similarity to measure directional alignment
        if grad.norm() > 1e-8 and memory_output.norm() > 1e-8:
            cosine_sim = torch.nn.functional.cosine_similarity(
                memory_output.unsqueeze(0),
                grad.unsqueeze(0),
            )
            reconstruction_loss = 1 - cosine_sim.mean()
        else:
            reconstruction_loss = torch.tensor(0.0, device=grad.device)

        # Component 2: Magnitude preservation
        # The output magnitude should be proportional to gradient magnitude
        target_magnitude = grad.norm()
        output_magnitude = memory_output.norm()
        if target_magnitude > 1e-8:
            magnitude_ratio = output_magnitude / target_magnitude
            magnitude_loss = (magnitude_ratio - 1).pow(2)
        else:
            magnitude_loss = output_magnitude.pow(2)

        # Component 3: Temporal smoothness (if we have history)
        # We compare current memory output to previous memory output (not raw gradient)
        # This encourages the learned memory transformation to evolve smoothly
        temporal_loss = torch.tensor(0.0, device=grad.device)
        if param_id in self._output_history and len(self._output_history[param_id]) > 0:
            prev_output = self._output_history[param_id][-1]
            if prev_output.shape == memory_output.shape:
                # Memory output should change smoothly over time
                temporal_loss = (memory_output - prev_output).pow(2).mean() * 0.1

        # Combine losses
        total_loss = reconstruction_loss + 0.1 * magnitude_loss + temporal_loss

        return total_loss

    def _compute_l2_regression_loss(
        self,
        memory_output: torch.Tensor,
        grad: torch.Tensor,
        bucket_size: int,
    ) -> torch.Tensor:
        """
        Compute paper-exact L² regression internal loss (Equations 21-23).

        L^(2) = ||memory(k) - P @ k||^2

        where k is the gradient (key), memory(k) is the memory output (value),
        and P is a learned projection matrix that provides the target.

        The memory module learns to approximate and improve upon this linear
        transformation, capturing non-linear gradient compression patterns.

        The projection P is updated via delta rule to track the memory's output:
            P += lr * (memory(k) - P @ k) @ k^T / ||k||^2

        Args:
            memory_output: Output from memory module (value)
            grad: Current gradient (key)
            bucket_size: Size bucket for projection matrix lookup

        Returns:
            L² regression loss
        """
        device = grad.device
        dtype = grad.dtype
        output_dim = memory_output.shape[0]
        input_dim = min(grad.shape[0], output_dim)  # Handle dimension mismatch

        # Initialize projection matrix if needed
        if bucket_size not in self._l2_projections:
            # Initialize P as scaled identity to start near identity transform
            P = torch.eye(output_dim, input_dim, device=device, dtype=dtype) * 0.1
            self._l2_projections[bucket_size] = P

        P = self._l2_projections[bucket_size]

        # Move P to correct device if needed
        if P.device != device:
            P = P.to(device)
            self._l2_projections[bucket_size] = P

        # Compute target: P @ k
        grad_for_proj = grad[:input_dim]  # Truncate if needed
        target = P @ grad_for_proj

        # L² loss: ||memory(k) - P @ k||^2
        loss = (memory_output - target).pow(2).mean()

        # Update projection matrix via delta rule (with gradients disabled)
        # This allows P to track what the memory is learning to output
        with torch.no_grad():
            grad_norm_sq = (grad_for_proj ** 2).sum()
            if grad_norm_sq > 1e-8:
                # Delta rule: P += lr * error @ k^T / ||k||^2
                error = memory_output.detach() - target.detach()
                outer = torch.outer(error, grad_for_proj)
                P_update = self.l2_projection_lr * outer / grad_norm_sq
                self._l2_projections[bucket_size] = P + P_update

        return loss

    def _update_output_history(self, param_id: int, memory_output: torch.Tensor):
        """Update memory output history for temporal smoothness loss.

        Note: We store memory outputs (not raw gradients) to encourage
        the learned transformation to evolve smoothly over time.
        """
        if param_id not in self._output_history:
            self._output_history[param_id] = []

        self._output_history[param_id].append(memory_output.detach().clone())

        # Keep only last N outputs
        if len(self._output_history[param_id]) > self._history_size:
            self._output_history[param_id].pop(0)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step with nested learning.

        This implements the full nested optimization:
        1. Process gradients through memory modules (compute internal loss)
        2. Update memory modules with internal loss (BEFORE modifying other tensors)
        3. Update model parameters with processed gradients
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Phase 1: Collect all gradient info and compute internal losses
        # We do this BEFORE any tensor modifications to support gradient checkpointing
        internal_losses = []
        param_updates = []  # Store (param, state, processed_grad, momentum_coef, lr)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum_coef = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                param_id = id(p)

                # Initialize state
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = state['momentum_buffer']

                # Flatten for memory processing
                grad_flat = grad.flatten()
                momentum_flat = momentum_buffer.flatten()
                param_numel = p.numel()

                # Process gradient through memory module
                if self.use_shared_memory:
                    # Get appropriate memory from pool (detached for outer loop)
                    memory_output = self.shared_memory(
                        grad_flat.detach(),
                        momentum_flat.detach(),
                        param_numel,
                    )

                    # Compute with grad enabled for internal loss
                    # Use gradient checkpointing if enabled
                    if self.gradient_checkpointing:
                        # Clone tensors to avoid in-place issues during recomputation
                        grad_for_ckpt = grad_flat.clone()
                        mom_for_ckpt = momentum_flat.clone()

                        # Capture param_numel in closure by binding it
                        numel = param_numel  # Local copy to avoid closure issues

                        def make_memory_forward(n):
                            def memory_forward(g, m):
                                return self.shared_memory(g, m, n)
                            return memory_forward

                        memory_output_for_loss = gradient_checkpoint(
                            make_memory_forward(numel),
                            grad_for_ckpt,
                            mom_for_ckpt,
                            use_reentrant=False,
                        )
                    else:
                        memory_output_for_loss = self.shared_memory(
                            grad_flat,
                            momentum_flat,
                            param_numel,
                        )
                else:
                    memory = state['memory_module']
                    capped_numel = min(param_numel, 4096)

                    memory_output = memory(
                        grad_flat[:capped_numel].detach().unsqueeze(0),
                        momentum_flat[:capped_numel].detach().unsqueeze(0),
                    ).squeeze(0)

                    if self.gradient_checkpointing:
                        grad_for_ckpt = grad_flat[:capped_numel].clone()
                        mom_for_ckpt = momentum_flat[:capped_numel].clone()

                        # Capture memory module in closure
                        mem = memory

                        def make_memory_forward_ind(m):
                            def memory_forward(g, mom):
                                return m(g.unsqueeze(0), mom.unsqueeze(0)).squeeze(0)
                            return memory_forward

                        memory_output_for_loss = gradient_checkpoint(
                            make_memory_forward_ind(mem),
                            grad_for_ckpt,
                            mom_for_ckpt,
                            use_reentrant=False,
                        )
                    else:
                        memory_output_for_loss = memory(
                            grad_flat[:capped_numel].unsqueeze(0),
                            momentum_flat[:capped_numel].unsqueeze(0),
                        ).squeeze(0)

                # Compute internal loss
                # Get bucket size for L2 projection lookup
                if self.use_shared_memory:
                    bucket_size = self.shared_memory.get_bucket(param_numel)
                else:
                    bucket_size = min(param_numel, 4096)

                internal_loss = self._compute_internal_loss(
                    memory_output_for_loss,
                    grad_flat[:memory_output_for_loss.shape[0]],
                    momentum_flat[:memory_output_for_loss.shape[0]],
                    param_id,
                    bucket_size,
                )
                internal_losses.append(internal_loss)

                # Store update info for Phase 3
                param_updates.append({
                    'param': p,
                    'state': state,
                    'param_id': param_id,
                    'memory_output': memory_output,
                    'grad': grad,
                    'param_numel': param_numel,
                    'momentum_coef': momentum_coef,
                    'lr': lr,
                })

        # Phase 2: Update memory modules with internal loss
        # Do this BEFORE modifying momentum buffers
        if len(internal_losses) > 0 and self.step_count % self.memory_update_freq == 0:
            total_internal_loss = torch.stack(internal_losses).mean()

            self.memory_optimizer.zero_grad()
            total_internal_loss.backward()

            # Clip gradients for stability
            if self.use_shared_memory:
                torch.nn.utils.clip_grad_norm_(
                    self.shared_memory.parameters(),
                    max_norm=1.0,
                )
            else:
                for memory in self.memory_modules.values():
                    torch.nn.utils.clip_grad_norm_(
                        memory.parameters(),
                        max_norm=1.0,
                    )

            self.memory_optimizer.step()

        # Phase 3: Apply parameter updates (now safe to modify tensors)
        for update in param_updates:
            p = update['param']
            state = update['state']
            memory_output = update['memory_output']
            grad = update['grad']
            param_numel = update['param_numel']
            momentum_coef = update['momentum_coef']
            lr = update['lr']
            param_id = update['param_id']

            momentum_buffer = state['momentum_buffer']
            grad_flat = grad.flatten()

            # Update output history for temporal smoothness
            self._update_output_history(param_id, memory_output.detach())

            # Reshape and apply to momentum
            if memory_output.shape[0] < param_numel:
                # Pad with original gradient
                full_output = grad_flat.clone()
                full_output[:memory_output.shape[0]] = memory_output
                processed_grad = full_output.view_as(grad)
            else:
                processed_grad = memory_output.view_as(grad)

            # Update momentum and parameters
            momentum_buffer.mul_(momentum_coef).add_(processed_grad.detach())
            p.data.add_(momentum_buffer, alpha=-lr)

        self.step_count += 1

        return loss

    def get_memory_modules(self) -> List[nn.Module]:
        """Return memory modules for inspection/saving."""
        if self.use_shared_memory:
            return [self.shared_memory]
        else:
            return list(self.memory_modules.values())

    def get_internal_loss_stats(self) -> Dict[str, float]:
        """Get statistics about internal loss for monitoring."""
        stats = {
            'step_count': self.step_count,
            'num_params_tracked': len(self._output_history),
            'internal_loss_mode': self.internal_loss_mode,
        }
        if self.internal_loss_mode == 'l2_regression':
            stats['num_l2_projections'] = len(self._l2_projections)
        return stats


class SimpleMomentumGD(Optimizer):
    """
    Standard momentum gradient descent for comparison.

    This is the baseline optimizer without deep memory.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

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
                buf.mul_(group['momentum']).add_(grad)
                p.add_(buf, alpha=-group['lr'])

        return loss
