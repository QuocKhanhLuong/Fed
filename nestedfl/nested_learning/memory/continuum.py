"""
Continuum Memory System (CMS)

Implements the multi-frequency memory from Section 3, Equations 30-31:
    y_t = MLP^(f_k)(...MLP^(f_1)(x_t))
    theta^(f_l)_{i+1} = theta^(f_l)_i - sum(...) if i = 0 (mod C^(l))

Each MLP level updates at a different frequency (chunk size), creating
a hierarchy of memory timescales:
- Level 0: Updates frequently (fast-changing patterns, working memory)
- Level 1: Updates less frequently (medium-term patterns)
- Level N: Updates rarely (slow-changing patterns, long-term memory)

The key insight: different information needs different timescales to be
captured effectively. Fast-changing context (recent tokens) vs slow-changing
context (document topic, style) require different update frequencies.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System with true multi-frequency updates.

    Each MLP layer operates at a different timescale:
    - Fast layers (small chunk_size): Capture rapidly changing patterns
    - Slow layers (large chunk_size): Capture slowly evolving context

    The gradient accumulation mechanism ensures:
    - Gradients are accumulated over multiple steps
    - Updates are applied in batches according to chunk_size
    - This creates implicit temporal averaging at different scales

    Args:
        dim: Model dimension
        hidden_dim: Hidden dimension for MLPs
        num_levels: Number of frequency levels
        chunk_sizes: List of chunk sizes (update frequency) for each level
        activation: Activation function ('relu', 'gelu', 'silu')
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_levels: int = 3,
        chunk_sizes: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
        self.num_levels = num_levels

        # Default chunk sizes: exponentially increasing
        # Level 0: every 8 steps, Level 1: every 32 steps, etc.
        if chunk_sizes is None:
            chunk_sizes = [2 ** (i + 3) for i in range(num_levels)]
        assert len(chunk_sizes) == num_levels, \
            f"chunk_sizes length {len(chunk_sizes)} != num_levels {num_levels}"

        self.chunk_sizes = chunk_sizes

        # Create MLP for each frequency level
        self.mlps = nn.ModuleList()
        for i in range(num_levels):
            mlp = ContinuumMLP(
                dim=dim,
                hidden_dim=self.hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            self.mlps.append(mlp)

        # Learnable level mixing weights (optional)
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

        # Step counter (as buffer so it's saved with model)
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        # Gradient accumulators for each level
        # These store accumulated gradients between updates
        self._grad_accumulators: List[Optional[List[Optional[torch.Tensor]]]] = [
            None for _ in range(num_levels)
        ]
        self._accumulator_counts: List[int] = [0] * num_levels

    def get_update_levels(self, step: int) -> List[bool]:
        """
        Determine which levels should be updated at current step.

        Args:
            step: Current training step

        Returns:
            List of booleans indicating which levels should apply their
            accumulated gradients at this step
        """
        return [step % chunk_size == 0 for chunk_size in self.chunk_sizes]

    def accumulate_gradients(self, level: int):
        """
        Accumulate gradients for a specific level.

        This should be called after backward() but BEFORE optimizer.step().
        It stores the gradients so they can be averaged when the level updates.

        Args:
            level: Index of the CMS level (0 to num_levels-1)
        """
        if level < 0 or level >= self.num_levels:
            return

        mlp = self.mlps[level]

        # First accumulation: initialize storage
        if self._grad_accumulators[level] is None:
            self._grad_accumulators[level] = []
            for p in mlp.parameters():
                if p.grad is not None:
                    self._grad_accumulators[level].append(p.grad.clone())
                else:
                    self._grad_accumulators[level].append(None)
            self._accumulator_counts[level] = 1
        else:
            # Add to existing accumulator
            for i, p in enumerate(mlp.parameters()):
                if p.grad is not None:
                    if self._grad_accumulators[level][i] is None:
                        self._grad_accumulators[level][i] = p.grad.clone()
                    else:
                        self._grad_accumulators[level][i].add_(p.grad)
            self._accumulator_counts[level] += 1

    def apply_accumulated_gradients(self, level: int):
        """
        Apply accumulated gradients to a specific level.

        This should be called when it's time for this level to update
        (i.e., when step % chunk_size == 0).

        The accumulated gradients are averaged over the accumulation period,
        which provides temporal smoothing at the level's timescale.

        Args:
            level: Index of the CMS level
        """
        if level < 0 or level >= self.num_levels:
            return

        if self._grad_accumulators[level] is None:
            return

        mlp = self.mlps[level]
        count = max(self._accumulator_counts[level], 1)

        # Set the averaged gradient as the parameter's gradient
        # The optimizer will then apply the update
        for i, p in enumerate(mlp.parameters()):
            if self._grad_accumulators[level][i] is not None:
                # Average the accumulated gradients
                p.grad = self._grad_accumulators[level][i] / count

        # Clear accumulator for next period
        self._grad_accumulators[level] = None
        self._accumulator_counts[level] = 0

    def clear_all_accumulators(self):
        """Clear all gradient accumulators (e.g., at start of training)."""
        for level in range(self.num_levels):
            self._grad_accumulators[level] = None
            self._accumulator_counts[level] = 0

    def forward(self, x: torch.Tensor, use_residual: bool = True) -> torch.Tensor:
        """
        Forward pass through all MLP levels.

        Args:
            x: Input tensor (batch, seq_len, dim)
            use_residual: If True, use weighted residual connections (default).
                         If False, use true nested composition per Equation 30:
                         y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        if use_residual:
            # Weighted residual mode (original behavior, more stable for training)
            weights = torch.softmax(self.level_weights, dim=0)
            h = x
            for i, mlp in enumerate(self.mlps):
                h = h + weights[i] * mlp(h)
        else:
            # True nested composition per paper Equation 30:
            # y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))
            h = x
            for mlp in self.mlps:
                h = mlp(h)

        # Increment step counter during training
        if self.training:
            self.step_count += 1

        return h

    def forward_hierarchical(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning outputs from each level separately.

        Useful for analysis and debugging to see what each level contributes.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            List of outputs, one per level
        """
        outputs = []
        h = x

        for mlp in self.mlps:
            h = h + mlp(h)
            outputs.append(h.clone())

        return outputs

    def get_level_stats(self) -> dict:
        """Get statistics about each level for monitoring."""
        stats = {
            'step_count': self.step_count.item(),
            'levels': [],
        }

        for i, (mlp, chunk) in enumerate(zip(self.mlps, self.chunk_sizes)):
            level_stats = {
                'level': i,
                'chunk_size': chunk,
                'updates_so_far': self.step_count.item() // chunk,
                'weight': torch.softmax(self.level_weights, dim=0)[i].item(),
                'accumulated_steps': self._accumulator_counts[i],
            }

            # Parameter statistics
            total_params = sum(p.numel() for p in mlp.parameters())
            level_stats['num_params'] = total_params

            stats['levels'].append(level_stats)

        return stats


class ContinuumMLP(nn.Module):
    """
    Single MLP module used in CMS.

    Standard transformer-style feedforward with:
    - Up projection
    - Activation
    - Down projection
    - Optional dropout

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension
        activation: Activation function name
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str = 'gelu',
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize for small initial contribution
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Initialize output projection with small weights
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class FullyNestedCMS(nn.Module):
    """
    Fully nested variant of CMS where each level wraps the previous.

    Implements paper's Equation 30 exactly:
        y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))

    Each outer MLP processes the output of inner MLPs, creating
    a true hierarchy where faster levels provide input to slower ones.

    NOTE: This is the mathematically correct version per the paper.
    For training stability, you may want to use ContinuumMemorySystem
    with use_residual=True instead.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_levels: int = 3,
        chunk_sizes: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_residual: bool = False,  # Default False for paper-faithful behavior
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
        self.num_levels = num_levels
        self.use_residual = use_residual

        if chunk_sizes is None:
            chunk_sizes = [2 ** (i + 3) for i in range(num_levels)]
        self.chunk_sizes = chunk_sizes

        # Create nested MLPs
        self.mlps = nn.ModuleList()
        for i in range(num_levels):
            mlp = ContinuumMLP(
                dim=dim,
                hidden_dim=self.hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            self.mlps.append(mlp)

        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through nested MLPs.

        Implements true nested composition per Equation 30:
            y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))
        """
        h = x
        for mlp in self.mlps:
            if self.use_residual:
                h = h + mlp(h)  # Residual mode (more stable)
            else:
                h = mlp(h)  # True nesting per paper

        if self.training:
            self.step_count += 1

        return h
