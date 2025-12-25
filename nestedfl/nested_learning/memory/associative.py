"""
Associative Memory Implementation

Based on Definition 1 from the paper:
    M* = argmin_M L~(M(K); V)

Associative memory maps keys K to values V. The optimization objective
determines the type of memory system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AssociativeMemory(nn.Module):
    """
    Base associative memory module that learns to map keys to values.

    Args:
        dim_key: Dimension of keys
        dim_value: Dimension of values
        objective: Objective function ('dot_product', 'l2', 'cosine')
        memory_size: Maximum memory capacity (optional)
    """

    def __init__(
        self,
        dim_key,
        dim_value,
        objective='dot_product',
        memory_size=None,
    ):
        super().__init__()
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.objective = objective
        self.memory_size = memory_size

        # Initialize memory matrix
        if memory_size is not None:
            # Explicit memory slots
            self.register_buffer('keys', torch.zeros(memory_size, dim_key))
            self.register_buffer('values', torch.zeros(memory_size, dim_value))
            self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        else:
            # Matrix-valued memory (like linear attention)
            self.register_buffer('memory', torch.zeros(dim_value, dim_key))

    def compute_similarity(self, query, keys):
        """Compute similarity based on objective."""
        if self.objective == 'dot_product':
            return torch.matmul(query, keys.T)
        elif self.objective == 'l2':
            # Negative L2 distance (higher is more similar)
            return -torch.cdist(query, keys, p=2)
        elif self.objective == 'cosine':
            query_norm = F.normalize(query, dim=-1)
            keys_norm = F.normalize(keys, dim=-1)
            return torch.matmul(query_norm, keys_norm.T)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def store(self, keys, values):
        """
        Store key-value pairs in memory.

        For matrix memory: M = M + v * k^T (gradient descent step)
        For slot memory: Store in explicit slots
        """
        if self.memory_size is not None:
            # Store in slots (FIFO or other policy)
            batch_size = keys.shape[0]
            ptr = int(self.memory_ptr.item())

            # Wrap around if necessary
            if ptr + batch_size > self.memory_size:
                # Store what fits, wrap around for the rest
                space_left = self.memory_size - ptr
                self.keys[ptr:] = keys[:space_left]
                self.values[ptr:] = values[:space_left]

                remaining = batch_size - space_left
                self.keys[:remaining] = keys[space_left:]
                self.values[:remaining] = values[space_left:]

                self.memory_ptr[0] = remaining
            else:
                self.keys[ptr:ptr + batch_size] = keys
                self.values[ptr:ptr + batch_size] = values
                self.memory_ptr[0] = ptr + batch_size
        else:
            # Matrix-valued memory update
            # M_{t+1} = M_t + sum(v_i * k_i^T)
            update = torch.matmul(values.T, keys)
            self.memory += update

    def retrieve(self, query):
        """
        Retrieve values given query.

        Returns: M(query) = memory * query
        """
        if self.memory_size is not None:
            # Retrieve from slots using similarity
            similarities = self.compute_similarity(query, self.keys)
            weights = F.softmax(similarities, dim=-1)
            return torch.matmul(weights, self.values)
        else:
            # Matrix-valued retrieval
            return torch.matmul(query, self.memory.T)

    def forward(self, query, keys=None, values=None, store=True):
        """
        Forward pass: optionally store and retrieve.

        Args:
            query: Query tensor for retrieval
            keys: Keys to store (optional)
            values: Values to store (optional)
            store: Whether to store keys/values
        """
        if store and keys is not None and values is not None:
            self.store(keys, values)

        return self.retrieve(query)


class LinearAttention(nn.Module):
    """
    Linear attention as associative memory.

    Based on Equations 12-14 from the paper:
        k_t = x_t W_k, v_t = x_t W_v, q_t = x_t W_q
        M_t = M_{t-1} + v_t * k_t^T
        y_t = M_t * q_t

    This is the "unnormalized" linear attention formulation.

    IMPORTANT: Memory is per-sequence, not shared across batch.
    Each sequence in the batch maintains its own memory state.
    """

    def __init__(
        self,
        dim,
        dim_key=None,
        dim_value=None,
        normalize=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_key = dim_key or dim
        self.dim_value = dim_value or dim
        self.normalize = normalize

        # Projection matrices (slow weights in NL perspective)
        self.W_q = nn.Linear(dim, self.dim_key, bias=False)
        self.W_k = nn.Linear(dim, self.dim_key, bias=False)
        self.W_v = nn.Linear(dim, self.dim_value, bias=False)

        # Output projection
        self.W_o = nn.Linear(self.dim_value, dim, bias=False)

        # Persistent memory for stateful processing (optional, for RNN-like use)
        # Set to None by default; use set_persistent_memory() to enable
        self._persistent_memory = None

    def reset_memory(self):
        """Reset the persistent memory state (if enabled)."""
        if self._persistent_memory is not None:
            self._persistent_memory.zero_()

    def set_persistent_memory(self, batch_size: int, device=None, dtype=None):
        """
        Enable persistent memory for RNN-like stateful processing.

        Args:
            batch_size: Number of sequences to track
            device: Device for memory tensor
            dtype: Data type for memory tensor
        """
        self._persistent_memory = torch.zeros(
            batch_size, self.dim_value, self.dim_key,
            device=device, dtype=dtype
        )

    def forward(self, x, reset_memory=False, return_memory=False):
        """
        Forward pass of linear attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            reset_memory: Whether to reset persistent memory before processing
            return_memory: Whether to return final memory state

        Returns:
            Output tensor of shape (batch, seq_len, dim)
            If return_memory=True: (output, memory) where memory is (batch, dim_value, dim_key)
        """
        if reset_memory:
            self.reset_memory()

        batch_size, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        # Project to keys, values, queries
        queries = self.W_q(x)  # (batch, seq_len, dim_key)
        keys = self.W_k(x)     # (batch, seq_len, dim_key)
        values = self.W_v(x)   # (batch, seq_len, dim_value)

        # Initialize per-sequence memory
        # Use persistent memory if available, otherwise start fresh
        if self._persistent_memory is not None and self._persistent_memory.shape[0] == batch_size:
            memory = self._persistent_memory.clone()
        else:
            memory = torch.zeros(
                batch_size, self.dim_value, self.dim_key,
                device=device, dtype=dtype
            )

        outputs = []

        for t in range(seq_len):
            # Get current token's keys, values, queries
            k_t = keys[:, t]      # (batch, dim_key)
            v_t = values[:, t]    # (batch, dim_value)
            q_t = queries[:, t]   # (batch, dim_key)

            # Update memory: M_t = M_{t-1} + v_t * k_t^T
            # Per-sequence update (NOT batch-averaged!)
            # memory: (batch, dim_value, dim_key)
            # v_t: (batch, dim_value), k_t: (batch, dim_key)
            memory_update = torch.einsum('bv,bk->bvk', v_t, k_t)
            memory = memory + memory_update

            # Retrieve: y_t = M_t * q_t
            # memory: (batch, dim_value, dim_key), q_t: (batch, dim_key)
            y_t = torch.einsum('bvk,bk->bv', memory, q_t)

            if self.normalize:
                # Normalized version (like softmax attention)
                # Compute normalization factor per sequence
                # Sum over value dimension of memory @ query
                norm = torch.einsum('bvk,bk->b', memory.abs(), q_t.abs()) + 1e-6
                y_t = y_t / norm.unsqueeze(-1)

            outputs.append(y_t)

        # Update persistent memory if enabled
        if self._persistent_memory is not None:
            self._persistent_memory = memory.detach()

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, dim_value)

        # Project to output dimension
        output = self.W_o(output)

        if return_memory:
            return output, memory
        return output


class DeltaRuleMemory(nn.Module):
    """
    Associative memory with delta-rule update.

    Uses L2 regression objective instead of dot-product similarity.
    This provides better capacity management.
    """

    def __init__(self, dim_key, dim_value, learning_rate=0.1):
        super().__init__()
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.learning_rate = learning_rate

        # Memory matrix
        self.register_buffer('memory', torch.zeros(dim_value, dim_key))

    def forward(self, keys, values, queries):
        """
        Forward pass with delta-rule update.

        Args:
            keys: (batch, seq_len, dim_key)
            values: (batch, seq_len, dim_value)
            queries: (batch, seq_len, dim_key)
        """
        batch_size, seq_len, _ = keys.shape
        outputs = []

        for t in range(seq_len):
            k_t = keys[:, t]      # (batch, dim_key)
            v_t = values[:, t]    # (batch, dim_value)
            q_t = queries[:, t]   # (batch, dim_key)

            # Predict using current memory
            y_pred = torch.einsum('dk,bk->bd', self.memory, k_t)

            # Compute error (delta)
            error = v_t - y_pred

            # Delta-rule update: M = M + lr * error * k^T
            update = torch.einsum('bd,bk->dk', error, k_t) / batch_size
            self.memory += self.learning_rate * update

            # Retrieve for query
            y_t = torch.einsum('dk,bk->bd', self.memory, q_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)