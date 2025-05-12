"""
Attention mechanisms for the TransformerNMT architecture.

These include the ScaledDotProductAttention and MultiHeadAttentionLayer components
as described in the "Attention is All You Need" paper.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in the original paper.
    
    Computes attention weights by taking the scaled dot product of queries and keys,
    and then applies these weights to values.
    
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    """
    
    def __init__(self):
        """Initialize the scaled dot-product attention module."""
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        epsilon: float = 1e-12
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, n_heads, seq_len, d_k]
            key: Key tensor [batch_size, n_heads, seq_len, d_k]
            value: Value tensor [batch_size, n_heads, seq_len, d_k]
            mask: Optional mask tensor [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
            epsilon: Small constant for numerical stability
            
        Returns:
            Tuple of:
                - Output tensor after attention [batch_size, n_heads, seq_len, d_k]
                - Attention scores [batch_size, n_heads, seq_len, seq_len]
        """
        # Get shape information
        batch_size, n_heads, seq_len, d_k = key.size()
        
        # Compute dot product between query and key
        # Transpose key for matrix multiplication: [batch_size, n_heads, d_k, seq_len]
        key_transposed = key.transpose(2, 3)
        
        # Compute attention scores: [batch_size, n_heads, seq_len, seq_len]
        # Scaling by sqrt(d_k) as in the paper to prevent vanishing gradients
        attention_scores = (query @ key_transposed) / math.sqrt(d_k)
        
        # Apply mask if provided (useful for padding and causal/autoregressive attention)
        if mask is not None:
            # Set masked positions to a large negative value before softmax
            # so they evaluate to ~0 after softmax
            # Using a smaller constant to avoid fp16 overflow
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        
        # Apply softmax to get attention weights (probabilities summing to 1)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention weights to values
        output = attention_weights @ value
        
        # Return both the output and the attention weights for visualization/analysis
        return output, attention_weights


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention layer as described in the "Attention is All You Need" paper.
    
    Splits the queries, keys, and values into multiple heads to enable the model
    to jointly attend to information from different representation subspaces.
    """
    
    def __init__(self, d_model: int, n_head: int):
        """
        Initialize multi-head attention layer.
        
        Args:
            d_model: Model dimensionality
            n_head: Number of attention heads
        """
        super().__init__()
        
        # Store number of heads
        self.n_head = n_head
        
        # Dimension of each head should divide d_model evenly
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        # Initialize the scaled dot-product attention
        self.attention = ScaledDotProductAttention()
        
        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # Final linear projection for the output
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            Output tensor after multi-head attention [batch_size, seq_len, d_model]
        """
        # Apply linear projections
        query = self.query_projection(query)  # [batch_size, seq_len, d_model]
        key = self.key_projection(key)        # [batch_size, seq_len, d_model]
        value = self.value_projection(value)  # [batch_size, seq_len, d_model]
        
        # Split into multiple heads
        query = self._split_heads(query)  # [batch_size, n_head, seq_len, d_k]
        key = self._split_heads(key)      # [batch_size, n_head, seq_len, d_k]
        value = self._split_heads(value)  # [batch_size, n_head, seq_len, d_k]
        
        # Apply scaled dot-product attention
        output, _ = self.attention(query, key, value, mask)  # [batch_size, n_head, seq_len, d_k]
        
        # Combine heads and project
        output = self._combine_heads(output)  # [batch_size, seq_len, d_model]
        output = self.output_projection(output)  # [batch_size, seq_len, d_model]
        
        return output
        
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_head, d_k).
        Transpose the result to [batch_size, n_head, seq_len, d_k].
        
        Args:
            tensor: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Reshaped tensor [batch_size, n_head, seq_len, d_k]
        """
        batch_size, seq_len, d_model = tensor.size()
        d_k = d_model // self.n_head
        
        # Reshape to [batch_size, seq_len, n_head, d_k]
        tensor = tensor.view(batch_size, seq_len, self.n_head, d_k)
        
        # Transpose to [batch_size, n_head, seq_len, d_k]
        # This puts the head dimension next to batch for easier computation
        tensor = tensor.transpose(1, 2)
        
        return tensor
        
    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Combine the multiple heads back into original shape.
        Transpose and reshape [batch_size, n_head, seq_len, d_k] to [batch_size, seq_len, d_model].
        
        Args:
            tensor: Input tensor [batch_size, n_head, seq_len, d_k]
            
        Returns:
            Reshaped tensor [batch_size, seq_len, d_model]
        """
        batch_size, n_head, seq_len, d_k = tensor.size()
        d_model = n_head * d_k
        
        # Transpose back to [batch_size, seq_len, n_head, d_k]
        tensor = tensor.transpose(1, 2)
        
        # Reshape to [batch_size, seq_len, d_model]
        tensor = tensor.contiguous().view(batch_size, seq_len, d_model)
        
        return tensor 