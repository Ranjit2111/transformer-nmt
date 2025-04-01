"""
Normalization layers for the TransformerNMT architecture.
"""
from typing import Optional

import torch
import torch.nn as nn


class TransformerLayerNorm(nn.Module):
    """
    Layer Normalization implementation for the Transformer.
    
    Normalizes the inputs across the last dimension (feature dimension),
    then applies learnable scale (gamma) and offset (beta) parameters.
    
    This is equivalent to PyTorch's nn.LayerNorm but implemented manually
    to maintain consistency with the original paper's implementation.
    """
    
    def __init__(self, feature_dim: int, epsilon: float = 1e-12):
        """
        Initialize layer normalization module.
        
        Args:
            feature_dim: Dimensionality of the input features
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        # Learnable scale parameter (initialized to ones)
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        
        # Learnable offset parameter (initialized to zeros)
        self.beta = nn.Parameter(torch.zeros(feature_dim))
        
        # Small constant for numerical stability
        self.epsilon = epsilon
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input.
        
        Args:
            x: Input tensor [..., feature_dim]
                The normalization is applied across the last dimension
                
        Returns:
            Normalized tensor of the same shape as input
        """
        # Compute mean and variance along the last dimension
        # keepdim=True to preserve the original tensor dimensions for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalize the input
        # (x - mean) / sqrt(var + epsilon)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)
        
        # Apply scale and offset
        # gamma * normalized + beta
        output = self.gamma * normalized + self.beta
        
        return output 