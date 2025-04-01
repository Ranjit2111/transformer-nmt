"""
Feed-forward network implementation for TransformerNMT.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseFeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network as described in "Attention is All You Need".
    
    This is a simple fully connected feed-forward network applied to each position 
    separately and identically, consisting of two linear transformations with a ReLU
    activation in between.
    
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize the position-wise feed-forward network.
        
        Args:
            d_model: Input and output dimensionality
            hidden_dim: Dimensionality of the hidden layer
            dropout: Dropout probability to use after the first layer
        """
        super().__init__()
        
        # First linear transformation
        self.linear1 = nn.Linear(d_model, hidden_dim)
        
        # Second linear transformation
        self.linear2 = nn.Linear(hidden_dim, d_model)
        
        # ReLU activation function
        self.activation = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward network to the input.
        
        Args:
            x: Input tensor of shape [..., d_model]
            
        Returns:
            Output tensor of shape [..., d_model]
        """
        # First linear layer
        x = self.linear1(x)
        
        # Apply ReLU activation
        x = self.activation(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        return x 