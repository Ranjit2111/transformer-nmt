"""
Positional encoding implementation for TransformerNMT.
Based on the sinusoidal position encoding from "Attention is All You Need" paper.
"""
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding module as described in the original paper.
    
    This embedding adds positional information to the input embeddings
    using fixed sinusoidal functions at different frequencies.
    """

    def __init__(self, d_model: int, max_seq_length: int, device: torch.device):
        """
        Initialize the sinusoidal position embedding.
        
        Args:
            d_model: Dimensionality of the model embeddings
            max_seq_length: Maximum sequence length to support
            device: Device to store the embeddings (CPU/GPU)
        """
        super().__init__()
        
        # Initialize position encodings with zeros
        position_encodings = torch.zeros(max_seq_length, d_model, device=device)
        
        # We don't want to update these encodings during training
        position_encodings.requires_grad = False
        
        # Create position indices tensor [0, 1, 2, ..., max_seq_length-1]
        positions = torch.arange(0, max_seq_length, device=device).float()
        positions = positions.unsqueeze(1)  # Shape: [max_seq_length, 1]
        
        # Create dimension indices for applying the encoding formula
        # These are the "i" values in the formula 2i and 2i+1
        dimension_indices = torch.arange(0, d_model, step=2, device=device).float()
        
        # Apply the sinusoidal formula from the paper:
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        # Division term in the formula
        div_term = torch.exp(dimension_indices * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin to even indices
        position_encodings[:, 0::2] = torch.sin(positions * div_term)
        
        # Apply cos to odd indices
        position_encodings[:, 1::2] = torch.cos(positions * div_term)
        
        # Store encodings as a buffer (not a parameter)
        self.register_buffer('position_encodings', position_encodings)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get positional encodings for the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len]
                This is typically the token indices before embedding
        
        Returns:
            Position encodings of shape [seq_len, d_model]
        """
        batch_size, seq_len = x.size()
        
        # Return only the encodings for the actual sequence length
        return self.position_encodings[:seq_len, :] 