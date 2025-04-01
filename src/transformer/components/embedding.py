"""
Word embedding components for TransformerNMT.
"""
from typing import Optional

import torch
import torch.nn as nn

from src.transformer.components.position import SinusoidalPositionEmbedding


class TokenEmbedding(nn.Embedding):
    """
    Token embedding module that converts token indices to dense vector representations.
    
    Extends the PyTorch Embedding layer with specific parameters for transformer models.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize token embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimensionality of the embeddings
        """
        # Set padding_idx=1 assuming <pad> token has index 1
        super().__init__(vocab_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding module for TransformerNMT.
    
    Combines token embeddings with positional encodings, applies 
    layer normalization, and dropout as described in the "Attention is All You Need" paper.
    """
    
    def __init__(
        self, 
        d_model: int, 
        vocab_size: int, 
        max_len: int, 
        drop_prob: float, 
        device: torch.device
    ):
        """
        Initialize the transformer embedding module.
        
        Args:
            d_model: Dimensionality of the model embeddings
            vocab_size: Size of the vocabulary
            max_len: Maximum sequence length
            drop_prob: Dropout probability
            device: Device to place the embeddings on (CPU/GPU)
        """
        super().__init__()
        
        # Token embeddings for converting word IDs to vectors
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        # Positional embeddings to capture position information
        self.position_embedding = SinusoidalPositionEmbedding(d_model, max_len, device)
        
        # Scale embeddings by sqrt(d_model) as per the paper
        self.embedding_scale = d_model ** 0.5
        
        # Regularization
        self.dropout = nn.Dropout(p=drop_prob)
        
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            Embedded representation with positional encodings [batch_size, seq_len, d_model]
        """
        # Get token embeddings and scale them
        token_embeddings = self.token_embedding(x) * self.embedding_scale
        
        # Get positional encodings
        positional_encodings = self.position_embedding(x)
        
        # Combine token embeddings with positional encodings
        embeddings = token_embeddings + positional_encodings
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings 