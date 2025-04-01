"""
Encoder components for the TransformerNMT architecture.
"""
from typing import Optional

import torch
import torch.nn as nn

from src.transformer.components.attention import MultiHeadAttentionLayer
from src.transformer.components.normalization import TransformerLayerNorm
from src.transformer.components.feed_forward import PointwiseFeedForwardNetwork
from src.transformer.components.embedding import TransformerEmbedding


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block as described in the "Attention is All You Need" paper.
    
    Each encoder block consists of:
    1. Multi-head self-attention layer
    2. Add & Norm (residual connection + layer normalization)
    3. Position-wise feed-forward network
    4. Add & Norm (residual connection + layer normalization)
    """
    
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float):
        """
        Initialize the encoder block.
        
        Args:
            d_model: Model dimensionality
            ffn_hidden: Hidden dimensionality in the feed-forward network
            n_head: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention layer
        self.attention = MultiHeadAttentionLayer(d_model=d_model, n_head=n_head)
        
        # First normalization layer
        self.norm1 = TransformerLayerNorm(feature_dim=d_model)
        
        # Dropout for regularization after attention
        self.dropout1 = nn.Dropout(p=dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = PointwiseFeedForwardNetwork(
            d_model=d_model, 
            hidden_dim=ffn_hidden, 
            dropout=dropout
        )
        
        # Second normalization layer
        self.norm2 = TransformerLayerNorm(feature_dim=d_model)
        
        # Dropout for regularization after feed-forward
        self.dropout2 = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through the encoder block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional source padding mask to avoid attending to padding tokens
            
        Returns:
            Output tensor after processing [batch_size, seq_len, d_model]
        """
        # Save input for residual connection
        residual = x
        
        # 1. Self-attention layer
        x = self.attention(query=x, key=x, value=x, mask=src_mask)
        
        # 2. Add (residual connection) & Normalize
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        # Save output of attention layer for next residual connection
        residual = x
        
        # 3. Position-wise feed-forward network
        x = self.feed_forward(x)
        
        # 4. Add (residual connection) & Normalize
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder as described in the "Attention is All You Need" paper.
    
    The encoder consists of:
    1. Input embeddings (token + positional)
    2. A stack of N identical encoder blocks
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        max_len: int, 
        d_model: int, 
        ffn_hidden: int, 
        n_head: int, 
        n_layers: int, 
        dropout: float, 
        device: torch.device
    ):
        """
        Initialize the transformer encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            max_len: Maximum sequence length
            d_model: Model dimensionality
            ffn_hidden: Hidden dimensionality in the feed-forward network
            n_head: Number of attention heads
            n_layers: Number of encoder blocks
            dropout: Dropout probability
            device: Device to place the model on (CPU/GPU)
        """
        super().__init__()
        
        # Embedding layer (token + positional)
        self.embedding = TransformerEmbedding(
            d_model=d_model,
            vocab_size=vocab_size,
            max_len=max_len,
            drop_prob=dropout,
            device=device
        )
        
        # Stack of encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through the full encoder.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            src_mask: Optional source padding mask to avoid attending to padding tokens
            
        Returns:
            Output tensor after encoding [batch_size, seq_len, d_model]
        """
        # Convert token indices to embeddings
        x = self.embedding(x)
        
        # Pass through each encoder block in sequence
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
            
        return x 