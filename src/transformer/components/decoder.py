"""
Decoder components for the TransformerNMT architecture.
"""
from typing import Optional

import torch
import torch.nn as nn

from src.transformer.components.attention import MultiHeadAttentionLayer
from src.transformer.components.normalization import TransformerLayerNorm
from src.transformer.components.feed_forward import PointwiseFeedForwardNetwork
from src.transformer.components.embedding import TransformerEmbedding


class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block as described in the "Attention is All You Need" paper.
    
    Each decoder block consists of:
    1. Masked multi-head self-attention layer
    2. Add & Norm (residual connection + layer normalization)
    3. Multi-head encoder-decoder attention layer
    4. Add & Norm (residual connection + layer normalization)
    5. Position-wise feed-forward network
    6. Add & Norm (residual connection + layer normalization)
    """
    
    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float):
        """
        Initialize the decoder block.
        
        Args:
            d_model: Model dimensionality
            ffn_hidden: Hidden dimensionality in the feed-forward network
            n_head: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Masked multi-head self-attention layer
        self.self_attention = MultiHeadAttentionLayer(d_model=d_model, n_head=n_head)
        
        # First normalization layer
        self.norm1 = TransformerLayerNorm(feature_dim=d_model)
        
        # Dropout for regularization after self-attention
        self.dropout1 = nn.Dropout(p=dropout)
        
        # Multi-head encoder-decoder attention layer
        self.encoder_decoder_attention = MultiHeadAttentionLayer(d_model=d_model, n_head=n_head)
        
        # Second normalization layer
        self.norm2 = TransformerLayerNorm(feature_dim=d_model)
        
        # Dropout for regularization after encoder-decoder attention
        self.dropout2 = nn.Dropout(p=dropout)
        
        # Position-wise feed-forward network
        self.feed_forward = PointwiseFeedForwardNetwork(
            d_model=d_model, 
            hidden_dim=ffn_hidden, 
            dropout=dropout
        )
        
        # Third normalization layer
        self.norm3 = TransformerLayerNorm(feature_dim=d_model)
        
        # Dropout for regularization after feed-forward
        self.dropout3 = nn.Dropout(p=dropout)
        
    def forward(
        self, 
        decoder_input: torch.Tensor, 
        encoder_output: Optional[torch.Tensor], 
        target_mask: Optional[torch.Tensor] = None, 
        source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through the decoder block.
        
        Args:
            decoder_input: Decoder input tensor [batch_size, seq_len, d_model]
            encoder_output: Output from the encoder [batch_size, seq_len, d_model]
            target_mask: Target padding and look-ahead mask
            source_mask: Source padding mask
            
        Returns:
            Output tensor after processing [batch_size, seq_len, d_model]
        """
        # Save input for residual connection
        residual = decoder_input
        
        # 1. Masked self-attention layer (prevent attending to future tokens)
        x = self.self_attention(
            query=decoder_input, 
            key=decoder_input, 
            value=decoder_input, 
            mask=target_mask
        )
        
        # 2. Add (residual connection) & Normalize
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        
        # Only perform encoder-decoder attention if encoder output is provided
        if encoder_output is not None:
            # Save output from self-attention for residual connection
            residual = x
            
            # 3. Encoder-decoder attention layer
            x = self.encoder_decoder_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                mask=source_mask
            )
            
            # 4. Add (residual connection) & Normalize
            x = self.dropout2(x)
            x = self.norm2(x + residual)
        
        # Save output from attention layers for residual connection
        residual = x
        
        # 5. Position-wise feed-forward network
        x = self.feed_forward(x)
        
        # 6. Add (residual connection) & Normalize
        x = self.dropout3(x)
        x = self.norm3(x + residual)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder as described in the "Attention is All You Need" paper.
    
    The decoder consists of:
    1. Output embeddings (token + positional)
    2. A stack of N identical decoder blocks
    3. Linear layer for output projection
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
        Initialize the transformer decoder.
        
        Args:
            vocab_size: Size of the target vocabulary
            max_len: Maximum sequence length
            d_model: Model dimensionality
            ffn_hidden: Hidden dimensionality in the feed-forward network
            n_head: Number of attention heads
            n_layers: Number of decoder blocks
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
        
        # Stack of decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Linear layer for output projection to vocabulary size
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(
        self, 
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Process input through the full decoder.
        
        Args:
            target: Target sequence tensor [batch_size, seq_len]
            encoder_output: Output from the encoder [batch_size, seq_len, d_model]
            target_mask: Target padding and look-ahead mask
            source_mask: Source padding mask
            
        Returns:
            Output logits tensor [batch_size, seq_len, vocab_size]
        """
        # Convert token indices to embeddings
        x = self.embedding(target)
        
        # Pass through each decoder block in sequence
        for decoder_block in self.decoder_blocks:
            x = decoder_block(
                decoder_input=x,
                encoder_output=encoder_output,
                target_mask=target_mask,
                source_mask=source_mask
            )
            
        # Project to vocabulary size
        output = self.output_projection(x)
        
        return output 