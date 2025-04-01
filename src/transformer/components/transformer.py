"""
Main TransformerNMT model implementation.
"""
from typing import Tuple

import torch
import torch.nn as nn

from src.transformer.components.encoder import TransformerEncoder
from src.transformer.components.decoder import TransformerDecoder


class TransformerNMT(nn.Module):
    """
    Transformer model for Neural Machine Translation as described in 
    "Attention is All You Need" (Vaswani et al., 2017).
    
    This is a complete encoder-decoder architecture for sequence-to-sequence tasks.
    """
    
    def __init__(
        self,
        src_pad_idx: int,
        trg_pad_idx: int,
        trg_sos_idx: int,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int,
        n_head: int,
        max_seq_length: int,
        ffn_hidden: int,
        n_layers: int,
        dropout: float,
        device: torch.device
    ):
        """
        Initialize the Transformer model.
        
        Args:
            src_pad_idx: Source padding token index
            trg_pad_idx: Target padding token index
            trg_sos_idx: Target start-of-sequence token index
            src_vocab_size: Size of the source vocabulary
            trg_vocab_size: Size of the target vocabulary
            d_model: Model dimensionality
            n_head: Number of attention heads
            max_seq_length: Maximum sequence length
            ffn_hidden: Hidden dimensionality in the feed-forward network
            n_layers: Number of encoder/decoder layers
            dropout: Dropout probability
            device: Device to place the model on (CPU/GPU)
        """
        super().__init__()
        
        # Store padding and start token indices
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        
        # Encoder component
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            max_len=max_seq_length,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout,
            device=device
        )
        
        # Decoder component
        self.decoder = TransformerDecoder(
            vocab_size=trg_vocab_size,
            max_len=max_seq_length,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout,
            device=device
        )
        
    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_len]
            trg: Target sequence tensor [batch_size, trg_seq_len]
                 Note: During training, this should be the target sequence shifted right
                 (excluding the last token and including the start token at the beginning)
                 
        Returns:
            Output logits tensor [batch_size, trg_seq_len, trg_vocab_size]
        """
        # Create padding masks
        src_mask = self._create_src_mask(src)
        trg_mask = self._create_trg_mask(trg)
        
        # Encode the source sequence
        enc_output = self.encoder(src, src_mask)
        
        # Decode with encoder output and both masks
        output = self.decoder(trg, enc_output, trg_mask, src_mask)
        
        return output
    
    def _create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create source padding mask to prevent attention to padding tokens.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_len]
            
        Returns:
            Source mask tensor [batch_size, 1, 1, src_seq_len]
        """
        # Create a binary mask: 1 for non-padding tokens, 0 for padding tokens
        # Shape: [batch_size, 1, 1, src_seq_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def _create_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Create target mask combining padding and look-ahead masks.
        
        The padding mask prevents attention to padding tokens.
        The look-ahead mask prevents attention to future tokens (causal attention).
        
        Args:
            trg: Target sequence tensor [batch_size, trg_seq_len]
            
        Returns:
            Target mask tensor [batch_size, 1, trg_seq_len, trg_seq_len]
        """
        # Create padding mask: [batch_size, 1, 1, trg_seq_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        
        # Create look-ahead mask (lower triangular matrix): [trg_seq_len, trg_seq_len]
        trg_seq_len = trg.size(1)
        trg_look_ahead_mask = torch.tril(
            torch.ones(trg_seq_len, trg_seq_len, device=self.device)
        ).bool()
        
        # Combine both masks using logical AND (both must be 1 for attention to be allowed)
        # trg_pad_mask: [batch_size, 1, 1, trg_seq_len] -> [batch_size, 1, trg_seq_len, trg_seq_len]
        # trg_look_ahead_mask: [trg_seq_len, trg_seq_len]
        trg_mask = trg_pad_mask & trg_look_ahead_mask
        
        return trg_mask 