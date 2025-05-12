"""
Translator utility for TransformerNMT model inference.
"""
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F

from src.transformer.components.transformer import TransformerNMT
from src.transformer.utils.tokenization import Tokenizer


class Translator:
    """
    Translator class for TransformerNMT model inference.
    """
    
    def __init__(
        self,
        model: TransformerNMT,
        tokenizer: Tokenizer,
        source_vocab: Any,
        target_vocab: Any,
        max_length: int = 256,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize the translator.
        
        Args:
            model: TransformerNMT model
            tokenizer: Tokenizer for processing text
            source_vocab: Source vocabulary (stoi)
            target_vocab: Target vocabulary (itos)
            max_length: Maximum sequence length
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_length = max_length
        self.device = device
        
        # Handle either vocabulary type (legacy or modern)
        if hasattr(self.target_vocab, 'stoi'):
            # Legacy vocabulary
            self.sos_idx = self.target_vocab.stoi['<sos>']
            self.eos_idx = self.target_vocab.stoi['<eos>']
            self.get_index = lambda vocab, token: vocab.stoi.get(token, vocab.stoi['<unk>'])
            self.get_token = lambda vocab, idx: vocab.itos[idx]
        else:
            # Modern vocabulary (ModernVocab from our implementation)
            self.sos_idx = self.target_vocab.get_stoi()['<sos>']
            self.eos_idx = self.target_vocab.get_stoi()['<eos>']
            self.get_index = lambda vocab, token: vocab.get_stoi().get(token, vocab.get_stoi()['<unk>'])
            self.get_token = lambda vocab, idx: vocab.get_itos()[idx]
        
    def translate(
        self,
        text: str,
        beam_size: int = 5,
        max_length: Optional[int] = None
    ) -> str:
        """
        Translate English text to Vietnamese.
        
        Args:
            text: English text to translate
            beam_size: Beam size for beam search (if 1, uses greedy decoding)
            max_length: Maximum output sequence length
            
        Returns:
            Translated text in Vietnamese
        """
        if max_length is None:
            max_length = self.max_length
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Tokenize the source text
        tokens = self.tokenizer.tokenize_en(text)
        
        # Convert tokens to indices - use the appropriate method based on vocabulary type
        token_indices = [self.get_index(self.source_vocab, token) for token in tokens]
        
        # Create a tensor and add batch dimension
        src_tensor = torch.LongTensor(token_indices).unsqueeze(0).to(self.device)
        
        # Create source mask
        src_mask = self.model._create_src_mask(src_tensor)
        
        # Encode the source sequence
        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor, src_mask)
        
        # Initialize target tensor with SOS token
        trg_indexes = [self.sos_idx]
        
        if beam_size > 1:
            # Use beam search for translation
            translations = self._beam_search(enc_src, src_mask, beam_size, max_length)
            best_translation = translations[0][1]  # Get tokens from best hypothesis
            
            # Convert token indices to tokens
            tokens = [self.get_token(self.target_vocab, idx) for idx in best_translation 
                     if idx != self.eos_idx and idx != self.sos_idx]
            
        else:
            # Use greedy decoding
            for i in range(max_length):
                # Convert target indices to tensor
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
                
                # Create target mask
                trg_mask = self.model._create_trg_mask(trg_tensor)
                
                # Pass through decoder
                with torch.no_grad():
                    output = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                
                # Get next token prediction
                pred_token = output.argmax(2)[:, -1].item()
                
                # Add predicted token to sequence
                trg_indexes.append(pred_token)
                
                # Stop if EOS token is predicted
                if pred_token == self.eos_idx:
                    break
            
            # Convert token indices to tokens (excluding SOS and EOS)
            tokens = [self.get_token(self.target_vocab, idx) for idx in trg_indexes 
                    if idx != self.eos_idx and idx != self.sos_idx]
            
        # Detokenize
        translation = self.tokenizer.detokenize_vi(tokens)
        
        return translation
    
    def _beam_search(
        self,
        enc_src: torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int,
        max_length: int
    ) -> List:
        """
        Perform beam search for translation.
        
        Args:
            enc_src: Encoded source sequence
            src_mask: Source mask
            beam_size: Beam size
            max_length: Maximum output sequence length
            
        Returns:
            List of (sequence, score) pairs, sorted by score
        """
        # Initialize with SOS token
        k = min(beam_size, len(self.target_vocab.stoi) - 1)  # Adjust k if vocabulary is smaller
        
        if k < 1:
            k = 1  # Ensure k is at least 1
            
        # Initial sequence: (score, sequence)
        sequences = [(0.0, [self.sos_idx])]
        
        # Expand until max length or all beams have ended
        for _ in range(max_length - 1):
            # Get all candidates from current sequences
            all_candidates = []
            
            # For each sequence in the beam
            for score, seq in sequences:
                # If the sequence has ended, keep it as a candidate
                if seq[-1] == self.eos_idx:
                    all_candidates.append((score, seq))
                    continue
                    
                # Convert sequence to tensor
                trg_tensor = torch.LongTensor(seq).unsqueeze(0).to(self.device)
                
                # Create target mask
                trg_mask = self.model._create_trg_mask(trg_tensor)
                
                # Pass through decoder
                with torch.no_grad():
                    output = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                
                # Get logits for the next token
                logits = output[0, -1, :]
                
                # Apply softmax to get probabilities
                probs = F.log_softmax(logits, dim=0)
                
                # Get top k candidates
                vocab_size = probs.shape[0]
                top_k = min(k, vocab_size)  # Ensure we don't request more candidates than vocab size
                top_probs, top_indices = probs.topk(top_k)
                
                # Add new candidates
                for i in range(top_k):
                    token_idx = top_indices[i].item()
                    token_prob = top_probs[i].item()
                    candidate = (score + token_prob, seq + [token_idx])
                    all_candidates.append(candidate)
                    
            # Select k best candidates
            sequences = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:k]
            
            # If all sequences end with EOS, stop early
            if all(s[-1] == self.eos_idx for _, s in sequences):
                break
                
        return sequences 