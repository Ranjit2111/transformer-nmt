"""
Translation script for TransformerNMT model.
"""
import os
import argparse
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.config import params
from src.transformer.components.transformer import TransformerNMT
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.data_handling import IWSLTDataset


class Translator:
    """
    Translator class for TransformerNMT model inference.
    """
    
    def __init__(
        self,
        model: TransformerNMT,
        tokenizer: Tokenizer,
        source_vocab: dict,
        target_vocab: dict,
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
        
        # Special token indices
        self.sos_idx = self.target_vocab.stoi['<sos>']
        self.eos_idx = self.target_vocab.stoi['<eos>']
        
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
        
        # Convert tokens to indices
        token_indices = [self.source_vocab.stoi.get(token, self.source_vocab.stoi['<unk>']) for token in tokens]
        
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
            best_translation = translations[0][0]  # Get tokens from best hypothesis
            
            # Convert token indices to tokens
            tokens = [self.target_vocab.itos[idx] for idx in best_translation if idx != self.eos_idx 
                     and idx != self.sos_idx]
            
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
            tokens = [self.target_vocab.itos[idx] for idx in trg_indexes if idx != self.eos_idx 
                    and idx != self.sos_idx]
            
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
        k = beam_size
        
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
                top_probs, top_indices = probs.topk(k)
                
                # Add new candidates
                for i in range(k):
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


def main():
    """
    Main function for translation with the TransformerNMT model.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Translate with TransformerNMT model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--interactive", action="store_true", help="Interactive translation mode")
    args = parser.parse_args()
    
    # Load hyperparameters
    if args.config:
        hyperparams = params.__class__(args.config)
    else:
        hyperparams = params
    
    # Set device
    device = hyperparams.device
    print(f"Using device: {device}")
    
    # Set up tokenizer
    tokenizer = Tokenizer()
    
    # Load dataset to get the vocabularies
    dataset = IWSLTDataset(
        tokenizer=tokenizer,
        batch_size=hyperparams.batch_size,
        device=device,
        max_length=hyperparams.max_seq_length,
        min_freq=hyperparams.min_freq
    )
    
    # Get special token indices
    src_pad_idx = dataset.source_field.vocab.stoi['<pad>']
    trg_pad_idx = dataset.target_field.vocab.stoi['<pad>']
    trg_sos_idx = dataset.target_field.vocab.stoi['<sos>']
    
    # Create model
    model = TransformerNMT(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        src_vocab_size=dataset.source_vocab_size,
        trg_vocab_size=dataset.target_vocab_size,
        d_model=hyperparams.d_model,
        n_head=hyperparams.n_heads,
        max_seq_length=hyperparams.max_seq_length,
        ffn_hidden=hyperparams.ffn_hidden,
        n_layers=hyperparams.n_layers,
        dropout=hyperparams.dropout,
        device=device
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {args.checkpoint}")
    
    # Create translator
    translator = Translator(
        model=model,
        tokenizer=tokenizer,
        source_vocab=dataset.source_field.vocab,
        target_vocab=dataset.target_field.vocab,
        max_length=hyperparams.max_seq_length,
        device=device
    )
    
    # Interactive mode
    if args.interactive:
        print("Interactive translation mode (Ctrl+C to exit):")
        try:
            while True:
                text = input("\nEnter English text: ")
                if not text.strip():
                    continue
                    
                translation = translator.translate(text, beam_size=args.beam_size)
                print(f"Vietnamese translation: {translation}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
    
    # Translate a single text
    elif args.text:
        translation = translator.translate(args.text, beam_size=args.beam_size)
        print(f"English: {args.text}")
        print(f"Vietnamese: {translation}")
        
    else:
        parser.error("Either --text or --interactive must be specified")


if __name__ == "__main__":
    main() 