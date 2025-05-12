#!/usr/bin/env python
"""
Unified script for running the TransformerNMT model.

This script serves as an entry point for training, evaluation and translation,
and works with both the legacy and modern torchtext APIs.
"""
import os
import sys
import argparse
import torch
import warnings

from src.config import params
from src.transformer.components.transformer import TransformerNMT
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.modern_data_handling import IWSLTDataset
from src.transformer.utils.training_utils import Trainer
from src.transformer.utils.translator import Translator

# Define USING_LEGACY = False since we've removed the legacy implementation
USING_LEGACY = False

# For compatibility with modern implementation
try:
    from src.transformer.utils.modern_data_handling import TranslationDataset
    HAS_MODERN = True
except ImportError:
    HAS_MODERN = False


def setup_model_and_dataset(args, hyperparams):
    """
    Set up the model and dataset.
    
    Args:
        args: Command line arguments
        hyperparams: Hyperparameters
        
    Returns:
        Tuple of (model, dataset, tokenizer)
    """
    # Set device
    device = hyperparams.device
    print(f"Using device: {device}")
    
    # Set up tokenizer
    tokenizer = Tokenizer()

    try:
        dataset = IWSLTDataset(
            tokenizer=tokenizer,
            batch_size=hyperparams.batch_size,
            device=device,
            max_length=hyperparams.max_seq_length,
            min_freq=hyperparams.min_freq
        )
        
        # Get special token indices
        src_pad_idx = dataset.pad_idx if hasattr(dataset, 'pad_idx') else dataset.source_field.vocab.stoi['<pad>']
        trg_pad_idx = dataset.pad_idx if hasattr(dataset, 'pad_idx') else dataset.target_field.vocab.stoi['<pad>']
        trg_sos_idx = dataset.sos_idx if hasattr(dataset, 'sos_idx') else dataset.target_field.vocab.stoi['<sos>']
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
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
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    return model, dataset, tokenizer


def train(args, hyperparams):
    """
    Train the TransformerNMT model.
    
    Args:
        args: Command line arguments
        hyperparams: Hyperparameters
    """
    # Set up model and dataset
    model, dataset, _ = setup_model_and_dataset(args, hyperparams)
    
    # Get special token indices and iterators
    trg_pad_idx = dataset.pad_idx if hasattr(dataset, 'pad_idx') else dataset.target_field.vocab.stoi['<pad>']
    train_iterator, valid_iterator, test_iterator = dataset.get_iterators()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=float(hyperparams.init_lr),
            betas=(0.9, 0.98),
            eps=float(hyperparams.adam_eps),
            weight_decay=float(hyperparams.weight_decay)
        ),
        criterion=torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx),
        scheduler=None,
        clip=float(hyperparams.clip),
        log_dir="results/logs",
        save_dir="results/checkpoints"
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    history = trainer.train(
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        pad_idx=trg_pad_idx,
        n_epochs=args.epochs or hyperparams.epochs,
        early_stopping_patience=hyperparams.patience
    )
    
    # Evaluate on test set
    test_loss, test_ppl = trainer.evaluate(test_iterator, trg_pad_idx)
    print(f"\nTest Loss: {test_loss:.4f} | Test PPL: {test_ppl:.4f}")


def evaluate(args, hyperparams):
    """
    Evaluate the TransformerNMT model.
    
    Args:
        args: Command line arguments
        hyperparams: Hyperparameters
    """
    # Set up model and dataset
    model, dataset, _ = setup_model_and_dataset(args, hyperparams)
    
    # Get special token indices and iterators
    trg_pad_idx = dataset.pad_idx if hasattr(dataset, 'pad_idx') else dataset.target_field.vocab.stoi['<pad>']
    _, _, test_iterator = dataset.get_iterators()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),  # Dummy optimizer, not used for evaluation
        criterion=torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx),
        scheduler=None,
        clip=1.0,
        log_dir=None,
        save_dir=None
    )
    
    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint)
    
    # Evaluate on test set
    test_loss, test_ppl = trainer.evaluate(test_iterator, trg_pad_idx)
    print(f"\nTest Loss: {test_loss:.4f} | Test PPL: {test_ppl:.4f}")


def translate(args, hyperparams):
    """
    Translate text using the TransformerNMT model.
    
    Args:
        args: Command line arguments
        hyperparams: Hyperparameters
    """
    # Set up model and dataset
    model, dataset, tokenizer = setup_model_and_dataset(args, hyperparams)
    
    # Create translator
    translator = Translator(
        model=model,
        tokenizer=tokenizer,
        source_vocab=dataset.source_field.vocab if hasattr(dataset, 'source_field') else dataset._dataset.source_field.vocab,
        target_vocab=dataset.target_field.vocab if hasattr(dataset, 'target_field') else dataset._dataset.target_field.vocab,
        max_length=hyperparams.max_seq_length,
        device=hyperparams.device
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=hyperparams.device)
    model.load_state_dict(checkpoint['model'])
    
    # Interactive mode
    if args.interactive:
        print("Interactive translation mode. Enter text to translate or 'q' to quit.")
        while True:
            text = input("\nEnglish: ")
            if text.lower() == 'q':
                break
            
            translation = translator.translate(text, beam_size=args.beam_size)
            print(f"Vietnamese: {translation}")
    
    # Single text translation
    elif args.text:
        translation = translator.translate(args.text, beam_size=args.beam_size)
        print(f"English: {args.text}")
        print(f"Vietnamese: {translation}")
    
    else:
        print("Please provide text to translate with --text or use --interactive mode.")


def main():
    """
    Main function for running the TransformerNMT model.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run TransformerNMT model")
    parser.add_argument("action", choices=["train", "evaluate", "translate"], help="Action to perform")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--interactive", action="store_true", help="Interactive translation mode")
    
    args = parser.parse_args()
    
    # Load hyperparameters
    if args.config:
        hyperparams = params.__class__(args.config)
    else:
        hyperparams = params
    
    # Check if we need to create directories
    if args.action == "train":
        os.makedirs("results/logs", exist_ok=True)
        os.makedirs("results/checkpoints", exist_ok=True)
    
    # Ensure checkpoint is provided for evaluation and translation
    if (args.action == "evaluate" or args.action == "translate") and not args.checkpoint:
        parser.error(f"The {args.action} action requires --checkpoint")
    
    # Run the requested action
    if args.action == "train":
        train(args, hyperparams)
    elif args.action == "evaluate":
        evaluate(args, hyperparams)
    elif args.action == "translate":
        translate(args, hyperparams)


if __name__ == "__main__":
    main() 