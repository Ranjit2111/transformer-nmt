"""
Training script for TransformerNMT model.
"""
import os
import argparse
import torch
import warnings

from src.config import params
from src.transformer.components.transformer import TransformerNMT
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.data_handling import IWSLTDataset
from src.transformer.utils.training_utils import Trainer, USING_LEGACY

# For compatibility with modern implementation
try:
    from src.transformer.utils.modern_data_handling import TranslationDataset
    HAS_MODERN = True
except ImportError:
    HAS_MODERN = False


def main():
    """
    Main function for training the TransformerNMT model.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train TransformerNMT model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    args = parser.parse_args()
    
    # Load hyperparameters
    if args.config:
        hyperparams = params.__class__(args.config)
    else:
        hyperparams = params
    
    # Set epochs if provided
    if args.epochs:
        hyperparams.epochs = args.epochs
    
    # Set device
    device = hyperparams.device
    print(f"Using device: {device}")
    
    # Set up tokenizer
    tokenizer = Tokenizer()
    
    # Display API version in use
    if USING_LEGACY:
        print("Using legacy torchtext API")
    else:
        print("Using modern torchtext API")
    
    # Load dataset
    try:
        dataset = IWSLTDataset(
            tokenizer=tokenizer,
            batch_size=hyperparams.batch_size,
            device=device,
            max_length=hyperparams.max_seq_length,
            min_freq=hyperparams.min_freq
        )
        
        # Get iterators and special token indices
        train_iterator, valid_iterator, test_iterator = dataset.get_iterators()
        
        # Get pad, sos indices
        src_pad_idx = dataset.pad_idx if hasattr(dataset, 'pad_idx') else dataset.source_field.vocab.stoi['<pad>']
        trg_pad_idx = dataset.pad_idx if hasattr(dataset, 'pad_idx') else dataset.target_field.vocab.stoi['<pad>']
        trg_sos_idx = dataset.sos_idx if hasattr(dataset, 'sos_idx') else dataset.target_field.vocab.stoi['<sos>']
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if not USING_LEGACY and not HAS_MODERN:
            raise ImportError(
                "Neither torchtext.legacy nor modern implementation could be loaded. "
                "Please install torchtext==0.15.0 with torch==2.0.0, or ensure the modern implementation is available."
            )
        raise
    
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
        n_epochs=hyperparams.epochs,
        early_stopping_patience=hyperparams.patience
    )
    
    # Evaluate on test set
    test_loss, test_ppl = trainer.evaluate(test_iterator, trg_pad_idx)
    print(f"\nTest Loss: {test_loss:.4f} | Test PPL: {test_ppl:.4f}")


if __name__ == "__main__":
    main() 