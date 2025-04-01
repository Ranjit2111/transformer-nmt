"""
Training script for TransformerNMT model.
"""
import os
import argparse
import torch

from src.config import params
from src.transformer.components.transformer import TransformerNMT
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.data_handling import IWSLTDataset
from src.transformer.utils.training_utils import Trainer


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
    
    # Load dataset
    dataset = IWSLTDataset(
        tokenizer=tokenizer,
        batch_size=hyperparams.batch_size,
        device=device,
        max_length=hyperparams.max_seq_length,
        min_freq=hyperparams.min_freq
    )
    
    # Get iterators and special token indices
    train_iterator, valid_iterator, test_iterator = dataset.get_iterators()
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
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        src_pad_idx=src_pad_idx,
        lr=hyperparams.init_lr,
        factor=hyperparams.factor,
        patience=hyperparams.patience,
        warmup_steps=hyperparams.warmup_steps,
        clip=hyperparams.clip,
        weight_decay=hyperparams.weight_decay,
        adam_eps=hyperparams.adam_eps,
        fp16=hyperparams.fp16,
        grad_accumulation_steps=hyperparams.gradient_accumulation_steps,
        save_dir="results/checkpoints"
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    history = trainer.train(
        train_iterator=train_iterator,
        valid_iterator=valid_iterator,
        epochs=hyperparams.epochs
    )
    
    # Evaluate on test set
    test_loss = trainer.evaluate(test_iterator)
    print(f"\nTest Loss: {test_loss:.4f} | Test PPL: {torch.exp(torch.tensor(test_loss)):7.3f}")


if __name__ == "__main__":
    main() 