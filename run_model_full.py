#!/usr/bin/env python
"""
Unified script for running the TransformerNMT model with the full IWSLT dataset.

This script uses the manually downloaded IWSLT'15 English-Vietnamese dataset and
serves as an entry point for training, evaluation and translation.
"""
import os
import sys
import argparse
import torch
import warnings
import time

from src.config import params
from src.transformer.components.transformer import TransformerNMT
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.training_utils import Trainer
from src.transformer.utils.modern_data_handling import TranslationDataset, ModernField, ModernBucketIterator
from src.transformer.utils.translator import Translator


class CustomIWSLTDataset:
    """
    Handler for the manually downloaded IWSLT English-Vietnamese dataset.
    """
    
    def __init__(
        self,
        tokenizer,
        batch_size,
        device,
        max_length=256,
        min_freq=2
    ):
        """
        Initialize the IWSLT dataset handler.
        
        Args:
            tokenizer: Tokenizer for English and Vietnamese
            batch_size: Batch size for iterators
            device: Device to place tensors on (CPU/GPU)
            max_length: Maximum sequence length to use
            min_freq: Minimum frequency for including words in vocabulary
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        
        # Handle device
        if isinstance(device, str):
            try:
                self.device = torch.device(device)
            except:
                self.device = device
        else:
            self.device = device
            
        self.max_length = max_length
        self.min_freq = min_freq
        
        # Initialize fields
        self.source_field = self._create_field(
            tokenize_fn=tokenizer.tokenize_en,
            is_source=True
        )
        
        self.target_field = self._create_field(
            tokenize_fn=tokenizer.tokenize_vi,
            is_source=False
        )
        
        # Load dataset
        self.train_data, self.valid_data, self.test_data = self._load_data()
        
        # Build vocabulary
        self._build_vocab()
        
        # Create iterators
        self.train_iterator, self.valid_iterator, self.test_iterator = self._create_iterators()
        
        # Store special token indices
        self.pad_idx = self.target_field.vocab.stoi['<pad>']
        self.sos_idx = self.target_field.vocab.stoi['<sos>']
        self.eos_idx = self.target_field.vocab.stoi['<eos>']
        
        # Store vocabulary sizes
        self.source_vocab_size = len(self.source_field.vocab)
        self.target_vocab_size = len(self.target_field.vocab)
        
    def _create_field(self, tokenize_fn, is_source):
        """
        Create a ModernField for either source or target language.
        
        Args:
            tokenize_fn: Function to tokenize the text
            is_source: Whether this is the source field
            
        Returns:
            Configured ModernField object
        """
        return ModernField(
            tokenize=tokenize_fn,
            init_token='<sos>' if not is_source else None,
            eos_token='<eos>',
            pad_token='<pad>',
            lower=True,
            batch_first=True,
            include_lengths=False,
            fix_length=self.max_length,
            use_vocab=True
        )
        
    def _load_data(self):
        """
        Load the manually downloaded IWSLT English-Vietnamese dataset.
        
        Returns:
            Tuple of (train, validation, test) datasets
        """
        print("Loading manually downloaded IWSLT English-Vietnamese dataset...")
        
        # Define file paths
        dataset_dir = os.path.join('data', "IWSLT'15 en-vi")
        train_en_path = os.path.join(dataset_dir, 'train.en.txt')
        train_vi_path = os.path.join(dataset_dir, 'train.vi.txt')
        valid_en_path = os.path.join(dataset_dir, 'tst2012.en.txt')
        valid_vi_path = os.path.join(dataset_dir, 'tst2012.vi.txt')
        test_en_path = os.path.join(dataset_dir, 'tst2013.en.txt')
        test_vi_path = os.path.join(dataset_dir, 'tst2013.vi.txt')
        
        # Check if all files exist
        all_paths = [train_en_path, train_vi_path, valid_en_path, valid_vi_path, test_en_path, test_vi_path]
        all_exist = all(os.path.exists(path) for path in all_paths)
        
        if not all_exist:
            missing_files = [path for path in all_paths if not os.path.exists(path)]
            raise FileNotFoundError(f"Missing dataset files: {missing_files}")
        
        # Read data
        def read_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        
        # Load all data
        train_src = read_file(train_en_path)
        train_tgt = read_file(train_vi_path)
        valid_src = read_file(valid_en_path)
        valid_tgt = read_file(valid_vi_path)
        test_src = read_file(test_en_path)
        test_tgt = read_file(test_vi_path)
        
        # Create datasets
        train_data = TranslationDataset(
            train_src, train_tgt, 
            self.source_field, self.target_field
        )
        
        valid_data = TranslationDataset(
            valid_src, valid_tgt, 
            self.source_field, self.target_field
        )
        
        test_data = TranslationDataset(
            test_src, test_tgt, 
            self.source_field, self.target_field
        )
        
        # Print dataset sizes
        print(f"  Train: {len(train_data)} examples")
        print(f"  Validation: {len(valid_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        
        return train_data, valid_data, test_data
        
    def _build_vocab(self):
        """
        Build vocabulary for source and target fields.
        """
        print("Building vocabulary...")
        
        # Get raw training data as strings
        train_attrs = self.train_data.get_attrs()
        print(f"Number of training examples: {len(train_attrs)}")
        
        # Tokenize the data explicitly before building vocabulary
        print("Tokenizing training data...")
        # Source data tokenization
        tokenized_src = []
        for example in train_attrs:
            if isinstance(example.src, str):
                tokens = self.source_field.tokenize(example.src)
                tokenized_src.append(tokens)
            else:
                tokenized_src.append(example.src)  # Already tokenized
                
        # Target data tokenization
        tokenized_tgt = []
        for example in train_attrs:
            if isinstance(example.trg, str):
                tokens = self.target_field.tokenize(example.trg)
                tokenized_tgt.append(tokens)
            else:
                tokenized_tgt.append(example.trg)  # Already tokenized
        
        # Show samples
        if tokenized_src:
            print(f"Sample tokenized source (first 10 tokens): {tokenized_src[0][:10]}")
        if tokenized_tgt:
            print(f"Sample tokenized target (first 10 tokens): {tokenized_tgt[0][:10]}")
        
        # Build source vocabulary from tokenized data
        print("Building source vocabulary...")
        self.source_field.build_vocab(
            tokenized_src, 
            min_freq=self.min_freq
        )
        
        # Build target vocabulary from tokenized data
        print("Building target vocabulary...")
        self.target_field.build_vocab(
            tokenized_tgt, 
            min_freq=self.min_freq
        )
        
        print(f"  Source vocabulary size: {len(self.source_field.vocab)}")
        print(f"  Target vocabulary size: {len(self.target_field.vocab)}")
        print(f"  Min frequency: {self.min_freq}")
        
    def _create_iterators(self):
        """
        Create iterators for train, validation, and test datasets.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        print("Creating iterators...")
        
        # Create iterators
        train_iterator = ModernBucketIterator(
            self.train_data,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x[0]),
            sort_within_batch=True,
            device=self.device
        )
        
        valid_iterator = ModernBucketIterator(
            self.valid_data,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x[0]),
            sort_within_batch=True,
            device=self.device
        )
        
        test_iterator = ModernBucketIterator(
            self.test_data,
            batch_size=self.batch_size,
            sort_key=lambda x: len(x[0]),
            sort_within_batch=True,
            device=self.device
        )
        
        return train_iterator, valid_iterator, test_iterator
        
    def get_iterators(self):
        """
        Get train, validation, and test iterators.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        return self.train_iterator, self.valid_iterator, self.test_iterator


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
        # Use our custom dataset loader that points to the manually downloaded files
        dataset = CustomIWSLTDataset(
            tokenizer=tokenizer,
            batch_size=hyperparams.batch_size,
            device=device,
            max_length=hyperparams.max_seq_length,
            min_freq=hyperparams.min_freq
        )
        
        # Get special token indices
        src_pad_idx = dataset.pad_idx
        trg_pad_idx = dataset.pad_idx
        trg_sos_idx = dataset.sos_idx
            
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
    trg_pad_idx = dataset.pad_idx
    train_iterator, valid_iterator, test_iterator = dataset.get_iterators()
    
    # Calculate expected training time
    print("\nCalculating expected training time for full training...")
    batch_size = hyperparams.batch_size
    n_batches_per_epoch = len(train_iterator)
    n_examples = len(dataset.train_data)
    
    # Time a single batch
    model.train()
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=float(hyperparams.init_lr))
    
    # Get a batch from the iterator
    for batch in train_iterator:
        start_time = time.time()
        
        src = batch.src
        trg = batch.trg
        
        dummy_optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        
        # Just do a forward pass without backward
        end_time = time.time()
        batch_time = end_time - start_time
        break
    
    # Calculate estimates
    epoch_time_estimate = batch_time * n_batches_per_epoch
    total_time_estimate = epoch_time_estimate * hyperparams.epochs
    
    # Convert to reasonable units
    if total_time_estimate < 60:
        time_str = f"{total_time_estimate:.2f} seconds"
    elif total_time_estimate < 3600:
        time_str = f"{total_time_estimate / 60:.2f} minutes"
    elif total_time_estimate < 86400:
        time_str = f"{total_time_estimate / 3600:.2f} hours"
    else:
        time_str = f"{total_time_estimate / 86400:.2f} days"
    
    print(f"Dataset size: {n_examples} examples")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {n_batches_per_epoch}")
    print(f"Estimated time per epoch: {epoch_time_estimate:.2f} seconds")
    print(f"Estimated total training time for {hyperparams.epochs} epochs: {time_str}")
    
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
    trg_pad_idx = dataset.pad_idx
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
        source_vocab=dataset.source_field.vocab,
        target_vocab=dataset.target_field.vocab,
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
    Main function for running the TransformerNMT model with the full dataset.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run TransformerNMT model with full IWSLT dataset")
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