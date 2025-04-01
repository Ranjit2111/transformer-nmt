"""
Data handling utilities for the TransformerNMT model.
"""
from typing import Dict, List, Tuple, Iterator, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import IWSLT

from src.transformer.utils.tokenization import Tokenizer


class IWSLTDataset:
    """
    Handler for the IWSLT English-Vietnamese dataset.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        batch_size: int,
        device: torch.device,
        max_length: int = 256,
        min_freq: int = 2
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
        
    def _create_field(
        self, 
        tokenize_fn: callable, 
        is_source: bool
    ) -> Field:
        """
        Create a torchtext Field for either source or target language.
        
        Args:
            tokenize_fn: Function to tokenize the text
            is_source: Whether this is the source field
            
        Returns:
            Configured Field object
        """
        return Field(
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
        
    def _load_data(self) -> Tuple[Any, Any, Any]:
        """
        Load the IWSLT English-Vietnamese dataset.
        
        Returns:
            Tuple of (train, validation, test) datasets
        """
        print("Loading IWSLT English-Vietnamese dataset...")
        
        # Load IWSLT dataset with English and Vietnamese
        # The IWSLT dataset in torchtext uses the 2015 version by default
        fields = (self.source_field, self.target_field)
        
        train_data, valid_data, test_data = IWSLT.splits(
            exts=('.en', '.vi'),
            fields=fields,
            filter_pred=lambda x: len(vars(x)['src']) <= self.max_length and
                                  len(vars(x)['trg']) <= self.max_length
        )
        
        print(f"  Train: {len(train_data)} examples")
        print(f"  Validation: {len(valid_data)} examples")
        print(f"  Test: {len(test_data)} examples")
        
        return train_data, valid_data, test_data
        
    def _build_vocab(self):
        """
        Build vocabulary for source and target fields.
        """
        print("Building vocabulary...")
        
        # Build source vocabulary from training data
        self.source_field.build_vocab(
            self.train_data, 
            min_freq=self.min_freq
        )
        
        # Build target vocabulary from training data
        self.target_field.build_vocab(
            self.train_data, 
            min_freq=self.min_freq
        )
        
        print(f"  Source vocabulary size: {len(self.source_field.vocab)}")
        print(f"  Target vocabulary size: {len(self.target_field.vocab)}")
        
    def _create_iterators(self) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:
        """
        Create iterators for train, validation, and test datasets.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        print("Creating iterators...")
        
        # Use BucketIterator to batch sequences of similar lengths together
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            datasets=(self.train_data, self.valid_data, self.test_data),
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True,
            device=self.device
        )
        
        return train_iterator, valid_iterator, test_iterator
        
    def get_iterators(self) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:
        """
        Get train, validation, and test iterators.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        return self.train_iterator, self.valid_iterator, self.test_iterator 