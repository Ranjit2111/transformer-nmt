"""
Data handling utilities for the TransformerNMT model.
"""
from typing import Dict, List, Tuple, Iterator, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader

# Define Field class to handle the case when legacy is not available
class Field:
    """Fallback Field class for when torchtext.legacy is not available."""
    def __init__(self, tokenize=None, init_token=None, eos_token=None, 
                 pad_token=None, lower=False, batch_first=False, 
                 include_lengths=False, fix_length=None, use_vocab=True):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.lower = lower
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.fix_length = fix_length
        self.use_vocab = use_vocab
        self.vocab = None
    
    def build_vocab(self, *args, min_freq=1, **kwargs):
        """Stub for vocab building when using new API"""
        pass

# Import correct modules based on available API
try:
    from torchtext.legacy.data import Field as LegacyField
    from torchtext.legacy.data import BucketIterator
    from torchtext.legacy.datasets import IWSLT
    USING_LEGACY = True
    Field = LegacyField  # Use the legacy Field
    print("Using torchtext.legacy API")
except ImportError:
    # Use newer torchtext API as fallback
    import warnings
    warnings.warn("Using new torchtext API. Data processing will use alternative implementation.")
    # Import the modern implementation
    try:
        from src.transformer.utils.modern_data_handling import (
            IWSLTDataset as ModernIWSLTDataset,
            ModernField,
            ModernBucketIterator,
            TranslationDataset
        )
        USING_LEGACY = False
        Field = ModernField  # Use the modern Field class
        print("Using modern torchtext API")
    except ImportError:
        # If even the modern implementation is not available, raise an error
        raise ImportError(
            "Neither torchtext.legacy nor modern implementation is available. "
            "Please install torchtext==0.15.0 with torch==2.0.0."
        )

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
        # For non-legacy implementation, use the modern dataset handler
        if not USING_LEGACY:
            self._dataset = ModernIWSLTDataset(
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=device,
                max_length=max_length,
                min_freq=min_freq
            )
            # Expose all attributes from the modern implementation
            self.tokenizer = self._dataset.tokenizer
            self.batch_size = self._dataset.batch_size
            self.device = self._dataset.device
            self.max_length = self._dataset.max_length
            self.min_freq = self._dataset.min_freq
            self.source_field = self._dataset.source_field
            self.target_field = self._dataset.target_field
            self.train_data = self._dataset.train_data
            self.valid_data = self._dataset.valid_data
            self.test_data = self._dataset.test_data
            self.train_iterator = self._dataset.train_iterator
            self.valid_iterator = self._dataset.valid_iterator
            self.test_iterator = self._dataset.test_iterator
            self.pad_idx = self._dataset.pad_idx
            self.sos_idx = self._dataset.sos_idx
            self.eos_idx = self._dataset.eos_idx
            self.source_vocab_size = self._dataset.source_vocab_size
            self.target_vocab_size = self._dataset.target_vocab_size
            return
            
        # Legacy implementation follows
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # Handle if device is a string instead of torch.device
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
        
        try:
            train_data, valid_data, test_data = IWSLT.splits(
                exts=('.en', '.vi'),
                fields=fields,
                filter_pred=lambda x: len(vars(x)['src']) <= self.max_length and
                                  len(vars(x)['trg']) <= self.max_length
            )
        except Exception as e:
            print(f"Error loading IWSLT dataset: {e}")
            print("Trying alternative loading method...")
            try:
                # Alternative loading method for newer torchtext versions
                train_data, valid_data, test_data = IWSLT.splits(
                    exts=('.en', '.vi'),
                    fields=fields
                )
            except Exception as inner_e:
                raise RuntimeError(f"Failed to load IWSLT dataset: {inner_e}")
        
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
        
    def _create_iterators(self) -> Tuple[Any, Any, Any]:
        """
        Create iterators for train, validation, and test datasets.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        print("Creating iterators...")
        
        # Use BucketIterator to batch sequences of similar lengths together
        try:
            device_arg = self.device
            # Convert string device to proper format if needed
            if isinstance(device_arg, str):
                if device_arg == "cpu":
                    device_arg = -1
                else:
                    try:
                        device_arg = torch.device(device_arg)
                    except:
                        device_arg = -1
                        
            train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
                datasets=(self.train_data, self.valid_data, self.test_data),
                batch_size=self.batch_size,
                sort_key=lambda x: len(x.src),
                sort_within_batch=True,
                device=device_arg
            )
            return train_iterator, valid_iterator, test_iterator
        except Exception as e:
            print(f"Error creating iterators: {e}")
            raise RuntimeError(f"Failed to create iterators: {e}")
        
    def get_iterators(self) -> Tuple[Any, Any, Any]:
        """
        Get train, validation, and test iterators.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        if not USING_LEGACY:
            # Use the modern implementation's get_iterators method
            return self._dataset.get_iterators()
            
        if self.train_iterator is None or self.valid_iterator is None or self.test_iterator is None:
            raise RuntimeError("Iterators have not been created successfully.")
        return self.train_iterator, self.valid_iterator, self.test_iterator 