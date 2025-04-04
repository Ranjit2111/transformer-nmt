#!/usr/bin/env python
"""
Test script for the data loading with the modern implementation.

This script tests loading the dataset using the modern torchtext API
implementation to verify the data pipeline works correctly.
"""
import sys
import os
import torch
from pathlib import Path
import random
import numpy as np
import time

# Add the project directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent))

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Import the data handling utilities
try:
    # Check if legacy API is available
    try:
        import torchtext.legacy
        from src.transformer.utils.data_handling import IWSLTDataset
        print("\nUsing legacy torchtext API")
        using_legacy = True
    except ImportError:
        # Use the modern implementation
        from src.transformer.utils.modern_data_handling import IWSLTDataset
        print("\nUsing modern torchtext API")
        using_legacy = False
except ImportError as e:
    print(f"Error importing data handling utilities: {e}")
    sys.exit(1)

def test_data_loading():
    """
    Test the data loading functionality.
    This verifies that we can load the dataset and create iterators.
    """
    print("\n=== Testing Data Loading ===")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tokenizer
    try:
        from src.transformer.utils.tokenization import Tokenizer
        tokenizer = Tokenizer()
        print("Tokenizer created successfully")
    except ImportError as e:
        print(f"Error creating tokenizer: {e}")
        sys.exit(1)
    
    # Load dataset
    try:
        print("\nLoading dataset...")
        dataset = IWSLTDataset(
            tokenizer=tokenizer,
            batch_size=16,
            device=device,
            max_length=50
        )
        
        print(f"Source vocabulary size: {len(dataset.source_field.vocab)}")
        print(f"Target vocabulary size: {len(dataset.target_field.vocab)}")
        
        # Get train, validation, and test iterators
        train_iterator, valid_iterator, test_iterator = dataset.get_iterators()
        
        # Test iterating through a batch
        batch = next(iter(train_iterator))
        print(f"\nBatch information:")
        print(f"Source shape: {batch.src.shape if hasattr(batch.src, 'shape') else batch.src[0].shape}")
        print(f"Target shape: {batch.trg.shape if hasattr(batch.trg, 'shape') else batch.trg[0].shape}")
        
        # Print special token indices
        print(f"\nSpecial token indices:")
        print(f"PAD index: {dataset.pad_idx}")
        print(f"SOS index: {dataset.sos_idx}")
        print(f"EOS index: {dataset.eos_idx}")
        
        print("\nSuccessfully loaded dataset and created iterators!")
        return True
        
    except Exception as e:
        import traceback
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\nData loading test completed successfully!")
        print("The modern implementation is working correctly.")
    else:
        print("\nData loading test failed. Please check the error messages.")
        sys.exit(1) 