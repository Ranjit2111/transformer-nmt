"""
Data handling utilities for the TransformerNMT model.

This module provides modern implementation of the data handling utilities
compatible with torchtext 0.15.0+.
"""
import torch

from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.modern_data_handling import (
    IWSLTDataset as ModernIWSLTDataset,
    TranslationDataset,
    ModernVocab,
    ModernField,
    ModernBucketIterator,
    Batch
)

# Directly using the modern implementation
class IWSLTDataset(ModernIWSLTDataset):
    """Handler for the IWSLT English-Vietnamese dataset."""
    pass  # Inherit everything from ModernIWSLTDataset

# For backward compatibility if needed
USING_LEGACY = False 