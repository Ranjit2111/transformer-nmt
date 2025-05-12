#!/usr/bin/env python
"""
Test script to debug vocabulary building issue.
"""
import os
import torch
import time
from collections import namedtuple
from src.transformer.utils.tokenization import Tokenizer
from src.transformer.utils.modern_data_handling import TranslationDataset, ModernField

def load_vocab_files():
    """Load existing vocabulary files to compare with what we're building."""
    vocab_en_path = os.path.join('data', "IWSLT'15 en-vi", 'vocab.en.txt')
    vocab_vi_path = os.path.join('data', "IWSLT'15 en-vi", 'vocab.vi.txt')
    
    vocab_en = []
    vocab_vi = []
    
    if os.path.exists(vocab_en_path):
        with open(vocab_en_path, 'r', encoding='utf-8') as f:
            vocab_en = [line.strip() for line in f]
        print(f"Loaded English vocabulary with {len(vocab_en)} tokens")
    
    if os.path.exists(vocab_vi_path):
        with open(vocab_vi_path, 'r', encoding='utf-8') as f:
            vocab_vi = [line.strip() for line in f]
        print(f"Loaded Vietnamese vocabulary with {len(vocab_vi)} tokens")
    
    return vocab_en, vocab_vi

def read_data():
    """Read training data to check tokenization and vocabulary building."""
    # Define file paths
    dataset_dir = os.path.join('data', "IWSLT'15 en-vi")
    train_en_path = os.path.join(dataset_dir, 'train.en.txt')
    train_vi_path = os.path.join(dataset_dir, 'train.vi.txt')
    
    # Check if files exist
    if not os.path.exists(train_en_path) or not os.path.exists(train_vi_path):
        print(f"Training files not found at {train_en_path} or {train_vi_path}")
        return None, None
    
    # Read data
    with open(train_en_path, 'r', encoding='utf-8') as f:
        train_src = [line.strip() for line in f]
    
    with open(train_vi_path, 'r', encoding='utf-8') as f:
        train_tgt = [line.strip() for line in f]
    
    print(f"Loaded {len(train_src)} English sentences")
    print(f"Loaded {len(train_tgt)} Vietnamese sentences")
    
    if train_src:
        print(f"Sample English: '{train_src[0]}'")
    if train_tgt:
        print(f"Sample Vietnamese: '{train_tgt[0]}'")
    
    return train_src, train_tgt

def test_tokenization():
    """Test tokenization functions."""
    tokenizer = Tokenizer()
    
    test_en = "Hello, this is a test sentence for tokenization."
    test_vi = "Xin chào, đây là một câu thử nghiệm cho việc phân đoạn."
    
    tokens_en = tokenizer.tokenize_en(test_en)
    tokens_vi = tokenizer.tokenize_vi(test_vi)
    
    print(f"English tokenization: {tokens_en}")
    print(f"Vietnamese tokenization: {tokens_vi}")
    
    return tokenizer

def build_test_vocab():
    """Build a test vocabulary and check results."""
    tokenizer = test_tokenization()
    train_src, train_tgt = read_data()
    
    if not train_src or not train_tgt:
        return
    
    # Create fields
    source_field = ModernField(
        tokenize=tokenizer.tokenize_en,
        init_token=None,
        eos_token='<eos>',
        pad_token='<pad>',
        lower=True,
        batch_first=True,
        fix_length=None,
        use_vocab=True
    )
    
    target_field = ModernField(
        tokenize=tokenizer.tokenize_vi,
        init_token='<sos>',
        eos_token='<eos>',
        pad_token='<pad>',
        lower=True,
        batch_first=True,
        fix_length=None,
        use_vocab=True
    )
    
    # Process a sample to see what the tokenized data looks like
    processed_en = source_field.process(train_src[0])
    processed_vi = target_field.process(train_tgt[0])
    
    print(f"Processed English example: {processed_en}")
    print(f"Processed Vietnamese example: {processed_vi}")
    
    # Create dataset WITHOUT setting use_vocab=True in fields first
    # This is the key difference - we need to tokenize the data before building vocab
    tokenized_src = [tokenizer.tokenize_en(text) for text in train_src[:1000]]  # Use a subset for testing
    tokenized_tgt = [tokenizer.tokenize_vi(text) for text in train_tgt[:1000]]
    
    print(f"Sample tokenized English: {tokenized_src[0]}")
    print(f"Sample tokenized Vietnamese: {tokenized_tgt[0]}")
    
    # Build vocabulary directly from tokenized data
    print("Building source vocabulary...")
    source_field.build_vocab(tokenized_src, min_freq=2)
    
    print("Building target vocabulary...")
    target_field.build_vocab(tokenized_tgt, min_freq=2)
    
    print(f"Source vocabulary size: {len(source_field.vocab)}")
    print(f"Target vocabulary size: {len(target_field.vocab)}")
    
    # Compare to expected vocabulary sizes
    vocab_en, vocab_vi = load_vocab_files()
    
    print(f"Expected English vocabulary size: {len(vocab_en)}")
    print(f"Expected Vietnamese vocabulary size: {len(vocab_vi)}")
    
    return source_field, target_field

if __name__ == "__main__":
    # Run tests
    build_test_vocab() 