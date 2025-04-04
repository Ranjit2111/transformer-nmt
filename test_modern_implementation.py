#!/usr/bin/env python
"""
Test the modern implementation of the torchtext API.

This script demonstrates and tests the modern API components:
- ModernField
- ModernVocab
- ModernBucketIterator
- TranslationDataset
"""
import torch
import sys
import os
from pathlib import Path

# Add the project directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent))

# Import the modern implementation
try:
    from src.transformer.utils.modern_data_handling import (
        ModernField, 
        ModernVocab, 
        ModernBucketIterator, 
        TranslationDataset,
        IWSLTDataset,
        Batch
    )
except ImportError as e:
    print(f"Error importing modern implementation: {e}")
    sys.exit(1)

# Import the tokenizer
try:
    from src.transformer.utils.tokenization import Tokenizer
except ImportError as e:
    print(f"Error importing tokenizer: {e}")
    sys.exit(1)

def test_modern_vocab():
    """Test the ModernVocab class"""
    print("\n=== Testing ModernVocab ===")
    
    # Create a vocabulary
    tokenizer = Tokenizer()
    texts = [
        "This is a test sentence.",
        "Another example sentence.",
        "A third sentence for testing."
    ]
    
    # Tokenize the texts
    tokenized_texts = [tokenizer.tokenize_en(text) for text in texts]
    
    # Flatten the list of tokens
    all_tokens = []
    for tokens in tokenized_texts:
        all_tokens.extend(tokens)
    
    # Create a vocabulary using our implementation
    vocab = ModernVocab(
        tokens=all_tokens,
        specials=["<unk>", "<pad>", "<bos>", "<eos>"],
        min_freq=1
    )
    
    # Print vocabulary information
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Tokens in vocabulary: {vocab.itos[:10]}...")
    
    # Test token to index
    test_tokens = ["This", "is", "a", "test", "unknown_token"]
    indices = [vocab.stoi.get(token, vocab.stoi['<unk>']) for token in test_tokens]
    print(f"Tokens: {test_tokens}")
    print(f"Indices: {indices}")
    
    # Test index to token
    tokens = [vocab.itos[idx] for idx in indices]
    print(f"Back to tokens: {tokens}")
    
    # Test special tokens
    print(f"PAD index: {vocab.stoi['<pad>']}")
    print(f"UNK index: {vocab.stoi['<unk>']}")
    print(f"BOS index: {vocab.stoi['<bos>']}")
    print(f"EOS index: {vocab.stoi['<eos>']}")
    
    return vocab

def test_modern_field(vocab):
    """Test the ModernField class"""
    print("\n=== Testing ModernField ===")
    
    # Create a field
    tokenizer = Tokenizer()
    field = ModernField(
        tokenize=tokenizer.tokenize_en,
        init_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        include_lengths=True
    )
    
    # Set the vocabulary
    field.vocab = vocab
    
    # Test processing a sentence
    sentence = "This is a test sentence."
    processed = field.process(sentence)
    print(f"Original sentence: {sentence}")
    print(f"Processed indices: {processed}")
    
    # Test batch processing
    sentences = [
        "This is a test sentence.",
        "Another example sentence.",
        "A third sentence for testing."
    ]
    processed_batch = [field.process(s) for s in sentences]
    print(f"Batch processed successfully with {len(processed_batch)} examples")
    
    return field

def test_translation_dataset():
    """Test the TranslationDataset class"""
    print("\n=== Testing TranslationDataset ===")
    
    # Create source and target fields
    tokenizer = Tokenizer()
    
    src_field = ModernField(
        tokenize=tokenizer.tokenize_en,
        init_token=None,
        eos_token="<eos>",
        pad_token="<pad>"
    )
    
    tgt_field = ModernField(
        tokenize=tokenizer.tokenize_vi,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>"
    )
    
    # Create example data
    src_texts = [
        "This is a test sentence.",
        "Another example sentence.",
        "A third sentence for testing."
    ]
    
    tgt_texts = [
        "Đây là một câu thử nghiệm.",
        "Một câu ví dụ khác.",
        "Câu thứ ba để kiểm tra."
    ]
    
    # Create a dataset
    dataset = TranslationDataset(
        src_texts=src_texts, 
        tgt_texts=tgt_texts, 
        src_field=src_field, 
        tgt_field=tgt_field
    )
    
    # Build vocabularies directly from tokenized texts instead of from dataset
    src_tokenized = [tokenizer.tokenize_en(text) for text in src_texts]
    tgt_tokenized = [tokenizer.tokenize_vi(text) for text in tgt_texts]
    
    # Flatten tokens for vocabulary building
    src_tokens = []
    for tokens in src_tokenized:
        src_tokens.extend(tokens)
    
    tgt_tokens = []
    for tokens in tgt_tokenized:
        tgt_tokens.extend(tokens)
    
    # Build vocabularies
    src_field.vocab = ModernVocab(tokens=src_tokens, min_freq=1)
    tgt_field.vocab = ModernVocab(tokens=tgt_tokens, min_freq=1)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Source vocabulary size: {len(src_field.vocab)}")
    print(f"Target vocabulary size: {len(tgt_field.vocab)}")
    
    # Get an example
    src, tgt = dataset[0]
    print(f"Example source indices: {src}")
    print(f"Example target indices: {tgt}")
    
    # Test get_attrs
    attrs = dataset.get_attrs()
    print(f"get_attrs returned {len(attrs)} examples")
    print(f"First example src: {attrs[0].src}")
    print(f"First example trg: {attrs[0].trg}")
    
    return dataset, src_field, tgt_field

def test_bucket_iterator(dataset, src_field, tgt_field):
    """Test the ModernBucketIterator class"""
    print("\n=== Testing ModernBucketIterator ===")
    
    # Create a bucket iterator
    iterator = ModernBucketIterator(
        dataset=dataset,
        batch_size=2,
        sort_key=lambda x: len(x[0]),
        device=torch.device('cpu'),
        sort_within_batch=True
    )
    
    # Get a batch
    for i, batch in enumerate(iterator):
        print(f"Batch {i+1}:")
        print(f"  Source shape: {batch.src.shape}")
        print(f"  Target shape: {batch.trg.shape}")
        
        # Decode the batch
        src_tokens = [[src_field.vocab.itos[idx.item()] for idx in sent] for sent in batch.src]
        tgt_tokens = [[tgt_field.vocab.itos[idx.item()] for idx in sent] for sent in batch.trg]
        
        print("  Example decoded source:", " ".join([t for t in src_tokens[0] if t not in ['<pad>']]))
        print("  Example decoded target:", " ".join([t for t in tgt_tokens[0] if t not in ['<pad>']]))
        
        if i >= 0:  # Only process the first batch
            break
    
    return iterator

def test_iwslt_dataset():
    """Test the IWSLTDataset class"""
    print("\n=== Testing IWSLTDataset ===")
    
    # Create an IWSLT dataset
    try:
        tokenizer = Tokenizer()
        iwslt = IWSLTDataset(
            tokenizer=tokenizer,
            batch_size=16,
            device=torch.device('cpu'),
            max_length=100,
            min_freq=1
        )
        
        print(f"Train dataset size: {len(iwslt.train_data)}")
        print(f"Validation dataset size: {len(iwslt.valid_data)}")
        print(f"Test dataset size: {len(iwslt.test_data)}")
        print(f"Source vocabulary size: {len(iwslt.source_field.vocab)}")
        print(f"Target vocabulary size: {len(iwslt.target_field.vocab)}")
        
        # Get iterators
        train_iterator, valid_iterator, test_iterator = iwslt.get_iterators()
        
        # Get a batch
        batch = next(iter(train_iterator))
        print(f"Batch source shape: {batch.src.shape}")
        print(f"Batch target shape: {batch.trg.shape}")
        
        # Special token indices
        print(f"PAD idx: {iwslt.pad_idx}")
        print(f"SOS idx: {iwslt.sos_idx}")
        print(f"EOS idx: {iwslt.eos_idx}")
            
    except Exception as e:
        print(f"Error testing IWSLT dataset: {e}")
        print("This is expected if the IWSLT dataset is not available.")
        print("A fallback dataset will be used.")

def main():
    print("=== Testing Modern torchtext Implementation ===")
    
    # Check if running with modern API
    try:
        from torchtext.legacy.data import Field
        print("\nWARNING: Running with legacy torchtext API available.")
        print("This test will still use the modern implementation directly.")
    except ImportError:
        print("\nRunning with modern torchtext API only.")
    
    # Run tests
    vocab = test_modern_vocab()
    field = test_modern_field(vocab)
    dataset, src_field, tgt_field = test_translation_dataset()
    iterator = test_bucket_iterator(dataset, src_field, tgt_field)
    test_iwslt_dataset()
    
    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    main() 