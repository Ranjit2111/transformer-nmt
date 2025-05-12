"""
Modern data handling utilities for the TransformerNMT model using torchtext 0.15.0+.

This module provides a modern implementation of the data handling utilities
that are compatible with torchtext 0.15.0 and later, which have removed
the legacy API including Field, BucketIterator, and the legacy IWSLT dataset.
"""
from typing import Tuple
import os
import urllib.request
from collections import namedtuple, Counter


import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Replace torchtext dependency with our custom implementation
from src.transformer.utils.vocab_utils import build_vocab_from_iterator, Vocab

from src.transformer.utils.tokenization import Tokenizer

class Batch:
    """
    A container for batched data with src and trg attributes.
    """
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
        
    def __iter__(self):
        """
        Make Batch iterable to support unpacking like: src, trg = batch
        """
        yield self.src
        yield self.trg

class ModernVocab:
    """
    A modern replacement for the legacy torchtext Field.vocab with compatible interface.
    """
    def __init__(self, tokens=None, specials=None, min_freq=1):
        """
        Initialize a ModernVocab.
        
        Args:
            tokens: Iterator of tokens to build vocabulary from
            specials: Special tokens to include
            min_freq: Minimum frequency for tokens to be included
        """
        if specials is None:
            specials = ['<unk>', '<pad>', '<sos>', '<eos>']
            
        # Ensure <unk> is the first special token
        if '<unk>' not in specials:
            specials = ['<unk>'] + specials
        elif specials[0] != '<unk>':
            # Move <unk> to the front
            specials.remove('<unk>')
            specials = ['<unk>'] + specials
            
        # Build vocabulary
        counter = Counter()
        if tokens:
            # If tokens is a set, convert to list
            if isinstance(tokens, set):
                tokens = list(tokens)
                
            # Count token frequencies
            if isinstance(tokens, (list, tuple)):
                counter.update(tokens)
                
        # Create token to index mapping
        self.stoi = {}
        idx = 0
        
        # Add specials first
        for token in specials:
            self.stoi[token] = idx
            idx += 1
            
        # Add tokens that meet min_freq
        for token, count in counter.most_common():
            if token not in self.stoi and count >= min_freq:
                self.stoi[token] = idx
                idx += 1
                
        # Create reverse mapping
        self.itos = [None] * len(self.stoi)
        for token, idx in self.stoi.items():
            self.itos[idx] = token
        
    def __len__(self):
        """
        Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        return len(self.stoi)

class ModernField:
    """
    A modern replacement for the legacy torchtext Field.
    """
    def __init__(
        self,
        tokenize=None,
        init_token=None,
        eos_token=None,
        pad_token=None,
        lower=False,
        batch_first=False,
        include_lengths=False,
        fix_length=None,
        use_vocab=True
    ):
        """
        Initialize a ModernField.
        
        Args:
            tokenize: Function to tokenize strings
            init_token: Token to start sequences (e.g., '<sos>')
            eos_token: Token to end sequences (e.g., '<eos>')
            pad_token: Token to pad sequences (e.g., '<pad>')
            lower: Whether to lowercase text
            batch_first: Whether batch dimension is first
            include_lengths: Whether to return lengths
            fix_length: Fixed length for sequences
            use_vocab: Whether to use a vocab object
        """
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.lower = lower
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.fix_length = fix_length
        self.use_vocab = use_vocab
        
        # Create an empty vocabulary
        self.vocab = None
        # Initialize vocab with just special tokens
        specials = []
        if self.init_token:
            specials.append(self.init_token)
        if self.eos_token:
            specials.append(self.eos_token)
        if self.pad_token:
            specials.append(self.pad_token)
        if 'unk' not in specials and '<unk>' not in specials:
            specials = ['<unk>'] + specials
            
        # Create empty vocab with just special tokens
        self.vocab = ModernVocab(None, specials=specials)
        
    def build_vocab(self, *args, min_freq=1, **kwargs):
        """
        Build vocabulary from data.
        
        Args:
            *args: Datasets to use for building vocabulary
            min_freq: Minimum frequency for tokens to be included
            **kwargs: Additional arguments
        """
        # Extract all tokens from all datasets
        counter = Counter()
        all_tokens = []
        print(f"Building vocabulary with {len(args)} datasets")
        
        token_count = 0
        example_count = 0
        
        for dataset in args:
            if isinstance(dataset, list):
                example_count += len(dataset)
                # Dataset is a list of tokenized sentences
                for tokens in dataset:
                    if tokens:
                        token_count += len(tokens)
                        counter.update(tokens)
                        all_tokens.extend(tokens)
            else:
                example_count += 1
                # Handle other types of datasets
                for example in dataset:
                    if hasattr(example, 'src'):
                        # If example is from legacy dataset
                        text = example.src
                    else:
                        # Assume example is a string or list of tokens
                        text = example
                    
                    # Tokenize and add to tokens
                    if isinstance(text, str):
                        tokens = self.tokenize(text)
                    else:
                        # Already tokenized
                        tokens = text
                    
                    token_count += len(tokens)
                    counter.update(tokens)
                    all_tokens.extend(tokens)
        
        print(f"Processed {example_count} examples with {token_count} total tokens")
        print(f"Found {len(counter)} unique tokens")
                
        # Create vocabulary with special tokens
        specials = []
        if self.init_token:
            specials.append(self.init_token)
        if self.eos_token:
            specials.append(self.eos_token)
        if self.pad_token:
            specials.append(self.pad_token)
        if '<unk>' not in specials:
            specials = ['<unk>'] + specials
        
        print(f"Adding {len(specials)} special tokens: {specials}")
        
        # Build vocabulary directly using counter
        token_to_idx = {}
        idx = 0
        
        # Add specials first
        for special in specials:
            token_to_idx[special] = idx
            idx += 1
        
        # Add all tokens that meet min_freq
        for token, count in counter.most_common():
            if count >= min_freq and token not in token_to_idx:
                token_to_idx[token] = idx
                idx += 1
        
        # Create vocab with the mapping
        self.vocab = Vocab(token_to_idx)
        print(f"Final vocabulary size: {len(self.vocab)}")
        
    def process(self, text):
        """
        Process text using this field.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        if self.lower and isinstance(text, str):
            text = text.lower()
            
        # Tokenize if needed
        if isinstance(text, str) and self.tokenize:
            tokens = self.tokenize(text)
        else:
            tokens = text
            
        # Add init and eos tokens if needed
        if self.init_token:
            tokens = [self.init_token] + list(tokens)
        if self.eos_token:
            tokens = list(tokens) + [self.eos_token]
            
        # Convert to indices if vocabulary exists
        if self.use_vocab and self.vocab:
            indices = [self.vocab.stoi.get(token, self.vocab.stoi['<unk>']) for token in tokens]
            return indices
        
        return tokens

class TranslationDataset(Dataset):
    """
    Dataset for machine translation with source and target texts.
    """
    def __init__(self, src_texts, tgt_texts, src_field, tgt_field):
        """
        Initialize a TranslationDataset.
        
        Args:
            src_texts: List of source texts
            tgt_texts: List of target texts
            src_field: Field for source language
            tgt_field: Field for target language
        """
        assert len(src_texts) == len(tgt_texts), "Source and target texts must have the same length"
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_field = src_field
        self.tgt_field = tgt_field
        
    def __len__(self):
        """
        Get dataset size.
        
        Returns:
            Number of examples in dataset
        """
        return len(self.src_texts)
        
    def __getitem__(self, idx):
        """
        Get example at index.
        
        Args:
            idx: Index
            
        Returns:
            (source_text, target_text) pair
        """
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        # Process using fields
        src_processed = self.src_field.process(src_text)
        tgt_processed = self.tgt_field.process(tgt_text)
        
        return src_processed, tgt_processed
    
    def get_attrs(self):
        """
        Create a namedtuple with attributes like the legacy dataset.
        
        Returns:
            namedtuple with src and trg attributes
        """
        Example = namedtuple('Example', ['src', 'trg'])
        return [Example(src=src, trg=tgt) for src, tgt in zip(self.src_texts, self.tgt_texts)]

class ModernBucketIterator:
    """
    A modern replacement for the legacy torchtext BucketIterator.
    """
    def __init__(self, dataset, batch_size, sort_key=None, device=None, sort_within_batch=False):
        """
        Initialize a ModernBucketIterator.
        
        Args:
            dataset: Dataset to iterate over
            batch_size: Batch size
            sort_key: Function to sort examples
            device: Device to place tensors on
            sort_within_batch: Whether to sort within each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.device = device
        self.sort_within_batch = sort_within_batch
        
        # Get special token indices
        if hasattr(dataset, 'src_field') and dataset.src_field.vocab is not None:
            self.src_pad_idx = dataset.src_field.vocab.stoi.get('<pad>', 0)
        else:
            self.src_pad_idx = 0
            
        if hasattr(dataset, 'tgt_field') and dataset.tgt_field.vocab is not None:
            self.tgt_pad_idx = dataset.tgt_field.vocab.stoi.get('<pad>', 0)
        else:
            self.tgt_pad_idx = 0
        
        # Create dataloader with appropriate collate function
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=not sort_within_batch
        )
        
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: Batch of examples
            
        Returns:
            Batch object with src and trg attributes
        """
        # Sort batch by source length if required
        if self.sort_within_batch:
            batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Split into source and target
        src_batch, tgt_batch = zip(*batch)
        
        # Convert to tensors and pad
        src_tensors = [torch.tensor(src, dtype=torch.long) for src in src_batch]
        tgt_tensors = [torch.tensor(tgt, dtype=torch.long) for tgt in tgt_batch]
        
        # Pad sequences
        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=self.src_pad_idx)
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=self.tgt_pad_idx)
        
        # Move to device if specified
        if self.device is not None:
            src_padded = src_padded.to(self.device)
            tgt_padded = tgt_padded.to(self.device)
            
        # Create and return Batch
        return Batch(src=src_padded, trg=tgt_padded)
        
    def __iter__(self):
        """
        Iterate over batches.
        
        Returns:
            Iterator over batches
        """
        return iter(self.dataloader)
        
    def __len__(self):
        """
        Get number of batches.
        
        Returns:
            Number of batches
        """
        return len(self.dataloader)

def download_iwslt_dataset(root="data", language_pair=('en', 'vi')):
    """
    Download the IWSLT dataset manually since torchtext 0.15.0 might have issues.
    
    Args:
        root: Directory to save dataset
        language_pair: Language pair to download
        
    Returns:
        Dictionary with train, valid, and test data
    """
    # Create directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Check if files already exist
    train_files = ["train.en", "train.vi"]
    tst2012_files = ["tst2012.en", "tst2012.vi"]
    tst2013_files = ["tst2013.en", "tst2013.vi"]
    
    all_files = train_files + tst2012_files + tst2013_files
    all_files_exist = all(os.path.exists(os.path.join(root, file)) for file in all_files)
    
    if not all_files_exist:
        print("Some IWSLT dataset files are missing. Using fallback sample dataset.")
        # Create small sample dataset for testing
        sample_data = {
            'train': (
                ["Hello world", "This is a test", "How are you doing?", "Machine translation is interesting", 
                 "Deep learning is powerful", "Neural networks can learn complex patterns", 
                 "Attention mechanisms improve performance", "Transformers are state of the art",
                 "English is a global language", "This is a simple example",
                 "I am a student", "What is your name?", "Nice to meet you", "Where do you live?",
                 "The weather is nice today", "I like programming", "Python is easy to learn",
                 "The transformer model works well", "Artificial intelligence is the future",
                 "Natural language processing is fun"],
                ["Xin chào thế giới", "Đây là một bài kiểm tra", "Bạn khỏe không?", "Dịch máy rất thú vị", 
                 "Học sâu rất mạnh mẽ", "Mạng nơ-ron có thể học các mẫu phức tạp", 
                 "Cơ chế chú ý cải thiện hiệu suất", "Transformers là tiên tiến nhất",
                 "Tiếng Anh là ngôn ngữ toàn cầu", "Đây là một ví dụ đơn giản",
                 "Tôi là sinh viên", "Tên bạn là gì?", "Rất vui được gặp bạn", "Bạn sống ở đâu?",
                 "Thời tiết hôm nay đẹp", "Tôi thích lập trình", "Python dễ học",
                 "Mô hình transformer hoạt động tốt", "Trí tuệ nhân tạo là tương lai",
                 "Xử lý ngôn ngữ tự nhiên rất thú vị"]
            ),
            'valid': (
                ["I love programming", "Python is easy to learn", "What time is it?", "Can you help me?"],
                ["Tôi thích lập trình", "Python dễ học", "Mấy giờ rồi?", "Bạn có thể giúp tôi không?"]
            ),
            'test': (
                ["This is a test sentence", "Translation should work well", "Hello, how are you?", "My name is John"],
                ["Đây là câu kiểm tra", "Bản dịch nên hoạt động tốt", "Xin chào, bạn khỏe không?", "Tên tôi là John"]
            )
        }
        return sample_data
        
    # Try to download from original source if files don't exist
    try:
        # Define URLs and paths for IWSLT data
        url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"
        
        # Download files if they don't exist
        for file in all_files:
            file_path = os.path.join(root, file)
            if not os.path.exists(file_path):
                print(f"Downloading {file}...")
                urllib.request.urlretrieve(url + file, file_path)
    except Exception as e:
        print(f"Error downloading IWSLT dataset: {e}")
        print("Using fallback sample dataset instead.")
        # Create small sample dataset for testing
        sample_data = {
            'train': (
                ["Hello world", "This is a test", "How are you doing?", "Machine translation is interesting", 
                 "Deep learning is powerful", "Neural networks can learn complex patterns", 
                 "Attention mechanisms improve performance", "Transformers are state of the art",
                 "English is a global language", "This is a simple example",
                 "I am a student", "What is your name?", "Nice to meet you", "Where do you live?",
                 "The weather is nice today", "I like programming", "Python is easy to learn",
                 "The transformer model works well", "Artificial intelligence is the future",
                 "Natural language processing is fun"],
                ["Xin chào thế giới", "Đây là một bài kiểm tra", "Bạn khỏe không?", "Dịch máy rất thú vị", 
                 "Học sâu rất mạnh mẽ", "Mạng nơ-ron có thể học các mẫu phức tạp", 
                 "Cơ chế chú ý cải thiện hiệu suất", "Transformers là tiên tiến nhất",
                 "Tiếng Anh là ngôn ngữ toàn cầu", "Đây là một ví dụ đơn giản",
                 "Tôi là sinh viên", "Tên bạn là gì?", "Rất vui được gặp bạn", "Bạn sống ở đâu?",
                 "Thời tiết hôm nay đẹp", "Tôi thích lập trình", "Python dễ học",
                 "Mô hình transformer hoạt động tốt", "Trí tuệ nhân tạo là tương lai",
                 "Xử lý ngôn ngữ tự nhiên rất thú vị"]
            ),
            'valid': (
                ["I love programming", "Python is easy to learn", "What time is it?", "Can you help me?"],
                ["Tôi thích lập trình", "Python dễ học", "Mấy giờ rồi?", "Bạn có thể giúp tôi không?"]
            ),
            'test': (
                ["This is a test sentence", "Translation should work well", "Hello, how are you?", "My name is John"],
                ["Đây là câu kiểm tra", "Bản dịch nên hoạt động tốt", "Xin chào, bạn khỏe không?", "Tên tôi là John"]
            )
        }
        return sample_data
            
    # Read data
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
            
    # Load all data
    train_src = read_file(os.path.join(root, train_files[0]))
    train_tgt = read_file(os.path.join(root, train_files[1]))
    
    valid_src = read_file(os.path.join(root, tst2012_files[0]))
    valid_tgt = read_file(os.path.join(root, tst2012_files[1]))
    
    test_src = read_file(os.path.join(root, tst2013_files[0]))
    test_tgt = read_file(os.path.join(root, tst2013_files[1]))
    
    # Create dataset dictionary
    return {
        'train': (train_src, train_tgt),
        'valid': (valid_src, valid_tgt),
        'test': (test_src, test_tgt)
    }

class IWSLTDataset:
    """
    Handler for the IWSLT English-Vietnamese dataset, modernized version.
    This maintains the same interface as the legacy version for compatibility.
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
        
    def _create_field(
        self, 
        tokenize_fn: callable, 
        is_source: bool
    ) -> ModernField:
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
        
    def _load_data(self) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
        """
        Load the IWSLT English-Vietnamese dataset.
        
        Returns:
            Tuple of (train, validation, test) datasets
        """
        print("Loading IWSLT English-Vietnamese dataset...")
        
        # Download dataset manually
        data = download_iwslt_dataset()
        
        # Create datasets
        train_data = TranslationDataset(
            data['train'][0], data['train'][1], 
            self.source_field, self.target_field
        )
        
        valid_data = TranslationDataset(
            data['valid'][0], data['valid'][1], 
            self.source_field, self.target_field
        )
        
        test_data = TranslationDataset(
            data['test'][0], data['test'][1], 
            self.source_field, self.target_field
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
        train_attrs = self.train_data.get_attrs()
        self.source_field.build_vocab(
            [ex.src for ex in train_attrs], 
            min_freq=self.min_freq
        )
        
        # Build target vocabulary from training data
        self.target_field.build_vocab(
            [ex.trg for ex in train_attrs], 
            min_freq=self.min_freq
        )
        
        print(f"  Source vocabulary size: {len(self.source_field.vocab)}")
        print(f"  Target vocabulary size: {len(self.target_field.vocab)}")
        
    def _create_iterators(self) -> Tuple[ModernBucketIterator, ModernBucketIterator, ModernBucketIterator]:
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
        
    def get_iterators(self) -> Tuple[ModernBucketIterator, ModernBucketIterator, ModernBucketIterator]:
        """
        Get train, validation, and test iterators.
        
        Returns:
            Tuple of (train, validation, test) iterators
        """
        return self.train_iterator, self.valid_iterator, self.test_iterator

    def _create_fallback_dataset(self):
        """
        Create a small fallback dataset when the IWSLT dataset is not available.
        """
        # Create a sample dataset with some English-Vietnamese pairs
        print("Creating fallback sample dataset...")
        
        # Training pairs (source, target)
        train_pairs = [
            # Basic sentences for demonstration
            ('hello', 'xin chào'),
            ('thank you', 'cảm ơn bạn'),
            ('how are you', 'bạn khỏe không'),
            ('good morning', 'chào buổi sáng'),
            ('good night', 'chúc ngủ ngon'),
            ('my name is john', 'tôi tên là john'),
            ('i am a student', 'tôi là sinh viên'),
            ('i like to read books', 'tôi thích đọc sách'),
            ('what is your name', 'tên bạn là gì'),
            ('where are you from', 'bạn đến từ đâu'),
            ('i am from america', 'tôi đến từ mỹ'),
            ('how old are you', 'bạn bao nhiêu tuổi'),
            ('i am twenty years old', 'tôi hai mươi tuổi'),
            ('the weather is nice today', 'thời tiết hôm nay đẹp'),
            ('i am learning vietnamese', 'tôi đang học tiếng việt'),
            ('do you speak english', 'bạn có nói tiếng anh không'),
            ('i do not understand', 'tôi không hiểu'),
            ('please speak slowly', 'xin hãy nói chậm'),
            ('can you help me', 'bạn có thể giúp tôi không'),
            ('i need to go now', 'tôi cần phải đi bây giờ'),
        ]
        
        # Validation pairs (source, target)
        valid_pairs = [
            ('excuse me', 'xin lỗi'),
            ('this is delicious', 'cái này ngon'),
            ('how much is this', 'cái này giá bao nhiêu'),
            ('have a nice day', 'chúc một ngày tốt lành'),
        ]
        
        # Test pairs (source, target)
        test_pairs = [
            ('goodbye', 'tạm biệt'),
            ('see you tomorrow', 'hẹn gặp lại ngày mai'),
            ('i love vietnam', 'tôi yêu việt nam'),
            ('i want to eat pho', 'tôi muốn ăn phở'),
        ]
        
        print(f"  Train: {len(train_pairs)} examples")
        print(f"  Validation: {len(valid_pairs)} examples")
        print(f"  Test: {len(test_pairs)} examples")
        
        # Convert pairs to examples
        train_examples = self._pairs_to_examples(train_pairs)
        valid_examples = self._pairs_to_examples(valid_pairs)
        test_examples = self._pairs_to_examples(test_pairs)
        
        return train_examples, valid_examples, test_examples 