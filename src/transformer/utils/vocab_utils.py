"""
Custom vocabulary utilities for the TransformerNMT model.

This module provides vocabulary-related functions that replace torchtext dependencies
with custom implementations for better compatibility.
"""
from typing import Iterable, List, Dict, Optional, Callable, Set, Iterator
from collections import Counter, OrderedDict


class Vocab:
    """
    Custom Vocab class that mimics torchtext.vocab.Vocab functionality.
    
    Maps tokens to indices and provides utility methods for working with vocabularies.
    """
    
    def __init__(self, tokens_to_idx: Dict[str, int]):
        """
        Initialize the Vocab object.
        
        Args:
            tokens_to_idx: Dictionary mapping tokens to indices
        """
        self.stoi = tokens_to_idx
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self._default_index = None
        
    def __getitem__(self, token: str) -> int:
        """
        Get the index of a token.
        
        Args:
            token: The token to look up
            
        Returns:
            The index of the token
        """
        if token in self.stoi:
            return self.stoi[token]
        if self._default_index is not None:
            return self._default_index
        raise RuntimeError(f"Token '{token}' not found in vocabulary")
    
    def __len__(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            The number of tokens in the vocabulary
        """
        return len(self.stoi)
    
    def __contains__(self, token: str) -> bool:
        """
        Check if a token is in the vocabulary.
        
        Args:
            token: The token to check
            
        Returns:
            True if the token is in the vocabulary, False otherwise
        """
        return token in self.stoi
    
    def set_default_index(self, index: Optional[int]) -> None:
        """
        Set the default index to use for OOV tokens.
        
        Args:
            index: The default index or None to disable
        """
        self._default_index = index
        
    def get_default_index(self) -> Optional[int]:
        """
        Get the default index used for OOV tokens.
        
        Returns:
            The default index or None if not set
        """
        return self._default_index
    
    def get_itos(self) -> List[str]:
        """
        Get the list of token strings indexed by their position.
        
        Returns:
            List mapping indices to tokens
        """
        return [self.itos[i] for i in range(len(self.itos))]
    
    def get_stoi(self) -> Dict[str, int]:
        """
        Get the dictionary mapping tokens to indices.
        
        Returns:
            Dictionary mapping tokens to indices
        """
        return self.stoi
    
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """
        Look up the indices of multiple tokens.
        
        Args:
            tokens: List of tokens to look up
            
        Returns:
            List of indices
        """
        return [self[token] for token in tokens]
    
    def lookup_token(self, index: int) -> str:
        """
        Look up the token for an index.
        
        Args:
            index: Index to look up
            
        Returns:
            The token at the given index
        """
        if index not in self.itos:
            raise RuntimeError(f"Index {index} not found in vocabulary")
        return self.itos[index]
    
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        """
        Look up the tokens for multiple indices.
        
        Args:
            indices: List of indices to look up
            
        Returns:
            List of tokens
        """
        return [self.lookup_token(index) for index in indices]
    
    def append_token(self, token: str) -> None:
        """
        Append a token to the vocabulary.
        
        Args:
            token: Token to append
            
        Raises:
            RuntimeError: If the token is already in the vocabulary
        """
        if token in self.stoi:
            raise RuntimeError(f"Token '{token}' already exists in the vocabulary")
        idx = len(self.stoi)
        self.stoi[token] = idx
        self.itos[idx] = token
        
    def insert_token(self, token: str, index: int) -> None:
        """
        Insert a token at a specific index.
        
        Args:
            token: Token to insert
            index: Index to insert at
            
        Raises:
            RuntimeError: If the token is already in the vocabulary or the index is out of range
        """
        if token in self.stoi:
            raise RuntimeError(f"Token '{token}' already exists in the vocabulary")
        if index < 0 or index > len(self.stoi):
            raise RuntimeError(f"Index {index} out of range [0, {len(self.stoi)}]")
            
        # Shift indices of existing tokens
        new_stoi = {}
        new_itos = {}
        
        # Add tokens with indices less than the insertion point
        for old_token, old_idx in self.stoi.items():
            if old_idx < index:
                new_stoi[old_token] = old_idx
                new_itos[old_idx] = old_token
                
        # Add the new token
        new_stoi[token] = index
        new_itos[index] = token
        
        # Add tokens with indices greater than or equal to the insertion point
        for old_token, old_idx in self.stoi.items():
            if old_idx >= index:
                new_stoi[old_token] = old_idx + 1
                new_itos[old_idx + 1] = old_token
                
        self.stoi = new_stoi
        self.itos = new_itos


def _flatten_iterator(iterator: Iterable) -> Iterator:
    """
    Flatten a nested iterator into a single iterator.
    
    Args:
        iterator: Nested iterator
        
    Returns:
        Flattened iterator
    """
    for item in iterator:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None
) -> Vocab:
    """
    Build a vocabulary from an iterator.
    
    Args:
        iterator: Iterator of tokens or lists of tokens
        min_freq: Minimum frequency for a token to be included
        specials: Special tokens to include regardless of frequency
        special_first: Whether to put special tokens at the beginning
        max_tokens: Maximum number of tokens to include
        
    Returns:
        A Vocab object
    """
    counter = Counter()
    for tokens in iterator:
        counter.update(_flatten_iterator(tokens))
        
    specials = specials or []
    
    # Filter and sort tokens by frequency
    filtered_tokens = OrderedDict()
    
    # Add special tokens first if special_first is True
    if special_first:
        for token in specials:
            filtered_tokens[token] = counter.get(token, 0)
    
    # Add tokens that meet the minimum frequency
    token_freq_pairs = sorted(
        ((tok, freq) for tok, freq in counter.items() if freq >= min_freq and tok not in specials),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Apply max_tokens limit if specified
    if max_tokens is not None:
        remaining_slots = max_tokens - len(specials)
        token_freq_pairs = token_freq_pairs[:remaining_slots]
    
    # Add tokens to filtered_tokens
    for token, freq in token_freq_pairs:
        filtered_tokens[token] = freq
    
    # Add special tokens at the end if special_first is False
    if not special_first:
        for token in specials:
            if token not in filtered_tokens:
                filtered_tokens[token] = counter.get(token, 0)
    
    # Create vocab object
    token_to_idx = {token: idx for idx, token in enumerate(filtered_tokens.keys())}
    
    return Vocab(token_to_idx) 