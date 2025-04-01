"""
Tokenization utilities for the TransformerNMT model.
"""
from typing import List

import spacy
from spacy.language import Language


class Tokenizer:
    """
    Tokenizer class for English and Vietnamese text.
    """
    
    def __init__(self):
        """
        Initialize English and Vietnamese tokenizers.
        
        For Vietnamese, use basic whitespace tokenization with some preprocessing.
        """
        # Load English tokenizer
        try:
            self.english_nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading English language model for spaCy...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.english_nlp = spacy.load("en_core_web_sm")
            
        # Vietnamese doesn't have an official spaCy model, so we'll use
        # a simple whitespace tokenizer with some preprocessing
    
    def tokenize_en(self, text: str) -> List[str]:
        """
        Tokenize English text.
        
        Args:
            text: English text to tokenize
            
        Returns:
            List of tokens
        """
        return [token.text.lower() for token in self.english_nlp.tokenizer(text)]
    
    def tokenize_vi(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text.
        
        Args:
            text: Vietnamese text to tokenize
            
        Returns:
            List of tokens
        """
        # Basic preprocessing
        text = text.lower()
        
        # Handle some Vietnamese specific characters
        text = text.replace('\u0027', "'")  # Normalize apostrophes
        
        # Simple whitespace tokenization for Vietnamese
        # For a production system, consider using a specialized Vietnamese tokenizer
        return text.strip().split()
        
    def detokenize_en(self, tokens: List[str]) -> str:
        """
        Convert English tokens back to text.
        
        Args:
            tokens: List of English tokens
            
        Returns:
            Detokenized text
        """
        return " ".join(tokens)
        
    def detokenize_vi(self, tokens: List[str]) -> str:
        """
        Convert Vietnamese tokens back to text.
        
        Args:
            tokens: List of Vietnamese tokens
            
        Returns:
            Detokenized text
        """
        return " ".join(tokens) 