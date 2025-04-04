"""
Hyperparameters compatibility layer.

This module provides backward compatibility with the config-based 
hyperparameters system. It imports from src.config.params and exposes
a simplified interface for testing and development.
"""
import torch
from dataclasses import dataclass

# Try to import from config, fall back to defaults if not available
try:
    from src.config import params as config_params
    
    @dataclass
    class ModelConfig:
        """
        Configuration class for TransformerNMT model using config values.
        """
        # Model architecture parameters
        d_model: int = config_params.d_model
        n_heads: int = config_params.n_heads
        n_layers: int = config_params.n_layers
        d_ff: int = config_params.ffn_hidden
        dropout: float = config_params.dropout
        
        # Training parameters
        lr: float = config_params.init_lr
        batch_size: int = config_params.batch_size
        num_epochs: int = config_params.epochs
        
        # Data parameters
        max_len: int = config_params.max_seq_length
        min_freq: int = config_params.min_freq
        
except ImportError:
    # Default values if config is not available
    @dataclass
    class ModelConfig:
        """
        Configuration class for TransformerNMT model with default values.
        """
        # Model architecture parameters
        d_model: int = 512
        n_heads: int = 8
        n_layers: int = 6
        d_ff: int = 2048
        dropout: float = 0.1
        
        # Training parameters
        lr: float = 0.0001
        batch_size: int = 32
        num_epochs: int = 10
        
        # Data parameters
        max_len: int = 100
        min_freq: int = 2


def get_model_config():
    """
    Returns a model configuration with values from config or defaults.
    
    Returns:
        ModelConfig with configured values
    """
    return ModelConfig()


def get_training_hyperparams():
    """
    Returns a dictionary of training hyperparameters.
    
    Returns:
        Dict of hyperparameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_model_config()
    
    return {
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers,
        'd_ff': config.d_ff,
        'dropout': config.dropout,
        'lr': config.lr,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'max_len': config.max_len,
        'min_freq': config.min_freq,
        'device': device
    } 