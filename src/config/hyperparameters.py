"""
Hyperparameters for the TransformerNMT model.
Loads configuration from YAML file.
"""
import os
import yaml
import torch
from typing import Dict, Any, Optional


class HyperParameters:
    """
    Configuration handler for the TransformerNMT model.
    Loads parameters from a YAML file and provides access to them.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize hyperparameters from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
                         If None, defaults to config.yaml in the same directory.
        """
        if config_path is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(config_dir, 'config.yaml')
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Device configuration
        self.device = torch.device(
            self.config['device'] if torch.cuda.is_available() else "cpu"
        )
        
        # Model parameters
        self.d_model = self.config['model']['d_model']
        self.n_layers = self.config['model']['n_layers']
        self.n_heads = self.config['model']['n_heads']
        self.ffn_hidden = self.config['model']['ffn_hidden']
        self.max_seq_length = self.config['model']['max_seq_length']
        self.dropout = self.config['model']['dropout']
        
        # Training parameters
        self.batch_size = self.config['training']['batch_size']
        self.init_lr = self.config['training']['init_lr']
        self.factor = self.config['training']['factor']
        self.adam_eps = self.config['training']['adam_eps']
        self.patience = self.config['training']['patience']
        self.warmup_steps = self.config['training']['warmup_steps']
        self.epochs = self.config['training']['epochs']
        self.clip = self.config['training']['clip']
        self.weight_decay = self.config['training']['weight_decay']
        
        # GPU optimization parameters
        self.fp16 = self.config['training']['fp16']
        self.gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        self.pin_memory = self.config['training']['pin_memory']
        self.num_workers = self.config['training']['num_workers']
        
        # Data parameters
        self.dataset = self.config['data']['dataset']
        self.min_freq = self.config['data']['min_freq']
        
        # Constants
        self.inf = float('inf')
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert hyperparameters to a dictionary.
        
        Returns:
            Dict containing all hyperparameters.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'config'}
        
    def __str__(self) -> str:
        """
        String representation of hyperparameters.
        
        Returns:
            String with all hyperparameters.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)


# Create a global instance for easy importing
params = HyperParameters()


if __name__ == "__main__":
    # Print hyperparameters when run directly
    print(params) 