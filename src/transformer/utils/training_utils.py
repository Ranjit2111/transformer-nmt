"""
Training utilities for the TransformerNMT model.
"""
import math
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Remove torchtext.legacy imports
# Define fallbacks without mentioning legacy
class Batch:
    """Simple batch container with src and trg attributes."""
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

import numpy as np
from tqdm import tqdm

from src.transformer.components.transformer import TransformerNMT

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class NoamScheduler(_LRScheduler):
    """
    Learning rate scheduler from 'Attention is All You Need' paper.
    
    It increases the learning rate linearly for the first warmup_steps, and then
    decreases it proportionally to the inverse square root of the step number.
    """
    
    def __init__(
        self, 
        optimizer: optim.Optimizer, 
        model_size: int, 
        warmup_steps: int, 
        factor: float = 1.0, 
        last_epoch: int = -1
    ):
        """
        Initialize Noam scheduler.
        
        Args:
            optimizer: The optimizer to adjust
            model_size: Dimensionality of the model
            warmup_steps: Number of warmup steps
            factor: Scaling factor for the learning rate
            last_epoch: The index of the last epoch
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(NoamScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate based on current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        step = max(1, self._step_count)
        
        # Calculate scaling factor based on step and warmup steps
        scale = self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        
        return [scale for _ in self.base_lrs]


class Trainer:
    """Trainer for the TransformerNMT model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        clip: float = 1.0,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        print_every: int = 100,
        save_every: int = 1000
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            optimizer: Optimizer for updating weights
            criterion: Loss function
            scheduler: Learning rate scheduler
            clip: Gradient clipping threshold
            log_dir: Directory for TensorBoard logs
            save_dir: Directory for saving checkpoints
            print_every: Print training info every N steps
            save_every: Save checkpoint every N steps
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip = clip
        self.save_dir = save_dir
        self.print_every = print_every
        self.save_every = save_every
        
        # Create log directory and writer if specified
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            
        # Create save directory if specified
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            
        # Initialize tracking variables
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        self.valid_losses = []
        self.train_ppls = []
        self.valid_ppls = []
        self.best_valid_loss = float('inf')
        
    def train_epoch(
        self, 
        iterator: Any, 
        pad_idx: int
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            iterator: Data iterator
            pad_idx: Padding token index
            
        Returns:
            Tuple of (epoch loss, epoch perplexity)
        """
        self.model.train()
        epoch_loss = 0
        
        start_time = time.time()
        
        for i, batch in enumerate(iterator):
            # Extract source and target from batch
            if hasattr(batch, 'src') and hasattr(batch, 'trg'):
                src = batch.src
                trg = batch.trg
            else:
                # Handle case when batch is a tuple of tensors
                if isinstance(batch, tuple) and len(batch) == 2:
                    src, trg = batch
                else:
                    # Try to extract source and target from dictionary or other format
                    try:
                        src = batch['src']
                        trg = batch['trg']
                    except:
                        raise ValueError(f"Unrecognized batch format: {type(batch)}")
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # The model expects (src, trg) as inputs
            output = self.model(src, trg[:, :-1])
            
            # The loss function expects (output, target)
            # Reshape: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            
            # Exclude the <sos> token (first token)
            # Reshape: [batch_size, seq_len] -> [batch_size * seq_len]
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = self.criterion(output, trg)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            self.optimizer.step()
            
            # Update learning rate scheduler if it exists
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Update tracking variables
            epoch_loss += loss.item()
            self.step += 1
            
            # Log training info
            if self.step % self.print_every == 0:
                step_time = time.time() - start_time
                steps_per_sec = self.print_every / step_time
                
                # Calculate perplexity
                step_loss = epoch_loss / (i + 1)
                step_ppl = math.exp(step_loss)
                
                print(f'Epoch: {self.epoch+1} | '
                      f'Step: {self.step} | '
                      f'Loss: {step_loss:.4f} | '
                      f'PPL: {step_ppl:.4f} | '
                      f'Steps/s: {steps_per_sec:.2f}')
                
                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', step_loss, self.step)
                    self.writer.add_scalar('train/ppl', step_ppl, self.step)
                    
                    # Log learning rate if scheduler is used
                    if self.scheduler is not None:
                        self.writer.add_scalar(
                            'train/lr', 
                            self.scheduler.get_lr()[0], 
                            self.step
                        )
                        
                start_time = time.time()
                
            # Save checkpoint
            if self.save_dir is not None and self.step % self.save_every == 0:
                self.save_checkpoint(f'step_{self.step}')
                
        # Calculate epoch loss and perplexity
        epoch_loss = epoch_loss / len(iterator)
        epoch_ppl = math.exp(epoch_loss)
        
        return epoch_loss, epoch_ppl
    
    def evaluate(
        self, 
        iterator: Any, 
        pad_idx: int
    ) -> Tuple[float, float]:
        """
        Evaluate the model on the provided data.
        
        Args:
            iterator: Data iterator
            pad_idx: Padding token index
            
        Returns:
            Tuple of (validation loss, validation perplexity)
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                # Extract source and target from batch
                if hasattr(batch, 'src') and hasattr(batch, 'trg'):
                    src = batch.src
                    trg = batch.trg
                else:
                    # Handle case when batch is a tuple of tensors
                    if isinstance(batch, tuple) and len(batch) == 2:
                        src, trg = batch
                    else:
                        # Try to extract source and target from dictionary or other format
                        try:
                            src = batch['src']
                            trg = batch['trg']
                        except:
                            raise ValueError(f"Unrecognized batch format: {type(batch)}")
                
                # Forward pass
                output = self.model(src, trg[:, :-1])
                
                # Reshape for loss calculation
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # Calculate loss
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        
        # Calculate epoch loss and perplexity
        epoch_loss = epoch_loss / len(iterator)
        epoch_ppl = math.exp(epoch_loss)
        
        return epoch_loss, epoch_ppl
    
    def train(
        self, 
        train_iterator: Any, 
        valid_iterator: Any, 
        pad_idx: int, 
        n_epochs: int, 
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            train_iterator: Training data iterator
            valid_iterator: Validation data iterator
            pad_idx: Padding token index
            n_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping (if None, no early stopping)
            
        Returns:
            Dictionary of training history
        """
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            
            # Train
            start_time = time.time()
            train_loss, train_ppl = self.train_epoch(train_iterator, pad_idx)
            train_time = time.time() - start_time
            
            # Evaluate
            valid_loss, valid_ppl = self.evaluate(valid_iterator, pad_idx)
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_ppls.append(train_ppl)
            self.valid_ppls.append(valid_ppl)
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('epoch/train_loss', train_loss, epoch+1)
                self.writer.add_scalar('epoch/valid_loss', valid_loss, epoch+1)
                self.writer.add_scalar('epoch/train_ppl', train_ppl, epoch+1)
                self.writer.add_scalar('epoch/valid_ppl', valid_ppl, epoch+1)
                
            # Print epoch summary
            print(f'Epoch: {epoch+1} | Time: {train_time:.2f}s')
            print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {train_ppl:.4f}')
            print(f'\tValid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.4f}')
            
            # Save best model
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                if self.save_dir is not None:
                    self.save_checkpoint('best')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Save epoch checkpoint
            if self.save_dir is not None:
                self.save_checkpoint(f'epoch_{epoch+1}')
                
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
                
        # Return training history
        return {
            'train_loss': self.train_losses,
            'valid_loss': self.valid_losses,
            'train_ppl': self.train_ppls,
            'valid_ppl': self.valid_ppls
        }
    
    def save_checkpoint(self, name: str):
        """
        Save model checkpoint.
        
        Args:
            name: Name of the checkpoint
        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_ppls': self.train_ppls,
            'valid_ppls': self.valid_ppls,
            'best_valid_loss': self.best_valid_loss
        }
        
        # Add scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, os.path.join(self.save_dir, f'{name}.pt'))
        
    def load_checkpoint(self, path: str, map_location: Optional[torch.device] = None):
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            map_location: Device to load the checkpoint to
        """
        checkpoint = torch.load(path, map_location=map_location)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.valid_losses = checkpoint['valid_losses']
        self.train_ppls = checkpoint['train_ppls']
        self.valid_ppls = checkpoint['valid_ppls']
        self.best_valid_loss = checkpoint['best_valid_loss']
        
        # Load scheduler state if it exists
        if self.scheduler is not None and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def plot_loss(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot training and validation loss.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.valid_losses, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Loss')
        plt.show()
        
    def plot_perplexity(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot training and validation perplexity.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(self.train_ppls, label='Train')
        plt.plot(self.valid_ppls, label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        plt.title('Perplexity')
        plt.show() 