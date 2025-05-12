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
        
        # Create a tqdm progress bar
        total_batches = len(iterator)
        progress_bar = tqdm(
            enumerate(iterator),
            total=total_batches,
            desc=f"Epoch {self.epoch+1}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
        
        for i, batch in progress_bar:
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
            
            # Update progress bar with current loss
            current_loss = epoch_loss / (i + 1)
            current_ppl = math.exp(current_loss)
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'ppl': f'{current_ppl:.2f}'
            })
            
            # Log training info
            if self.step % self.print_every == 0:
                step_time = time.time() - start_time
                steps_per_sec = self.print_every / step_time
                
                # Calculate perplexity
                step_loss = epoch_loss / (i + 1)
                step_ppl = math.exp(step_loss)
                
                # Log to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', step_loss, self.step)
                    self.writer.add_scalar('train/ppl', step_ppl, self.step)
                    
                    # Log learning rate if scheduler is used
                    if self.scheduler is not None:
                        self.writer.add_scalar(
                            'train/lr', 
                            self.scheduler.get_last_lr()[0], 
                            self.step
                        )
                        
                start_time = time.time()
                
            # Save checkpoint
            if self.save_dir is not None and self.step % self.save_every == 0:
                self.save_checkpoint(f'step_{self.step}')
        
        # Close progress bar
        progress_bar.close()
                
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
        
        # Create a validation progress bar
        total_batches = len(iterator)
        progress_bar = tqdm(
            enumerate(iterator),
            total=total_batches,
            desc="Validating",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
        
        with torch.no_grad():
            for i, batch in progress_bar:
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
                
                # Update progress bar with current loss
                current_loss = epoch_loss / (i + 1)
                current_ppl = math.exp(current_loss)
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'ppl': f'{current_ppl:.2f}'
                })
        
        # Close progress bar
        progress_bar.close()
        
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
        training_start_time = time.time()
        
        print(f"Starting training for {n_epochs} epochs")
        print(f"Training on {len(train_iterator)} batches per epoch")
        print(f"Validating on {len(valid_iterator)} batches per epoch")
        print("-" * 60)
        
        # Create an epochs progress tracker
        epochs_progress = tqdm(
            range(n_epochs),
            desc="Training Progress",
            unit="epoch",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
        
        for epoch in epochs_progress:
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_ppl = self.train_epoch(train_iterator, pad_idx)
            
            # Evaluate
            valid_loss, valid_ppl = self.evaluate(valid_iterator, pad_idx)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time
            
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
                
            # Update progress bar with epoch results
            epochs_progress.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_ppl': f'{train_ppl:.2f}',
                'valid_loss': f'{valid_loss:.4f}',
                'valid_ppl': f'{valid_ppl:.2f}',
                'time': f'{epoch_time:.1f}s'
            })
                
            # Print epoch results
            print(f"Epoch: {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | Valid PPL: {valid_ppl:.4f} | "
                  f"Epoch time: {epoch_time:.2f}s | Total time: {total_time/60:.1f}m")
            
            # Check if this is the best model
            if valid_loss < self.best_valid_loss:
                patience_counter = 0
                self.best_valid_loss = valid_loss
                
                # Save best model
                if self.save_dir is not None:
                    print(f"New best validation loss: {valid_loss:.4f}. Saving checkpoint...")
                    self.save_checkpoint('best_model')
            else:
                patience_counter += 1
                
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break
                
            # Save checkpoint at the end of each epoch
            if self.save_dir is not None:
                self.save_checkpoint(f'epoch_{epoch+1}')
        
        # Close the progress bar
        epochs_progress.close()
        
        # Print final training time
        total_time = time.time() - training_start_time
        print(f"Training complete. Total time: {total_time/60:.2f} minutes")
                
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