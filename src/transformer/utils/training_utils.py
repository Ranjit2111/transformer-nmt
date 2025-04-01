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
from torchtext.legacy.data import BucketIterator, Batch
import numpy as np
from tqdm import tqdm

from src.transformer.components.transformer import TransformerNMT


class Trainer:
    """
    Trainer class for the TransformerNMT model.
    """
    
    def __init__(
        self,
        model: TransformerNMT,
        src_pad_idx: int,
        lr: float,
        factor: float,
        patience: int,
        warmup_steps: int,
        clip: float,
        weight_decay: float,
        adam_eps: float,
        fp16: bool,
        grad_accumulation_steps: int,
        save_dir: str = "results/checkpoints",
        log_interval: int = 100
    ):
        """
        Initialize the trainer.
        
        Args:
            model: TransformerNMT model to train
            src_pad_idx: Source padding token index
            lr: Initial learning rate
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which learning rate will be reduced
            warmup_steps: Number of warmup steps for learning rate scheduler
            clip: Gradient clipping value
            weight_decay: Weight decay for optimizer
            adam_eps: Epsilon value for Adam optimizer
            fp16: Whether to use mixed precision training
            grad_accumulation_steps: Number of steps to accumulate gradients
            save_dir: Directory to save model checkpoints
            log_interval: Interval for logging training progress
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.src_pad_idx = src_pad_idx
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=adam_eps,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True
        )
        
        # Loss function that ignores padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
        
        # Training parameters
        self.clip = clip
        self.warmup_steps = warmup_steps
        self.fp16 = fp16
        self.grad_accumulation_steps = grad_accumulation_steps
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        
        # Initialize mixed precision scaler if FP16 is enabled
        self.scaler = GradScaler() if fp16 else None
        
        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking variables
        self.global_step = 0
        self.best_valid_loss = float('inf')
        
    def train_epoch(self, iterator: BucketIterator) -> float:
        """
        Train the model for one epoch.
        
        Args:
            iterator: Data iterator for training
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0
        total_batches = len(iterator)
        
        pbar = tqdm(total=total_batches, desc="Training")
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            # Forward pass
            if self.fp16:
                with autocast():
                    output = self.model(src, trg[:, :-1])
                    output_dim = output.shape[-1]
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:, 1:].contiguous().view(-1)
                    loss = self.criterion(output, trg)
                    loss = loss / self.grad_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % self.grad_accumulation_steps == 0 or i == total_batches - 1:
                    # Clip gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = self.criterion(output, trg)
                loss = loss / self.grad_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % self.grad_accumulation_steps == 0 or i == total_batches - 1:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update tracking variables
            epoch_loss += loss.item() * self.grad_accumulation_steps
            self.global_step += 1
            
            # Update progress bar
            if i % self.log_interval == 0:
                pbar.set_postfix({"loss": f"{epoch_loss / (i + 1):.4f}"})
                pbar.update(min(self.log_interval, i - pbar.n))
        
        pbar.close()
        return epoch_loss / total_batches
        
    def evaluate(self, iterator: BucketIterator) -> float:
        """
        Evaluate the model on validation or test data.
        
        Args:
            iterator: Data iterator for evaluation
            
        Returns:
            Average loss for the evaluation
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in iterator:
                src = batch.src
                trg = batch.trg
                
                # Forward pass with teacher forcing
                output = self.model(src, trg[:, :-1])
                
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # Calculate loss
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
                
        return epoch_loss / len(iterator)
        
    def train(
        self, 
        train_iterator: BucketIterator, 
        valid_iterator: BucketIterator, 
        epochs: int
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_iterator: Data iterator for training
            valid_iterator: Data iterator for validation
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary with training history
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Training history
        history = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rates': []
        }
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_iterator)
            history['train_loss'].append(train_loss)
            
            # Evaluate on validation set
            valid_loss = self.evaluate(valid_iterator)
            history['valid_loss'].append(valid_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Calculate training time
            end_time = time.time()
            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)
            
            # Update learning rate based on validation loss
            if epoch >= self.warmup_steps:
                self.scheduler.step(valid_loss)
            
            # Save checkpoint if validation loss improves
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self._save_checkpoint(f"model_epoch{epoch}_loss{valid_loss:.4f}.pt")
            
            # Print epoch summary
            print(f"Epoch: {epoch+1:03}/{epochs} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.4f} | PPL: {math.exp(train_loss):7.3f}")
            print(f"\tValid Loss: {valid_loss:.4f} | PPL: {math.exp(valid_loss):7.3f}")
            print(f"\tLearning Rate: {current_lr}")
        
        return history
    
    def _save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Filename for checkpoint
        """
        checkpoint_path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'global_step': self.global_step,
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.global_step = checkpoint['global_step']
        print(f"Model loaded from {checkpoint_path}")
    
    @staticmethod
    def _epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
        """
        Calculate time taken for epoch.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Tuple of (minutes, seconds)
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs 