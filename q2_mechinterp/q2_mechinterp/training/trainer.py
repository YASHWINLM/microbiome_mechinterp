"""
Training utilities for VAE models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import Callable, Optional, Tuple, List, Dict, Any
import os

from q2_mechinterp.core.utils import EarlyStopping, check_tensor_validity


class VAETrainer:
    """
    Generic trainer for VAE models with robust error handling.
    
    Supports custom loss functions, learning rate scheduling,
    and comprehensive logging.
    
    Args:
        model: VAE model to train.
        device: PyTorch device.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        loss_fn: Loss function (default uses vae_loss from core).
    
    Example:
        >>> from q2_mechinterp import MicrobiomeVAE, VAETrainer
        >>> vae = MicrobiomeVAE(input_dim=500, latent_dim=50)
        >>> trainer = VAETrainer(vae, device='cuda')
        >>> train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_fn: Optional[Callable] = None,
        learning_rate: Optional[float] = None,
        gradient_clip: float = 1.0,
        mixed_precision: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate if learning_rate is not None else lr
        self.loss_fn = loss_fn
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        self.scheduler = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'l1_loss': []
        }
    
    def set_scheduler(
        self,
        scheduler_type: str = 'plateau',
        **kwargs
    ) -> None:
        """
        Set learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('plateau', 'cosine', 'step').
            **kwargs: Additional arguments for scheduler.
        """
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        beta: float = 0.01,
        alpha: float = 0.001,
        noise_factor: float = 0.01,
        grad_clip: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            beta: KL divergence weight.
            alpha: L1 regularization weight.
            noise_factor: Noise factor for data augmentation.
            grad_clip: Gradient clipping value (uses self.gradient_clip if None).
        
        Returns:
            Tuple of (total_loss, loss_components dict).
        """
        self.model.train()
        
        if grad_clip is None:
            grad_clip = self.gradient_clip
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_l1 = 0
        valid_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(self.device)
            
            # Skip invalid batches
            if not check_tensor_validity(data, "input"):
                continue
            
            # Add small noise for regularization
            if noise_factor > 0:
                data = data + torch.randn_like(data) * noise_factor * 0.1
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    recon_batch, mu, logvar = self.model(data)
                    
                    # Compute loss
                    if self.loss_fn is not None:
                        loss, recon_loss, kl_loss, l1_loss = self.loss_fn(
                            recon_batch, data, mu, logvar, beta, alpha
                        )
                    else:
                        from q2_mechinterp.core.vae import vae_loss
                        loss, recon_loss, kl_loss, l1_loss = vae_loss(
                            recon_batch, data, mu, logvar, beta, alpha
                        )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                
                # Unscale and clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                # Step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                recon_batch, mu, logvar = self.model(data)
                
                # Check output validity
                if not (check_tensor_validity(recon_batch, "reconstruction") and
                        check_tensor_validity(mu, "mu") and
                        check_tensor_validity(logvar, "logvar")):
                    continue
                
                # Compute loss
                if self.loss_fn is not None:
                    loss, recon_loss, kl_loss, l1_loss = self.loss_fn(
                        recon_batch, data, mu, logvar, beta, alpha
                    )
                else:
                    from q2_mechinterp.core.vae import vae_loss
                    loss, recon_loss, kl_loss, l1_loss = vae_loss(
                        recon_batch, data, mu, logvar, beta, alpha
                    )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check gradient validity
                valid_grads = True
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_grads = False
                            break
                
                if not valid_grads:
                    continue
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_l1 += l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
            valid_batches += 1
        
        if valid_batches == 0:
            print("Warning: No valid batches in this epoch!")
            return float('inf'), {}
        
        avg_loss = total_loss / valid_batches
        loss_components = {
            'recon': total_recon / valid_batches,
            'kl': total_kl / valid_batches,
            'l1': total_l1 / valid_batches
        }
        
        return avg_loss, loss_components
    
    def validate(
        self,
        val_loader: DataLoader,
        beta: float = 0.01,
        alpha: float = 0.001
    ) -> float:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader.
            beta: KL divergence weight.
            alpha: L1 regularization weight.
        
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for data in val_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(self.device)
                
                if not check_tensor_validity(data, "val_input"):
                    continue
                
                recon_batch, mu, logvar = self.model(data)
                
                if not (check_tensor_validity(recon_batch, "val_recon") and
                        check_tensor_validity(mu, "val_mu") and
                        check_tensor_validity(logvar, "val_logvar")):
                    continue
                
                if self.loss_fn is not None:
                    loss, _, _, _ = self.loss_fn(
                        recon_batch, data, mu, logvar, beta, alpha
                    )
                else:
                    from q2_mechinterp.core.vae import vae_loss
                    loss, _, _, _ = vae_loss(
                        recon_batch, data, mu, logvar, beta, alpha
                    )
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1
        
        if valid_batches == 0:
            return float('inf')
        
        return total_loss / valid_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        beta: float = 0.01,
        alpha: float = 0.001,
        noise_factor: float = 0.01,
        patience: int = 15,
        grad_clip: float = 0.1,
        save_path: Optional[str] = None,
        verbose: int = 10,
        show_progress: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Train the VAE model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            epochs: Number of training epochs.
            beta: KL divergence weight.
            alpha: L1 regularization weight.
            noise_factor: Noise for data augmentation.
            patience: Early stopping patience.
            grad_clip: Gradient clipping value.
            save_path: Path to save best model.
            verbose: Print progress every N epochs (0 for silent).
            show_progress: Show progress bar for epochs.
        
        Returns:
            Tuple of (train_losses, val_losses).
        """
        from q2_mechinterp.core.logging import get_progress_bar
        
        train_losses = []
        val_losses = []
        
        early_stopping = EarlyStopping(patience=patience)
        best_model_state = None
        
        if verbose > 0:
            print(f"Training VAE for {epochs} epochs...")
            print(f"Parameters: beta={beta}, alpha={alpha}, lr={self.lr}")
        
        start_time = time.time()
        
        # Get progress bar iterator
        epoch_iter = range(1, epochs + 1)
        if show_progress and verbose > 0:
            epoch_iter = get_progress_bar(
                epoch_iter,
                desc="Training",
                total=epochs
            )
        
        for epoch in epoch_iter:
            # Training
            train_loss, loss_components = self.train_epoch(
                train_loader, beta, alpha, noise_factor, grad_clip
            )
            train_losses.append(train_loss)
            
            # Validation
            val_loss = float('inf')
            if val_loader is not None:
                val_loss = self.validate(val_loader, beta, alpha)
                val_losses.append(val_loss)
            
            # Handle infinite losses
            if train_loss == float('inf') or val_loss == float('inf'):
                if verbose > 0:
                    print(f"Infinite loss at epoch {epoch}, stopping training")
                break
            
            # Update history
            self.history['train_loss'].append(train_loss)
            if val_loader is not None:
                self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader else train_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            if verbose > 0 and not show_progress and (epoch % verbose == 0 or epoch == 1):
                msg = f"Epoch {epoch}: Train={train_loss:.6f}"
                if val_loader is not None:
                    msg += f", Val={val_loss:.6f}"
                print(msg)
            
            # Update progress bar description
            if show_progress and hasattr(epoch_iter, 'set_postfix'):
                postfix = {'train': f'{train_loss:.4f}'}
                if val_loader is not None:
                    postfix['val'] = f'{val_loss:.4f}'
                epoch_iter.set_postfix(postfix)
            
            # Early stopping
            if val_loader is not None:
                if early_stopping(val_loss, self.model):
                    if verbose > 0:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Save best model
            if save_path is not None and val_loader is not None:
                if val_loss <= min(val_losses):
                    torch.save(self.model.state_dict(), save_path)
        
        training_time = time.time() - start_time
        if verbose > 0:
            print(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model if early stopping was used
        if early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)
            if verbose > 0:
                print("Loaded best model weights")
        
        return train_losses, val_losses
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        train_loss: float,
        val_loss: float
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': self.history
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        return checkpoint


def create_data_loaders(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch data loaders from numpy arrays.
    
    Args:
        X_train: Training data.
        X_val: Validation data (optional).
        batch_size: Batch size.
        shuffle: Whether to shuffle training data.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_tensor = torch.FloatTensor(X_train)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_loader = None
    if X_val is not None:
        val_tensor = torch.FloatTensor(X_val)
        val_dataset = TensorDataset(val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
