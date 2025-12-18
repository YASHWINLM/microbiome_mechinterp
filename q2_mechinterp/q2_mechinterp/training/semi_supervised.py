"""
Semi-supervised VAE (M2 model) for combined labeled/unlabeled training.

When only a subset of samples have labels, this model can leverage
both labeled and unlabeled data to learn better representations.

Reference:
    Kingma et al. "Semi-Supervised Learning with Deep Generative Models"
    NeurIPS 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import warnings

from q2_mechinterp.core.vae import BaseVAE


class SemiSupervisedVAE(nn.Module):
    """
    Semi-supervised VAE (M2 model).
    
    For data where only some samples have labels, this model:
    1. Learns a generative model p(x|z,y) that conditions on labels
    2. Learns a classifier q(y|x) for inferring labels
    3. Uses labeled data for supervised classification
    4. Uses unlabeled data for unsupervised representation learning
    
    Architecture:
    - Encoder q(z|x): Maps input to latent space
    - Classifier q(y|x): Predicts label from input
    - Decoder p(x|z,y): Reconstructs input from latent + label
    
    Example:
        >>> # Create model
        >>> model = SemiSupervisedVAE(
        ...     input_dim=500,
        ...     n_classes=5,
        ...     latent_dim=50
        ... )
        >>> 
        >>> # Forward with labels (labeled data)
        >>> recon, z, mu, logvar, y_pred = model(x, y=labels)
        >>> 
        >>> # Forward without labels (unlabeled data)
        >>> recon, z, mu, logvar, y_pred = model(x)
    
    Args:
        input_dim: Input feature dimension
        n_classes: Number of classes
        hidden_dims: Hidden layer dimensions for encoder/decoder
        latent_dim: Latent space dimension
        classifier_dims: Hidden dimensions for classifier (None uses hidden_dims)
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 50,
        classifier_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        classifier_dims = classifier_dims or [hidden_dims[0] // 2]
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Encoder q(z|x)
        self.encoder = self._build_encoder(input_dim, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Classifier q(y|x) - shares early layers with encoder for efficiency
        self.classifier = self._build_classifier(
            hidden_dims[-1],  # Takes encoder output
            classifier_dims,
            n_classes
        )
        
        # Decoder p(x|z,y)
        # Input is z concatenated with one-hot y
        decoder_input_dim = latent_dim + n_classes
        self.decoder = self._build_decoder(
            decoder_input_dim,
            hidden_dims,
            input_dim
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int]
    ) -> nn.Module:
        """Build encoder network."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_classifier(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_classes: int
    ) -> nn.Module:
        """Build classifier network."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        return nn.Sequential(*layers)
    
    def _build_decoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ) -> nn.Module:
        """Build decoder network."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[0], output_dim))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (mu, logvar)
        """
        h = self.encoder(self.input_norm(x))
        return self.fc_mu(h), self.fc_var(h)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input (get label logits).
        
        Args:
            x: Input tensor
        
        Returns:
            Class logits (batch_size, n_classes)
        """
        h = self.encoder(self.input_norm(x))
        return self.classifier(h)
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(
        self,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode from latent + label.
        
        Args:
            z: Latent tensor (batch_size, latent_dim)
            y: Label tensor - can be one-hot or class indices
        
        Returns:
            Reconstructed input (batch_size, input_dim)
        """
        # Convert to one-hot if needed
        if y.dim() == 1:
            y_onehot = F.one_hot(y, self.n_classes).float()
        else:
            y_onehot = y
        
        # Concatenate z and y
        zy = torch.cat([z, y_onehot], dim=1)
        
        return self.decoder(zy)
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            y: Optional labels (batch_size,) or one-hot (batch_size, n_classes)
               If None, uses predicted labels
        
        Returns:
            Tuple of (reconstruction, z, mu, logvar, y_logits)
        """
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Classify
        y_logits = self.classify(x)
        
        # Determine labels for decoding
        if y is not None:
            # Use provided labels
            if y.dim() == 1:
                y_for_decode = F.one_hot(y, self.n_classes).float()
            else:
                y_for_decode = y
        else:
            # Use predicted labels (soft)
            y_for_decode = F.softmax(y_logits, dim=1)
        
        # Decode
        recon = self.decode(z, y_for_decode)
        
        return recon, z, mu, logvar, y_logits
    
    def get_latent(
        self,
        x: torch.Tensor,
        return_std: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get latent representation.
        
        Args:
            x: Input tensor
            return_std: Whether to return std as well
        
        Returns:
            Latent mean (and optionally std)
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            if return_std:
                return mu, torch.exp(0.5 * logvar)
            return mu
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input tensor
        
        Returns:
            Predicted class indices (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.classify(x)
            return logits.argmax(dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor
        
        Returns:
            Class probabilities (batch_size, n_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.classify(x)
            return F.softmax(logits, dim=1)


def semi_supervised_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y_logits: torch.Tensor,
    y_true: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    alpha: float = 100.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Semi-supervised VAE loss.
    
    For labeled data: reconstruction + KL + classification
    For unlabeled data: reconstruction + KL + entropy regularization
    
    Args:
        x: Input tensor
        recon: Reconstruction
        mu: Latent mean
        logvar: Latent log variance
        y_logits: Classification logits
        y_true: True labels (None for unlabeled)
        beta: KL divergence weight
        alpha: Classification/entropy weight
    
    Returns:
        Tuple of (total_loss, loss_components dict)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Classification loss
    if y_true is not None:
        # Supervised: cross-entropy
        class_loss = F.cross_entropy(y_logits, y_true)
    else:
        # Unsupervised: negative entropy (encourage confident predictions)
        y_probs = F.softmax(y_logits, dim=1)
        entropy = -torch.sum(y_probs * torch.log(y_probs + 1e-8), dim=1)
        class_loss = torch.mean(entropy)  # Minimize entropy = maximize confidence
    
    total_loss = recon_loss + beta * kl_loss + alpha * class_loss
    
    components = {
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'class_loss': class_loss
    }
    
    return total_loss, components


class SemiSupervisedTrainer:
    """
    Trainer for Semi-supervised VAE.
    
    Handles training with both labeled and unlabeled batches,
    balancing supervised and unsupervised objectives.
    
    Example:
        >>> trainer = SemiSupervisedTrainer(model, device)
        >>> 
        >>> history = trainer.train(
        ...     labeled_loader=labeled_loader,
        ...     unlabeled_loader=unlabeled_loader,
        ...     epochs=100
        ... )
    """
    
    def __init__(
        self,
        model: SemiSupervisedVAE,
        device: torch.device,
        lr: float = 1e-3,
        beta: float = 1.0,
        alpha_labeled: float = 100.0,
        alpha_unlabeled: float = 0.1
    ):
        """
        Initialize trainer.
        
        Args:
            model: SemiSupervisedVAE model
            device: PyTorch device
            lr: Learning rate
            beta: KL divergence weight
            alpha_labeled: Classification weight for labeled data
            alpha_unlabeled: Entropy weight for unlabeled data
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.alpha_labeled = alpha_labeled
        self.alpha_unlabeled = alpha_unlabeled
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'labeled_loss': [],
            'unlabeled_loss': [],
            'accuracy': []
        }
    
    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            labeled_loader: DataLoader for (x, y) labeled pairs
            unlabeled_loader: Optional DataLoader for unlabeled x
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_labeled_loss = 0
        total_unlabeled_loss = 0
        n_labeled = 0
        n_unlabeled = 0
        
        # Create iterator for unlabeled data
        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)
        
        for batch_idx, (x_labeled, y_labeled) in enumerate(labeled_loader):
            x_labeled = x_labeled.to(self.device)
            y_labeled = y_labeled.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Labeled forward
            recon_l, z_l, mu_l, logvar_l, y_logits_l = self.model(x_labeled, y_labeled)
            
            loss_labeled, _ = semi_supervised_loss(
                x_labeled, recon_l, mu_l, logvar_l, y_logits_l,
                y_true=y_labeled,
                beta=self.beta,
                alpha=self.alpha_labeled
            )
            
            total_labeled_loss += loss_labeled.item() * len(x_labeled)
            n_labeled += len(x_labeled)
            
            # Unlabeled forward (if available)
            loss_unlabeled = torch.tensor(0.0, device=self.device)
            
            if unlabeled_loader is not None:
                try:
                    x_unlabeled = next(unlabeled_iter)
                    if isinstance(x_unlabeled, (list, tuple)):
                        x_unlabeled = x_unlabeled[0]
                    x_unlabeled = x_unlabeled.to(self.device)
                    
                    recon_u, z_u, mu_u, logvar_u, y_logits_u = self.model(x_unlabeled)
                    
                    loss_unlabeled, _ = semi_supervised_loss(
                        x_unlabeled, recon_u, mu_u, logvar_u, y_logits_u,
                        y_true=None,
                        beta=self.beta,
                        alpha=self.alpha_unlabeled
                    )
                    
                    total_unlabeled_loss += loss_unlabeled.item() * len(x_unlabeled)
                    n_unlabeled += len(x_unlabeled)
                    
                except StopIteration:
                    # Restart unlabeled iterator
                    unlabeled_iter = iter(unlabeled_loader)
            
            # Combined loss
            total_loss = loss_labeled + loss_unlabeled
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return {
            'labeled_loss': total_labeled_loss / n_labeled if n_labeled > 0 else 0,
            'unlabeled_loss': total_unlabeled_loss / n_unlabeled if n_unlabeled > 0 else 0
        }
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: DataLoader for (x, y) validation pairs
        
        Returns:
            Tuple of (val_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        n_samples = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                recon, z, mu, logvar, y_logits = self.model(x, y)
                
                loss, _ = semi_supervised_loss(
                    x, recon, mu, logvar, y_logits,
                    y_true=y,
                    beta=self.beta,
                    alpha=self.alpha_labeled
                )
                
                total_loss += loss.item() * len(x)
                correct += (y_logits.argmax(dim=1) == y).sum().item()
                n_samples += len(x)
        
        return total_loss / n_samples, correct / n_samples
    
    def train(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: int = 10
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            labeled_loader: DataLoader for labeled data
            unlabeled_loader: Optional DataLoader for unlabeled data
            val_loader: Optional validation DataLoader
            epochs: Number of epochs
            verbose: Print progress every N epochs
        
        Returns:
            Training history dictionary
        """
        for epoch in range(1, epochs + 1):
            # Train
            train_losses = self.train_epoch(labeled_loader, unlabeled_loader)
            
            self.history['labeled_loss'].append(train_losses['labeled_loss'])
            self.history['unlabeled_loss'].append(train_losses['unlabeled_loss'])
            self.history['train_loss'].append(
                train_losses['labeled_loss'] + train_losses['unlabeled_loss']
            )
            
            # Validate
            if val_loader is not None:
                val_loss, accuracy = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['accuracy'].append(accuracy)
            
            # Print progress
            if epoch % verbose == 0 or epoch == 1:
                msg = f"Epoch {epoch}: Labeled={train_losses['labeled_loss']:.4f}"
                if unlabeled_loader is not None:
                    msg += f", Unlabeled={train_losses['unlabeled_loss']:.4f}"
                if val_loader is not None:
                    msg += f", Val={self.history['val_loss'][-1]:.4f}"
                    msg += f", Acc={self.history['accuracy'][-1]:.3f}"
                print(msg)
        
        return self.history


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'SemiSupervisedVAE',
    'semi_supervised_loss',
    'SemiSupervisedTrainer',
]
