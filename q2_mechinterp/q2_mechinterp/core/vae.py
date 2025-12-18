"""
Base Variational Autoencoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseVAE(nn.Module, ABC):
    """
    Abstract base class for Variational Autoencoders.
    
    Provides common interface and loss functions for all VAE implementations.
    Subclasses should implement encode, decode, and reparameterize methods.
    
    Args:
        input_dim: Dimension of input features.
        latent_dim: Dimension of latent space.
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Tuple of (mu, logvar) tensors for latent distribution.
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim).
        
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        pass
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
        
        Returns:
            Sampled latent tensor.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor.
        
        Returns:
            Tuple of (reconstruction, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_latent(self, x: torch.Tensor, return_std: bool = False) -> torch.Tensor:
        """
        Get latent representation (mean) for input.
        
        Args:
            x: Input tensor.
            return_std: If True, also return standard deviation.
        
        Returns:
            Latent mean tensor, or tuple of (mean, std) if return_std=True.
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            if return_std:
                std = torch.exp(0.5 * logvar)
                return mu, std
            return mu


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss with optional beta-VAE weighting and L1 regularization.
    
    Args:
        recon_x: Reconstructed input.
        x: Original input.
        mu: Latent mean.
        logvar: Latent log variance.
        beta: Weight for KL divergence (beta-VAE).
        alpha: Weight for L1 regularization on latent space.
    
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss, l1_loss).
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence with clamping for stability
    logvar_clamped = torch.clamp(logvar, -10, 10)
    kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    
    # L1 regularization for sparsity
    l1_loss = alpha * torch.mean(torch.abs(mu)) if alpha > 0 else torch.tensor(0.0, device=mu.device)
    
    total_loss = recon_loss + beta * kl_loss + l1_loss
    
    # NaN safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN/Inf detected in loss, returning fallback")
        total_loss = torch.tensor(1e6, device=total_loss.device, requires_grad=True)
        recon_loss = torch.tensor(1e6, device=recon_loss.device)
        kl_loss = torch.tensor(0.0, device=kl_loss.device)
        l1_loss = torch.tensor(0.0, device=l1_loss.device)
    
    return total_loss, recon_loss, kl_loss, l1_loss


class ConcreteVAE(BaseVAE):
    """
    Concrete implementation of VAE with configurable architecture.
    
    Args:
        input_dim: Dimension of input features.
        hidden_dims: List of hidden layer dimensions for encoder/decoder.
        latent_dim: Dimension of latent space.
        dropout_rate: Dropout rate for regularization.
        use_batch_norm: Whether to use batch normalization.
        activation: Activation function ('leaky_relu', 'relu', 'elu').
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 50,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'leaky_relu'
    ):
        super().__init__(input_dim, latent_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Select activation function
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
