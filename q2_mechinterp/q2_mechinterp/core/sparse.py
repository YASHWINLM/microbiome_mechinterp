"""
Sparse Autoencoder for interpretable feature extraction from VAE latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable features from VAE latent space.
    
    Uses L1 regularization to encourage sparse activations, making features
    more interpretable and potentially mapping to biological concepts.
    
    Args:
        latent_dim: Dimension of input (VAE latent space).
        sparse_dim: Dimension of sparse representation (typically > latent_dim).
        l1_reg: L1 regularization coefficient for sparsity.
        use_batch_norm: Whether to use batch normalization in encoder.
    """
    
    def __init__(
        self,
        latent_dim: int,
        sparse_dim: int,
        l1_reg: float = 0.001,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.sparse_dim = sparse_dim
        self.l1_reg = l1_reg
        
        # Encoder: latent -> sparse (expanded)
        if use_batch_norm:
            self.encoder = nn.Sequential(
                nn.Linear(latent_dim, sparse_dim),
                nn.ReLU(),
                nn.BatchNorm1d(sparse_dim)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(latent_dim, sparse_dim),
                nn.ReLU()
            )
        
        # Decoder: sparse -> latent (compressed back)
        self.decoder = nn.Linear(sparse_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor from VAE latent space.
        
        Returns:
            Reconstructed latent representation.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_sparse_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sparse feature activations.
        
        Args:
            x: Input tensor from VAE latent space.
        
        Returns:
            Sparse feature activations.
        """
        return self.encoder(x)
    
    def get_encoder_weights(self) -> torch.Tensor:
        """Get encoder weight matrix for analysis."""
        # Get the linear layer (first module in encoder)
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                return module.weight.data.clone()
        raise RuntimeError("No Linear layer found in encoder")
    
    def get_decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix for analysis."""
        return self.decoder.weight.data.clone()


def sparse_autoencoder_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    model: SparseAutoencoder,
    l1_reg: Optional[float] = None
) -> torch.Tensor:
    """
    Compute sparse autoencoder loss with L1 regularization.
    
    Args:
        y_true: Original latent vectors.
        y_pred: Reconstructed latent vectors.
        model: Sparse autoencoder model (for weight access).
        l1_reg: L1 regularization coefficient (uses model.l1_reg if None).
    
    Returns:
        Total loss (MSE + L1).
    """
    l1_reg = l1_reg if l1_reg is not None else model.l1_reg
    
    # Reconstruction loss
    mse_loss = F.mse_loss(y_pred, y_true)
    
    # L1 regularization on encoder weights
    encoder_weights = model.get_encoder_weights()
    l1_loss = l1_reg * torch.sum(torch.abs(encoder_weights))
    
    return mse_loss + l1_loss


class TopKSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with Top-K activation sparsity constraint.
    
    Instead of L1 regularization, this enforces sparsity by only keeping
    the top-k activations for each sample.
    
    Args:
        latent_dim: Dimension of input (VAE latent space).
        sparse_dim: Dimension of sparse representation.
        k: Number of top activations to keep.
    """
    
    def __init__(self, latent_dim: int, sparse_dim: int, k: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.sparse_dim = sparse_dim
        self.k = k
        
        self.encoder_linear = nn.Linear(latent_dim, sparse_dim)
        self.decoder = nn.Linear(sparse_dim, latent_dim)
    
    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity constraint."""
        # Get top-k values and indices
        topk_vals, topk_idx = torch.topk(x, self.k, dim=1)
        
        # Create sparse output
        sparse_out = torch.zeros_like(x)
        sparse_out.scatter_(1, topk_idx, F.relu(topk_vals))
        
        return sparse_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_activation = self.encoder_linear(x)
        sparse = self.topk_activation(pre_activation)
        return self.decoder(sparse)
    
    def get_sparse_features(self, x: torch.Tensor) -> torch.Tensor:
        pre_activation = self.encoder_linear(x)
        return self.topk_activation(pre_activation)


class DeepSparseAutoencoder(nn.Module):
    """
    Multi-layer Sparse Autoencoder for hierarchical feature learning.
    
    Args:
        latent_dim: Dimension of input (VAE latent space).
        hidden_dims: List of hidden layer dimensions.
        sparse_dim: Final sparse representation dimension.
        l1_reg: L1 regularization coefficient.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list = [128],
        sparse_dim: int = 256,
        l1_reg: float = 0.001
    ):
        super().__init__()
        
        self.l1_reg = l1_reg
        
        # Build encoder
        encoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.extend([
            nn.Linear(prev_dim, sparse_dim),
            nn.ReLU(),
            nn.BatchNorm1d(sparse_dim)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror architecture)
        decoder_layers = []
        prev_dim = sparse_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
    def get_sparse_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
