"""
Microbiome-specific Variational Autoencoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from q2_mechinterp.core.vae import BaseVAE


class MicrobiomeVAE(BaseVAE):
    """
    Enhanced VAE optimized for microbiome data characteristics.
    
    Handles high sparsity and compositional nature of microbiome data
    with specialized architecture and regularization.
    
    Args:
        input_dim: Number of input features (OGUs/taxa).
        hidden_dims: List of hidden layer dimensions.
        latent_dim: Dimension of latent space.
        dropout_rate: Dropout rate for regularization.
        use_layer_norm: Whether to use layer normalization on input.
    
    Example:
        >>> vae = MicrobiomeVAE(input_dim=500, latent_dim=50)
        >>> x = torch.randn(32, 500)
        >>> recon, mu, logvar = vae(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 50,
        dropout_rate: float = 0.3,
        use_layer_norm: bool = True
    ):
        super().__init__(input_dim, latent_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Input normalization
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = nn.Identity()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights for stable training with sparse data."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        x = self.input_norm(x)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)


def microbiome_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01,
    alpha: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss function optimized for microbiome data with stability checks.
    
    Uses lower default beta (KL weight) for highly sparse data.
    
    Args:
        recon_x: Reconstructed input.
        x: Original input.
        mu: Latent mean.
        logvar: Latent log variance.
        beta: Weight for KL divergence (lower for sparse data).
        alpha: Weight for L1 regularization on latent space.
    
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss, l1_loss).
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence with clamping to prevent explosion
    logvar_clamped = torch.clamp(logvar, -10, 10)
    kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    
    # L1 regularization on latent space for sparsity
    l1_loss = alpha * torch.mean(torch.abs(mu))
    
    total_loss = recon_loss + beta * kl_loss + l1_loss
    
    # NaN/Inf safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN/Inf detected in loss, returning fallback")
        device = total_loss.device
        total_loss = torch.tensor(1e6, device=device, requires_grad=True)
        recon_loss = torch.tensor(1e6, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        l1_loss = torch.tensor(0.0, device=device)
    
    return total_loss, recon_loss, kl_loss, l1_loss


class MultiOmicsVAE(BaseVAE):
    """
    VAE for multi-omics integration (metagenomics + metatranscriptomics).
    
    Uses separate encoders for each data modality and combines them
    in the latent space.
    
    Args:
        metag_dim: Dimension of metagenomics features.
        metat_dim: Dimension of metatranscriptomics features.
        hidden_dims: List of hidden layer dimensions per modality.
        latent_dim: Dimension of shared latent space.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        metag_dim: int,
        metat_dim: int,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 50,
        dropout_rate: float = 0.3
    ):
        super().__init__(metag_dim + metat_dim, latent_dim)
        
        self.metag_dim = metag_dim
        self.metat_dim = metat_dim
        
        # Separate encoders for each modality
        self.metag_encoder = self._build_encoder(metag_dim, hidden_dims, dropout_rate)
        self.metat_encoder = self._build_encoder(metat_dim, hidden_dims, dropout_rate)
        
        # Fusion layer
        combined_dim = hidden_dims[-1] * 2
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Latent projections
        self.fc_mu = nn.Linear(combined_dim // 2, latent_dim)
        self.fc_var = nn.Linear(combined_dim // 2, latent_dim)
        
        # Separate decoders for each modality
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, combined_dim // 2),
            nn.BatchNorm1d(combined_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        self.metag_decoder = self._build_decoder(combined_dim // 2, hidden_dims, metag_dim, dropout_rate)
        self.metat_decoder = self._build_decoder(combined_dim // 2, hidden_dims, metat_dim, dropout_rate)
        
        self._initialize_weights()
    
    def _build_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float
    ) -> nn.Module:
        """Build encoder for one modality."""
        layers = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float
    ) -> nn.Module:
        """Build decoder for one modality."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode multi-omics input."""
        # Split input
        x_metag = x[:, :self.metag_dim]
        x_metat = x[:, self.metag_dim:]
        
        # Encode each modality
        h_metag = self.metag_encoder(x_metag)
        h_metat = self.metat_encoder(x_metat)
        
        # Fuse representations
        h_combined = torch.cat([h_metag, h_metat], dim=1)
        h_fused = self.fusion(h_combined)
        
        return self.fc_mu(h_fused), self.fc_var(h_fused)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to multi-omics reconstruction."""
        h = self.shared_decoder(z)
        
        # Decode each modality
        recon_metag = self.metag_decoder(h)
        recon_metat = self.metat_decoder(h)
        
        return torch.cat([recon_metag, recon_metat], dim=1)
