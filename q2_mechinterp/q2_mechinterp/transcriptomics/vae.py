"""
Transcriptomics-specific Variational Autoencoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from q2_mechinterp.core.vae import BaseVAE


class TranscriptomicsVAE(BaseVAE):
    """
    Enhanced VAE for gene expression (RNA-seq) data.
    
    Optimized for high-dimensional gene expression data with
    residual connections and layer normalization.
    
    Args:
        input_dim: Number of input genes.
        hidden_dims: List of hidden layer dimensions.
        latent_dim: Dimension of latent space.
        dropout_rate: Dropout rate for regularization.
        use_residual: Whether to use residual connections (if dims match).
    
    Example:
        >>> vae = TranscriptomicsVAE(input_dim=400, latent_dim=50)
        >>> x = torch.randn(32, 400)
        >>> recon, mu, logvar = vae(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [1024, 512, 256],
        latent_dim: int = 50,
        dropout_rate: float = 0.2,
        use_residual: bool = True
    ):
        super().__init__(input_dim, latent_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual and (input_dim == hidden_dims[0])
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_bn_layers = nn.ModuleList()
        
        self.encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        for dim in hidden_dims:
            self.encoder_bn_layers.append(nn.BatchNorm1d(dim))
        
        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        self.decoder_bn_layers = nn.ModuleList()
        
        self.decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
        self.decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        for dim in reversed(hidden_dims):
            self.decoder_bn_layers.append(nn.BatchNorm1d(dim))
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.input_layer_norm = nn.LayerNorm(input_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        x = self.input_layer_norm(x)
        h = x
        
        for i, (layer, bn) in enumerate(zip(self.encoder_layers, self.encoder_bn_layers)):
            h_new = F.leaky_relu(bn(layer(h)))
            h_new = self.dropout(h_new)
            
            # Residual connection for first layer if dimensions match
            if self.use_residual and i == 0:
                h_new = h_new + h
            
            h = h_new
        
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to gene expression reconstruction."""
        h = z
        
        for i, (layer, bn) in enumerate(zip(self.decoder_layers[:-1], self.decoder_bn_layers)):
            h = F.leaky_relu(bn(layer(h)))
            h = self.dropout(h)
        
        return self.decoder_layers[-1](h)


def transcriptomics_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss function for transcriptomics VAE.
    
    Args:
        recon_x: Reconstructed gene expression.
        x: Original gene expression.
        mu: Latent mean.
        logvar: Latent log variance.
        beta: Weight for KL divergence.
        alpha: Weight for L1 regularization on latent space.
    
    Returns:
        Tuple of (total_loss, mse_loss, kl_loss, l1_loss).
    """
    # MSE reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence normalized by latent dimension
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # L1 regularization for sparsity
    l1_loss = alpha * torch.mean(torch.abs(mu))
    
    total_loss = mse_loss + beta * kl_loss + l1_loss
    
    return total_loss, mse_loss, kl_loss, l1_loss


class ConditionalTranscriptomicsVAE(BaseVAE):
    """
    Conditional VAE for transcriptomics data.
    
    Conditions on sample labels/phenotypes for better separation
    in latent space.
    
    Args:
        input_dim: Number of input genes.
        num_classes: Number of condition classes.
        hidden_dims: List of hidden layer dimensions.
        latent_dim: Dimension of latent space.
        dropout_rate: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 50,
        dropout_rate: float = 0.2
    ):
        super().__init__(input_dim, latent_dim)
        
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # Encoder (input + one-hot condition)
        encoder_input_dim = input_dim + num_classes
        
        encoder_layers = []
        prev_dim = encoder_input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (latent + condition)
        decoder_input_dim = latent_dim + num_classes
        
        decoder_layers = []
        prev_dim = decoder_input_dim
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
        self.input_norm = nn.LayerNorm(input_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert labels to one-hot encoding."""
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return one_hot
    
    def encode(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input with condition.
        
        Args:
            x: Input gene expression.
            c: Condition labels (integers) or one-hot vectors.
        """
        x = self.input_norm(x)
        
        if c is not None:
            if c.dim() == 1:
                c = self._one_hot(c)
            x = torch.cat([x, c], dim=1)
        else:
            # If no condition, pad with zeros
            padding = torch.zeros(x.size(0), self.num_classes, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def decode(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent with condition.
        
        Args:
            z: Latent representation.
            c: Condition labels (integers) or one-hot vectors.
        """
        if c is not None:
            if c.dim() == 1:
                c = self._one_hot(c)
            z = torch.cat([z, c], dim=1)
        else:
            padding = torch.zeros(z.size(0), self.num_classes, device=z.device)
            z = torch.cat([z, padding], dim=1)
        
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional condition."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


# Alias for backward compatibility
EnhancedGeneVAE = TranscriptomicsVAE
