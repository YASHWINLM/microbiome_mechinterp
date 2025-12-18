"""
Phylogenetic distance-aware VAE for microbiome data.

PhyloVAE incorporates evolutionary relationships between taxa
to improve latent representations by encouraging similar
representations for evolutionarily related organisms.

Design Principles:
- Leverage phylogenetic tree information
- Soft constraints (not hard structural assumptions)
- Compatible with standard VAE training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import warnings

from q2_mechinterp.core.vae import BaseVAE


class PhyloVAE(BaseVAE):
    """
    Phylogenetic distance-aware VAE for microbiome data.
    
    PhyloVAE uses phylogenetic distances between taxa to:
    1. Weight features based on evolutionary relationships
    2. Promote similar latent representations for related taxa
    3. Incorporate prior biological knowledge
    
    The model has two encoding branches:
    - Standard encoder: Processes raw features
    - Phylogenetic encoder: Processes phylogenetically-weighted features
    
    These are combined with a learnable weight to balance the two views.
    
    Example:
        >>> # Load phylogenetic distances
        >>> phylo_distances = load_phylo_distances('tree.nwk', feature_names)
        >>> 
        >>> # Create model
        >>> vae = PhyloVAE(
        ...     input_dim=500,
        ...     phylo_distances=phylo_distances,
        ...     latent_dim=50,
        ...     phylo_weight=0.3
        ... )
        >>> 
        >>> # Training proceeds as normal
        >>> recon, mu, logvar = vae(x)
    
    Args:
        input_dim: Number of input features (taxa)
        phylo_distances: Pairwise phylogenetic distance matrix (n_features x n_features)
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        phylo_weight: Initial weight for phylogenetic branch (0-1)
        phylo_sigma: Sigma for Gaussian kernel on distances
        dropout_rate: Dropout rate for regularization
        learn_phylo_weight: Whether to learn phylo_weight during training
    """
    
    def __init__(
        self,
        input_dim: int,
        phylo_distances: Optional[np.ndarray] = None,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 50,
        phylo_weight: float = 0.3,
        phylo_sigma: float = 1.0,
        dropout_rate: float = 0.2,
        learn_phylo_weight: bool = True
    ):
        super().__init__(input_dim, latent_dim)
        
        self.hidden_dims = hidden_dims
        self.phylo_sigma = phylo_sigma
        self.dropout_rate = dropout_rate
        
        # Phylogenetic kernel (convert distances to similarities)
        if phylo_distances is not None:
            phylo_distances = np.asarray(phylo_distances, dtype=np.float32)
            if phylo_distances.shape != (input_dim, input_dim):
                raise ValueError(
                    f"phylo_distances shape {phylo_distances.shape} doesn't match "
                    f"input_dim {input_dim}"
                )
            phylo_kernel = self._compute_phylo_kernel(phylo_distances, phylo_sigma)
            self.register_buffer('phylo_kernel', torch.from_numpy(phylo_kernel))
        else:
            # Identity kernel if no phylogenetic info
            self.register_buffer('phylo_kernel', torch.eye(input_dim))
        
        # Learnable phylogenetic weight
        if learn_phylo_weight:
            self._phylo_weight = nn.Parameter(
                torch.tensor(phylo_weight).float()
            )
        else:
            self.register_buffer('_phylo_weight', torch.tensor(phylo_weight).float())
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Standard encoder branch
        self.standard_encoder = self._build_encoder(input_dim, hidden_dims)
        
        # Phylogenetic encoder branch (processes phylogenetically-weighted features)
        self.phylo_encoder = self._build_encoder(input_dim, hidden_dims)
        
        # Fusion layer (combines both branches)
        fusion_dim = hidden_dims[-1] * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        self.decoder = self._build_decoder(latent_dim, hidden_dims, input_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    @staticmethod
    def _compute_phylo_kernel(
        distances: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """
        Convert phylogenetic distances to similarity kernel.
        
        Uses Gaussian (RBF) kernel: K(i,j) = exp(-d(i,j)^2 / (2*sigma^2))
        """
        kernel = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalize rows to sum to 1
        row_sums = kernel.sum(axis=1, keepdims=True)
        kernel = kernel / (row_sums + 1e-8)
        
        return kernel.astype(np.float32)
    
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
    
    def _build_decoder(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int
    ) -> nn.Module:
        """Build decoder network."""
        layers = []
        prev_dim = latent_dim
        
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
    
    @property
    def phylo_weight(self) -> float:
        """Current phylogenetic weight (clamped to [0, 1])."""
        return torch.clamp(self._phylo_weight, 0.0, 1.0).item()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input using both standard and phylogenetic branches.
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        x = self.input_norm(x)
        
        # Standard encoding
        h_standard = self.standard_encoder(x)
        
        # Phylogenetic encoding (weight features by phylogenetic similarity)
        x_phylo = torch.mm(x, self.phylo_kernel)
        h_phylo = self.phylo_encoder(x_phylo)
        
        # Combine with learned weight
        w = torch.clamp(self._phylo_weight, 0.0, 1.0)
        h_combined = torch.cat([
            (1 - w) * h_standard,
            w * h_phylo
        ], dim=1)
        
        # Fusion
        h_fused = self.fusion(h_combined)
        
        return self.fc_mu(h_fused), self.fc_var(h_fused)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def get_phylo_weighted_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get phylogenetically-weighted features.
        
        Useful for analysis and interpretation.
        
        Args:
            x: Input tensor
        
        Returns:
            Phylogenetically-weighted features
        """
        with torch.no_grad():
            return torch.mm(x, self.phylo_kernel)
    
    def get_branch_contributions(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get separate contributions from standard and phylogenetic branches.
        
        Useful for understanding which branch drives representations.
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary with branch outputs
        """
        self.eval()
        with torch.no_grad():
            x = self.input_norm(x)
            
            h_standard = self.standard_encoder(x)
            x_phylo = torch.mm(x, self.phylo_kernel)
            h_phylo = self.phylo_encoder(x_phylo)
            
            return {
                'standard': h_standard,
                'phylogenetic': h_phylo,
                'phylo_weight': self.phylo_weight
            }


def load_phylo_distances(
    tree_path: Union[str, Path],
    feature_names: List[str],
    default_distance: float = 1.0
) -> np.ndarray:
    """
    Load phylogenetic distances from a Newick tree file.
    
    Requires BioPython for tree parsing.
    
    Args:
        tree_path: Path to Newick format tree file
        feature_names: List of feature (taxon) names in data order
        default_distance: Distance for taxa not in tree
    
    Returns:
        Distance matrix (n_features x n_features)
    """
    try:
        from Bio import Phylo
    except ImportError:
        raise ImportError(
            "BioPython required for tree loading. "
            "Install with: pip install biopython"
        )
    
    tree_path = Path(tree_path)
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    
    # Load tree
    tree = Phylo.read(str(tree_path), 'newick')
    
    # Get terminal names
    terminals = {term.name: term for term in tree.get_terminals()}
    
    n_features = len(feature_names)
    distances = np.full((n_features, n_features), default_distance, dtype=np.float32)
    np.fill_diagonal(distances, 0.0)
    
    # Compute pairwise distances
    matched = 0
    for i, name_i in enumerate(feature_names):
        if name_i not in terminals:
            continue
        
        for j, name_j in enumerate(feature_names):
            if i >= j:
                continue
            if name_j not in terminals:
                continue
            
            try:
                dist = tree.distance(terminals[name_i], terminals[name_j])
                distances[i, j] = dist
                distances[j, i] = dist
                matched += 1
            except Exception:
                pass
    
    print(f"Matched {matched} pairwise distances from tree")
    
    return distances


def phylo_regularized_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    phylo_kernel: torch.Tensor,
    beta: float = 0.01,
    alpha: float = 0.001,
    phylo_lambda: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss function with phylogenetic regularization.
    
    Adds a term that encourages similar latent representations
    for phylogenetically related taxa.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Latent mean
        logvar: Latent log variance
        phylo_kernel: Phylogenetic similarity kernel
        beta: KL divergence weight
        alpha: L1 regularization weight
        phylo_lambda: Phylogenetic regularization weight
    
    Returns:
        Tuple of (total_loss, recon_loss, kl_loss, l1_loss, phylo_loss)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    logvar_clamped = torch.clamp(logvar, -10, 10)
    kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp())
    
    # L1 regularization
    l1_loss = alpha * torch.mean(torch.abs(mu))
    
    # Phylogenetic regularization
    # Encourage reconstruction to respect phylogenetic relationships
    x_phylo = torch.mm(x, phylo_kernel)
    recon_phylo = torch.mm(recon_x, phylo_kernel)
    phylo_loss = phylo_lambda * F.mse_loss(recon_phylo, x_phylo, reduction='mean')
    
    total_loss = recon_loss + beta * kl_loss + l1_loss + phylo_loss
    
    return total_loss, recon_loss, kl_loss, l1_loss, phylo_loss


class PhyloVAETrainer:
    """
    Specialized trainer for PhyloVAE with phylogenetic loss.
    
    Extends standard training with phylogenetic regularization.
    """
    
    def __init__(
        self,
        model: PhyloVAE,
        device: torch.device,
        lr: float = 1e-3,
        phylo_lambda: float = 0.01
    ):
        self.model = model.to(device)
        self.device = device
        self.phylo_lambda = phylo_lambda
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.history = {'train_loss': [], 'val_loss': [], 'phylo_loss': []}
    
    def train_epoch(
        self,
        train_loader,
        beta: float = 0.01,
        alpha: float = 0.001
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch with phylogenetic regularization."""
        self.model.train()
        
        total_loss = 0
        total_phylo = 0
        n_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            recon, mu, logvar = self.model(batch)
            
            loss, recon_l, kl_l, l1_l, phylo_l = phylo_regularized_loss(
                recon, batch, mu, logvar,
                self.model.phylo_kernel,
                beta, alpha, self.phylo_lambda
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_phylo += phylo_l.item()
            n_batches += 1
        
        return total_loss / n_batches, {'phylo_loss': total_phylo / n_batches}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'PhyloVAE',
    'load_phylo_distances',
    'phylo_regularized_loss',
    'PhyloVAETrainer',
]
