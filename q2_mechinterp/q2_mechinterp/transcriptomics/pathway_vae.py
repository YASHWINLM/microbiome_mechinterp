"""
Pathway-informed VAE for transcriptomics data.

PathwayVAE incorporates gene-pathway membership information
to structure the encoder, aggregating gene-level information
through biological pathway nodes.

Design Principles:
- Leverage pathway databases (KEGG, GO, Reactome)
- Structured encoding via pathway aggregation
- Interpretable pathway activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import warnings

from q2_mechinterp.core.vae import BaseVAE


class PathwayVAE(BaseVAE):
    """
    Pathway-informed VAE for gene expression data.
    
    PathwayVAE uses gene-pathway membership to structure the encoder:
    1. Gene-level encoding: Standard encoding of individual genes
    2. Pathway aggregation: Aggregate gene info through pathway nodes
    3. Attention fusion: Combine gene and pathway views with attention
    
    This provides:
    - Biologically meaningful intermediate representations
    - Interpretable pathway activity scores
    - Better generalization via pathway-level regularization
    
    Example:
        >>> # Define pathway membership
        >>> pathway_dict = {
        ...     'KEGG_GLYCOLYSIS': ['HK1', 'GPI', 'PFKL', ...],
        ...     'KEGG_TCA_CYCLE': ['CS', 'ACO1', 'IDH1', ...],
        ...     ...
        ... }
        >>> 
        >>> vae = PathwayVAE(
        ...     input_dim=5000,
        ...     pathway_dict=pathway_dict,
        ...     gene_names=gene_names,
        ...     latent_dim=50
        ... )
        >>> 
        >>> # Get pathway activations
        >>> activations = vae.get_pathway_activations(x)
    
    Args:
        input_dim: Number of input genes
        pathway_dict: Dict mapping pathway names to gene lists
        gene_names: List of gene names in input order
        hidden_dims: Hidden layer dimensions
        latent_dim: Latent space dimension
        pathway_hidden_dim: Hidden dimension for pathway nodes
        dropout_rate: Dropout rate
        use_attention: Use attention to combine gene and pathway views
        min_pathway_genes: Minimum genes per pathway to include
    """
    
    def __init__(
        self,
        input_dim: int,
        pathway_dict: Dict[str, List[str]],
        gene_names: List[str],
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 50,
        pathway_hidden_dim: int = 64,
        dropout_rate: float = 0.2,
        use_attention: bool = True,
        min_pathway_genes: int = 5
    ):
        super().__init__(input_dim, latent_dim)
        
        self.hidden_dims = hidden_dims
        self.pathway_hidden_dim = pathway_hidden_dim
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Build pathway membership matrix
        self.pathway_names, membership = self._build_pathway_membership(
            pathway_dict, gene_names, min_pathway_genes
        )
        self.n_pathways = len(self.pathway_names)
        
        # Register membership as buffer (not trainable but saved with model)
        self.register_buffer('pathway_membership', torch.from_numpy(membership))
        
        print(f"PathwayVAE: {self.n_pathways} pathways, "
              f"{input_dim} genes, "
              f"avg {membership.sum(axis=0).mean():.1f} genes per pathway")
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Gene-level encoder
        self.gene_encoder = self._build_encoder(input_dim, hidden_dims)
        
        # Pathway encoder (processes pathway-aggregated features)
        self.pathway_transform = nn.Sequential(
            nn.Linear(self.n_pathways, pathway_hidden_dim),
            nn.BatchNorm1d(pathway_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Pathway-level encoder
        pathway_encoder_dims = [d // 2 for d in hidden_dims]  # Smaller for pathway branch
        self.pathway_encoder = self._build_encoder(
            pathway_hidden_dim, pathway_encoder_dims
        )
        
        # Attention mechanism (if enabled)
        if use_attention:
            combined_dim = hidden_dims[-1] + pathway_encoder_dims[-1]
            self.attention = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.Tanh(),
                nn.Linear(combined_dim // 2, 2),  # 2 attention weights
                nn.Softmax(dim=1)
            )
            fusion_input_dim = combined_dim
        else:
            fusion_input_dim = hidden_dims[-1] + pathway_encoder_dims[-1]
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (standard gene-level)
        self.decoder = self._build_decoder(latent_dim, hidden_dims, input_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_pathway_membership(
        self,
        pathway_dict: Dict[str, List[str]],
        gene_names: List[str],
        min_genes: int
    ) -> Tuple[List[str], np.ndarray]:
        """
        Build gene-pathway membership matrix.
        
        Returns:
            Tuple of (pathway_names, membership_matrix)
            where membership_matrix is (n_genes, n_pathways)
        """
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        
        valid_pathways = []
        memberships = []
        
        for pathway_name, genes in pathway_dict.items():
            # Find genes that are in our dataset
            pathway_indices = []
            for gene in genes:
                if gene in gene_to_idx:
                    pathway_indices.append(gene_to_idx[gene])
            
            # Only include pathways with enough genes
            if len(pathway_indices) >= min_genes:
                valid_pathways.append(pathway_name)
                
                # Create membership vector
                membership = np.zeros(len(gene_names), dtype=np.float32)
                membership[pathway_indices] = 1.0
                
                # Normalize by pathway size
                membership = membership / membership.sum()
                memberships.append(membership)
        
        if not valid_pathways:
            raise ValueError(
                f"No pathways have >= {min_genes} genes in the dataset"
            )
        
        membership_matrix = np.stack(memberships, axis=1)
        
        return valid_pathways, membership_matrix
    
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input using gene and pathway branches.
        
        Args:
            x: Input tensor (batch_size, n_genes)
        
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        x_norm = self.input_norm(x)
        
        # Gene-level encoding
        h_gene = self.gene_encoder(x_norm)
        
        # Pathway-level encoding
        # Aggregate genes by pathway membership
        pathway_values = torch.mm(x_norm, self.pathway_membership)  # (batch, n_pathways)
        pathway_transformed = self.pathway_transform(pathway_values)
        h_pathway = self.pathway_encoder(pathway_transformed)
        
        # Combine gene and pathway representations
        h_combined = torch.cat([h_gene, h_pathway], dim=1)
        
        if self.use_attention:
            # Compute attention weights
            attn_weights = self.attention(h_combined)  # (batch, 2)
            
            # Apply attention
            h_gene_weighted = h_gene * attn_weights[:, 0:1]
            h_pathway_weighted = h_pathway * attn_weights[:, 1:2]
            
            # Pad to same size
            if h_gene.size(1) != h_pathway.size(1):
                max_dim = max(h_gene.size(1), h_pathway.size(1))
                if h_gene.size(1) < max_dim:
                    h_gene_weighted = F.pad(h_gene_weighted, (0, max_dim - h_gene.size(1)))
                if h_pathway.size(1) < max_dim:
                    h_pathway_weighted = F.pad(h_pathway_weighted, (0, max_dim - h_pathway.size(1)))
            
            h_combined = torch.cat([h_gene_weighted, h_pathway_weighted], dim=1)
        
        # Fusion
        h_fused = self.fusion(h_combined)
        
        return self.fc_mu(h_fused), self.fc_var(h_fused)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to gene expression."""
        return self.decoder(z)
    
    def get_pathway_activations(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get pathway activity scores for input samples.
        
        Args:
            x: Input tensor (batch_size, n_genes)
            return_attention: Also return attention weights
        
        Returns:
            Pathway activations (batch_size, n_pathways)
            Optionally also attention weights if return_attention=True
        """
        self.eval()
        with torch.no_grad():
            x_norm = self.input_norm(x)
            
            # Aggregate genes by pathway
            pathway_values = torch.mm(x_norm, self.pathway_membership)
            pathway_transformed = self.pathway_transform(pathway_values)
            
            if return_attention and self.use_attention:
                h_gene = self.gene_encoder(x_norm)
                h_pathway = self.pathway_encoder(pathway_transformed)
                h_combined = torch.cat([h_gene, h_pathway], dim=1)
                attn_weights = self.attention(h_combined)
                return pathway_transformed, attn_weights
            
            return pathway_transformed
    
    def get_pathway_names(self) -> List[str]:
        """Get list of pathway names."""
        return list(self.pathway_names)
    
    def get_pathway_genes(self, pathway_name: str) -> List[int]:
        """
        Get gene indices for a specific pathway.
        
        Args:
            pathway_name: Name of pathway
        
        Returns:
            List of gene indices in the pathway
        """
        if pathway_name not in self.pathway_names:
            raise ValueError(f"Unknown pathway: {pathway_name}")
        
        idx = self.pathway_names.index(pathway_name)
        membership = self.pathway_membership[:, idx].cpu().numpy()
        return list(np.where(membership > 0)[0])


def load_kegg_pathways(
    species: str = 'hsa'
) -> Dict[str, List[str]]:
    """
    Load KEGG pathways for a species.
    
    Note: Requires internet connection and KEGG API access.
    For offline use, save results and load from file.
    
    Args:
        species: KEGG species code (e.g., 'hsa' for human)
    
    Returns:
        Dictionary mapping pathway names to gene lists
    """
    try:
        from Bio.KEGG import REST
    except ImportError:
        raise ImportError(
            "BioPython required for KEGG loading. "
            "Install with: pip install biopython"
        )
    
    pathways = {}
    
    try:
        # Get list of pathways
        pathway_list = REST.kegg_list('pathway', species).read()
        
        for line in pathway_list.strip().split('\n'):
            parts = line.split('\t')
            pathway_id = parts[0].replace('path:', '')
            pathway_name = parts[1].split(' - ')[0] if ' - ' in parts[1] else parts[1]
            
            # Get genes in pathway
            genes = REST.kegg_get(pathway_id).read()
            
            gene_list = []
            in_gene_section = False
            for gene_line in genes.split('\n'):
                if gene_line.startswith('GENE'):
                    in_gene_section = True
                    gene_list.append(gene_line.split()[1].rstrip(';'))
                elif in_gene_section and gene_line.startswith('            '):
                    parts = gene_line.strip().split()
                    if parts:
                        gene_list.append(parts[0].rstrip(';'))
                elif in_gene_section and not gene_line.startswith(' '):
                    break
            
            if gene_list:
                pathways[pathway_name] = gene_list
                
    except Exception as e:
        warnings.warn(f"Error loading KEGG pathways: {e}")
    
    return pathways


def load_pathway_gmt(
    gmt_path: Union[str, Path]
) -> Dict[str, List[str]]:
    """
    Load pathways from GMT (Gene Matrix Transposed) file.
    
    GMT is a common format used by MSigDB and other databases.
    
    Args:
        gmt_path: Path to GMT file
    
    Returns:
        Dictionary mapping pathway names to gene lists
    """
    gmt_path = Path(gmt_path)
    if not gmt_path.exists():
        raise FileNotFoundError(f"GMT file not found: {gmt_path}")
    
    pathways = {}
    
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pathway_name = parts[0]
                # parts[1] is usually a description URL, skip it
                genes = parts[2:]
                pathways[pathway_name] = genes
    
    return pathways


def pathway_activity_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pathway_membership: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 0.1,
    pathway_lambda: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss function with pathway activity preservation.
    
    Adds a term that encourages reconstruction to preserve
    pathway-level activity patterns.
    
    Args:
        recon_x: Reconstructed gene expression
        x: Original gene expression
        mu: Latent mean
        logvar: Latent log variance
        pathway_membership: Gene-pathway membership matrix
        beta: KL divergence weight
        alpha: L1 regularization weight
        pathway_lambda: Pathway preservation weight
    
    Returns:
        Tuple of (total, recon, kl, l1, pathway) losses
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # L1 regularization
    l1_loss = alpha * torch.mean(torch.abs(mu))
    
    # Pathway activity preservation
    x_pathway = torch.mm(x, pathway_membership)
    recon_pathway = torch.mm(recon_x, pathway_membership)
    pathway_loss = pathway_lambda * F.mse_loss(recon_pathway, x_pathway, reduction='mean')
    
    total_loss = recon_loss + beta * kl_loss + l1_loss + pathway_loss
    
    return total_loss, recon_loss, kl_loss, l1_loss, pathway_loss


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'PathwayVAE',
    'load_kegg_pathways',
    'load_pathway_gmt',
    'pathway_activity_loss',
]
