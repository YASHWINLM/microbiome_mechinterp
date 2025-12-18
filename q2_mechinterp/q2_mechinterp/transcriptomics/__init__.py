"""
Transcriptomics-specific components for VAE feature engineering.

Supports RNA-seq and other gene expression data processing,
with specialized VAE architectures for transcriptomic analysis.

Includes:
- TranscriptomicsVAE for standard gene expression
- ConditionalTranscriptomicsVAE for condition-aware modeling
- PathwayVAE for pathway-informed encoding
"""

from q2_mechinterp.transcriptomics.vae import (
    TranscriptomicsVAE,
    ConditionalTranscriptomicsVAE,
    EnhancedGeneVAE,
    transcriptomics_vae_loss
)
from q2_mechinterp.transcriptomics.processor import TranscriptomicsDataProcessor
from q2_mechinterp.transcriptomics.pathway_vae import (
    PathwayVAE,
    load_kegg_pathways,
    load_pathway_gmt,
    pathway_activity_loss,
)

__all__ = [
    # VAE
    "TranscriptomicsVAE",
    "ConditionalTranscriptomicsVAE",
    "EnhancedGeneVAE",  # Alias for backward compatibility
    "transcriptomics_vae_loss",
    # PathwayVAE
    "PathwayVAE",
    "load_kegg_pathways",
    "load_pathway_gmt",
    "pathway_activity_loss",
    # Processor
    "TranscriptomicsDataProcessor",
]
