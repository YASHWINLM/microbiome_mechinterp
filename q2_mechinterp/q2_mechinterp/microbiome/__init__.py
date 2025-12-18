"""
Microbiome-specific components for VAE feature engineering.

Supports metagenomics and metatranscriptomics data processing,
specialized VAE architectures, and taxonomy-aware analysis.

Includes:
- MicrobiomeVAE for standard microbiome data
- MultiOmicsVAE for paired metagenome/metatranscriptome
- PhyloVAE for phylogenetically-informed analysis
"""

from q2_mechinterp.microbiome.vae import (
    MicrobiomeVAE,
    MultiOmicsVAE,
    microbiome_vae_loss
)
from q2_mechinterp.microbiome.processor import MicrobiomeDataProcessor
from q2_mechinterp.microbiome.taxonomy import (
    find_ogu_by_species,
    get_taxonomy_info,
    extract_species_name,
    format_feature_name,
    get_taxonomic_level,
    parse_taxonomy,
    aggregate_by_taxonomy,
    filter_by_taxonomy,
    TAXONOMY_LEVELS
)
from q2_mechinterp.microbiome.phylo_vae import (
    PhyloVAE,
    load_phylo_distances,
    phylo_regularized_loss,
    PhyloVAETrainer,
)

__all__ = [
    # VAE
    "MicrobiomeVAE",
    "MultiOmicsVAE",
    "microbiome_vae_loss",
    # PhyloVAE
    "PhyloVAE",
    "load_phylo_distances",
    "phylo_regularized_loss",
    "PhyloVAETrainer",
    # Processor
    "MicrobiomeDataProcessor",
    # Taxonomy
    "find_ogu_by_species",
    "get_taxonomy_info",
    "extract_species_name",
    "format_feature_name",
    "get_taxonomic_level",
    "parse_taxonomy",
    "aggregate_by_taxonomy",
    "filter_by_taxonomy",
    "TAXONOMY_LEVELS",
]
