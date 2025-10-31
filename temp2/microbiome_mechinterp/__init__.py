"""
Microbiome Mechanistic Interpretability
Advanced microbiome analysis using VAE and Sparse Autoencoders
"""

__version__ = "0.1.0"

from .data_processing.processor import MicrobiomeDataProcessor
from .models.vae import VAE, VAELoss
from .models.sparse_ae import SparseAutoencoder, sparse_loss
from .training.trainer import MicrobiomeVAETrainer
from .feature_engineering.extractor import FeatureExtractor
from .feature_engineering.mapping import map_sparse_features_to_otus, map_sparse_features_with_taxonomy
from .classification.classifier import Q2SampleClassifier, FeatureClassifier
from .visualization.visualizer import Visualizer
from .config.settings import Config
from .utils.taxonomy import (
    find_ogu_by_species, get_taxonomy_info, extract_species_name,
    format_feature_name, get_taxonomic_level
)

__all__ = [
    "MicrobiomeDataProcessor", "VAE", "VAELoss", "SparseAutoencoder",
    "MicrobiomeVAETrainer", "FeatureExtractor", "Q2SampleClassifier",
    "FeatureClassifier", "Visualizer", "Config", "sparse_loss",
    "map_sparse_features_to_otus", "map_sparse_features_with_taxonomy",
    "find_ogu_by_species", "get_taxonomy_info", "extract_species_name",
    "format_feature_name", "get_taxonomic_level"
]
