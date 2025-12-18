"""
VAE Features - Feature Engineering using Variational Autoencoders

A comprehensive Python package for VAE-based feature extraction and engineering,
supporting microbiome (metagenomics/metatranscriptomics) and transcriptomics data.

Modules:
    core: Base classes, protocols, configs, validation, logging
    microbiome: MicrobiomeVAE, PhyloVAE, data processing
    transcriptomics: TranscriptomicsVAE, PathwayVAE, data processing
    training: VAETrainer, cross-validation, semi-supervised
    extraction: FeatureExtractor, sparse AE training
    evaluation: Feature quality metrics, model comparison
    preprocessing: Batch correction (ComBat, ConQuR)
    augmentation: Domain-specific data augmentation
    downstream: Feature selection, sklearn integration
    tuning: Hyperparameter optimization (Optuna)
    visualization: Plotting utilities

Example Usage:
    # Microbiome workflow
    from q2_mechinterp.microbiome import MicrobiomeVAE, MicrobiomeDataProcessor
    from q2_mechinterp.training import VAETrainer
    from q2_mechinterp.extraction import FeatureExtractor
    from q2_mechinterp.evaluation import FeatureEvaluator
    
    # Transcriptomics workflow  
    from q2_mechinterp.transcriptomics import TranscriptomicsVAE, TranscriptomicsDataProcessor
    
    # Configuration
    from q2_mechinterp.core import PipelineConfig, VAEConfig, TrainingConfig
    
    # Batch correction
    from q2_mechinterp.preprocessing import ComBatCorrector, ConQuRCorrector
    
    # Hyperparameter tuning
    from q2_mechinterp.tuning import VAETuner
    
    # Data augmentation
    from q2_mechinterp.augmentation import MicrobiomeAugmentor, TranscriptomicsAugmentor
    
    # Downstream integration
    from q2_mechinterp.downstream import FeatureSelector, SparseFeatureClassifier

CLI Usage:
    q2-mechinterp train --config config.yaml --data data.csv
    q2-mechinterp extract --checkpoint model.pt --data data.csv --output features.csv
    q2-mechinterp tune --config config.yaml --data data.csv --trials 50
    q2-mechinterp evaluate --features features.csv --labels labels.csv
"""

__version__ = "0.3.0"
__author__ = "Leo Joseph"
__email__ = ""

# Core imports
from q2_mechinterp.core.vae import BaseVAE, ConcreteVAE
from q2_mechinterp.core.sparse import SparseAutoencoder, TopKSparseAutoencoder
from q2_mechinterp.core.utils import get_device, set_seed, EarlyStopping
from q2_mechinterp.core.protocols import (
    PipelineConfig, VAEConfig, TrainingConfig, 
    SparseAEConfig, DataConfig, EvaluationConfig,
    create_default_config
)
from q2_mechinterp.core.logging import StructuredLogger, timer, print_model_summary, get_progress_bar
from q2_mechinterp.core.data import BaseDataProcessor, DataSplit
from q2_mechinterp.core.validation import ValidationError, validate_array

# Domain-specific imports
from q2_mechinterp.microbiome import MicrobiomeVAE, MicrobiomeDataProcessor, PhyloVAE
from q2_mechinterp.transcriptomics import (
    TranscriptomicsVAE, TranscriptomicsDataProcessor, PathwayVAE
)

# Training
from q2_mechinterp.training import VAETrainer, CrossValidator, SemiSupervisedVAE

# Extraction
from q2_mechinterp.extraction import FeatureExtractor

# Evaluation
from q2_mechinterp.evaluation import FeatureEvaluator, calculate_feature_quality_metrics

# Preprocessing (lazy import to avoid dependency issues)
def __getattr__(name):
    """Lazy imports for optional modules."""
    if name in ('ComBatCorrector', 'ConQuRCorrector', 'SimpleBatchCorrector'):
        from q2_mechinterp.preprocessing import ComBatCorrector, ConQuRCorrector, SimpleBatchCorrector
        return locals()[name]
    elif name == 'VAETuner':
        from q2_mechinterp.tuning import VAETuner
        return VAETuner
    elif name in ('MicrobiomeAugmentor', 'TranscriptomicsAugmentor'):
        from q2_mechinterp.augmentation import MicrobiomeAugmentor, TranscriptomicsAugmentor
        return locals()[name]
    elif name in ('FeatureSelector', 'SparseFeatureClassifier'):
        from q2_mechinterp.downstream import FeatureSelector, SparseFeatureClassifier
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseVAE",
    "ConcreteVAE",
    "SparseAutoencoder",
    "TopKSparseAutoencoder",
    "get_device",
    "set_seed",
    "EarlyStopping",
    # Data
    "BaseDataProcessor",
    "DataSplit",
    # Validation
    "ValidationError",
    "validate_array",
    # Configs
    "PipelineConfig",
    "VAEConfig",
    "TrainingConfig",
    "SparseAEConfig",
    "DataConfig",
    "EvaluationConfig",
    "create_default_config",
    # Logging
    "StructuredLogger",
    "timer",
    "print_model_summary",
    "get_progress_bar",
    # Microbiome
    "MicrobiomeVAE",
    "MicrobiomeDataProcessor",
    "PhyloVAE",
    # Transcriptomics
    "TranscriptomicsVAE",
    "TranscriptomicsDataProcessor",
    "PathwayVAE",
    # Training
    "VAETrainer",
    "CrossValidator",
    "SemiSupervisedVAE",
    # Extraction
    "FeatureExtractor",
    # Evaluation
    "FeatureEvaluator",
    "calculate_feature_quality_metrics",
    # Preprocessing (lazy loaded)
    "ComBatCorrector",
    "ConQuRCorrector",
    "SimpleBatchCorrector",
    # Tuning (lazy loaded)
    "VAETuner",
    # Augmentation (lazy loaded)
    "MicrobiomeAugmentor",
    "TranscriptomicsAugmentor",
    # Downstream (lazy loaded)
    "FeatureSelector",
    "SparseFeatureClassifier",
]
