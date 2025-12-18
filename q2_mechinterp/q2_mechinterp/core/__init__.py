"""
Core components for VAE feature engineering.

This module provides:
- Base VAE and concrete implementations
- Sparse autoencoders for interpretability
- Protocol classes for type checking
- Configuration dataclasses
- Logging and timing utilities
- Validation functions
- Base data processor
"""

from q2_mechinterp.core.vae import BaseVAE, ConcreteVAE, vae_loss
from q2_mechinterp.core.sparse import (
    SparseAutoencoder,
    TopKSparseAutoencoder,
    DeepSparseAutoencoder,
    sparse_autoencoder_loss
)
from q2_mechinterp.core.utils import (
    get_device,
    set_seed,
    add_noise,
    check_tensor_validity,
    clean_array,
    EarlyStopping
)
from q2_mechinterp.core.protocols import (
    VAEProtocol,
    DataProcessorProtocol,
    SparseAEProtocol,
    TrainerProtocol,
    VAEConfig,
    TrainingConfig,
    SparseAEConfig,
    DataConfig,
    EvaluationConfig,
    PipelineConfig,
    create_default_config,
)
from q2_mechinterp.core.logging import (
    get_logger,
    setup_logging,
    StructuredLogger,
    timer,
    Timer,
    print_model_summary,
    count_parameters,
    get_progress_bar,
)
from q2_mechinterp.core.data import (
    BaseDataProcessor,
    DataSplit,
    ProcessorState,
    ScalerMixin,
    BatchInfoMixin,
)
from q2_mechinterp.core.validation import (
    ValidationError,
    validate_array,
    validate_array_pair,
    validate_tensor,
    validate_positive,
    validate_range,
    validate_probability,
    validate_choice,
    validate_type,
    validate_path,
    validate_labels,
    validated,
    validate_inputs,
    check_is_fitted,
    assert_all_finite,
    coerce_to_array,
)

__all__ = [
    # VAE
    "BaseVAE",
    "ConcreteVAE",
    "vae_loss",
    # Sparse
    "SparseAutoencoder",
    "TopKSparseAutoencoder",
    "DeepSparseAutoencoder",
    "sparse_autoencoder_loss",
    # Utils
    "get_device",
    "set_seed",
    "add_noise",
    "check_tensor_validity",
    "clean_array",
    "EarlyStopping",
    # Protocols
    "VAEProtocol",
    "DataProcessorProtocol",
    "SparseAEProtocol",
    "TrainerProtocol",
    # Configs
    "VAEConfig",
    "TrainingConfig",
    "SparseAEConfig",
    "DataConfig",
    "EvaluationConfig",
    "PipelineConfig",
    "create_default_config",
    # Logging
    "get_logger",
    "setup_logging",
    "StructuredLogger",
    "timer",
    "Timer",
    "print_model_summary",
    "count_parameters",
    "get_progress_bar",
    # Data
    "BaseDataProcessor",
    "DataSplit",
    "ProcessorState",
    "ScalerMixin",
    "BatchInfoMixin",
    # Validation
    "ValidationError",
    "validate_array",
    "validate_array_pair",
    "validate_tensor",
    "validate_positive",
    "validate_range",
    "validate_probability",
    "validate_choice",
    "validate_type",
    "validate_path",
    "validate_labels",
    "validated",
    "validate_inputs",
    "check_is_fitted",
    "assert_all_finite",
    "coerce_to_array",
]
