"""
Training utilities for VAE models.

This module provides:
- VAETrainer for standard VAE training
- CrossValidator for k-fold cross-validation
- SemiSupervisedVAE for combined labeled/unlabeled training
"""

from q2_mechinterp.training.trainer import (
    VAETrainer,
    create_data_loaders
)
from q2_mechinterp.training.cross_validation import (
    CVResult,
    CrossValidator,
    NestedCV,
    RepeatedCV,
)
from q2_mechinterp.training.semi_supervised import (
    SemiSupervisedVAE,
    semi_supervised_loss,
    SemiSupervisedTrainer,
)

__all__ = [
    # Core training
    "VAETrainer",
    "create_data_loaders",
    # Cross-validation
    "CVResult",
    "CrossValidator",
    "NestedCV",
    "RepeatedCV",
    # Semi-supervised
    "SemiSupervisedVAE",
    "semi_supervised_loss",
    "SemiSupervisedTrainer",
]
