"""
Hyperparameter tuning module for VAE models.

This module provides tools for automated hyperparameter optimization:
- VAETuner: Optuna-based Bayesian optimization
- GridSearch: Exhaustive grid search for small spaces
- TuningResult: Structured results container

Example:
    >>> from q2_mechinterp.tuning import VAETuner, DEFAULT_PARAM_SPACE
    >>> tuner = VAETuner(MicrobiomeVAE, device, param_space=DEFAULT_PARAM_SPACE)
    >>> result = tuner.tune(X_train, X_val, n_trials=50)
"""

from q2_mechinterp.tuning.hyperparams import (
    TuningResult,
    VAETuner,
    GridSearch,
    DEFAULT_PARAM_SPACE,
)

__all__ = [
    "TuningResult",
    "VAETuner",
    "GridSearch",
    "DEFAULT_PARAM_SPACE",
]