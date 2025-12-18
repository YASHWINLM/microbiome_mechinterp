"""
Evaluation utilities for VAE feature quality assessment.

This module provides tools for:
- Feature evaluation with downstream ML models
- Model architecture comparison with statistical testing
- Feature quality metrics and representation similarity

Example:
    >>> from q2_mechinterp.evaluation import FeatureEvaluator, VAEComparator
    >>> evaluator = FeatureEvaluator(task='classification')
    >>> result = evaluator.evaluate(X_latent, y, model='logistic')
"""

from q2_mechinterp.evaluation.feature_evaluation import (
    EvaluationResult,
    ComparisonResult,
    FeatureEvaluator,
    ModelRegistry,
    calculate_feature_quality_metrics,
    calculate_downstream_utility,
    calculate_representation_similarity,
)

from q2_mechinterp.evaluation.model_comparison import (
    ModelComparisonResult,
    VAEComparator,
    EnsembleComparator,
    bootstrap_confidence_interval,
    permutation_test,
)

__all__ = [
    # Feature evaluation
    "EvaluationResult",
    "ComparisonResult",
    "FeatureEvaluator",
    "ModelRegistry",
    "calculate_feature_quality_metrics",
    "calculate_downstream_utility",
    "calculate_representation_similarity",
    # Model comparison
    "ModelComparisonResult",
    "VAEComparator",
    "EnsembleComparator",
    "bootstrap_confidence_interval",
    "permutation_test",
]
