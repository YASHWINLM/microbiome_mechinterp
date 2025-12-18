"""
Preprocessing utilities for batch effect correction.

This module provides tools for correcting batch effects in biological data:
- SimpleBatchCorrector: Basic centering/scaling methods
- ComBatCorrector: Empirical Bayes batch correction
- ConQuRCorrector: Conditional quantile regression for microbiome data
- PreprocessingPipeline: Chain multiple correction steps

Example:
    >>> from q2_mechinterp.preprocessing import ComBatCorrector, detect_batch_effects
    >>> combat = ComBatCorrector(parametric=True)
    >>> X_corrected = combat.fit_transform(X, batch_labels)
    >>> metrics = detect_batch_effects(X_corrected, batch_labels)
"""

from q2_mechinterp.preprocessing.batch_correction import (
    BaseBatchCorrector,
    SimpleBatchCorrector,
    ComBatCorrector,
    ConQuRCorrector,
    PreprocessingPipeline,
    detect_batch_effects,
    evaluate_correction,
)

__all__ = [
    "BaseBatchCorrector",
    "SimpleBatchCorrector",
    "ComBatCorrector",
    "ConQuRCorrector",
    "PreprocessingPipeline",
    "detect_batch_effects",
    "evaluate_correction",
]
