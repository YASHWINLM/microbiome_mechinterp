"""
Downstream task integration for VAE features.

This module provides sklearn-compatible tools for:
- Feature selection based on sparse AE activations
- End-to-end classifiers using VAE + sparse AE features
- Integration with sklearn pipelines
- Ensemble methods for VAE models

Example:
    >>> from q2_mechinterp.downstream import (
    ...     FeatureSelector,
    ...     SparseFeatureClassifier,
    ...     VAEFeatureTransformer
    ... )
    >>>
    >>> # Feature selection
    >>> selector = FeatureSelector(method='activation_frequency', n_features=100)
    >>> X_selected = selector.fit_transform(sparse_features)
    >>>
    >>> # End-to-end classifier
    >>> clf = SparseFeatureClassifier(vae=vae, classifier='logistic')
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>>
    >>> # Pipeline integration
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([
    ...     ('features', VAEFeatureTransformer(vae)),
    ...     ('classifier', LogisticRegression())
    ... ])
"""

from q2_mechinterp.downstream.downstream import (
    FeatureSelector,
    SupervisedFeatureSelector,
    SparseFeatureClassifier,
    VAEFeatureTransformer,
    VAEEnsembleClassifier,
)

__all__ = [
    "FeatureSelector",
    "SupervisedFeatureSelector",
    "SparseFeatureClassifier",
    "VAEFeatureTransformer",
    "VAEEnsembleClassifier",
]
