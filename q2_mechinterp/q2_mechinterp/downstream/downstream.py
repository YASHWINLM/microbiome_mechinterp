"""
Downstream task integration for VAE features.

This module provides:
- Feature selection based on sparse AE activations
- End-to-end classifiers using VAE + sparse AE features
- Integration with sklearn pipelines

Design Principles:
- Sklearn-compatible API (fit/transform/predict)
- Modular components that work standalone or combined
- Interpretable feature selection
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List, Dict, Any, Union, Tuple, Type, Callable
from dataclasses import dataclass
import warnings

# Sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# =============================================================================
# Feature Selection
# =============================================================================


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on sparse autoencoder activations.

    This selector identifies which original features are most important
    by analyzing which sparse AE units are most active and which
    original features contribute most to those units.

    Selection methods:
    - 'activation_frequency': Select features that activate most frequently
    - 'mean_activation': Select features with highest mean activation
    - 'variance': Select features with highest activation variance
    - 'top_k_per_unit': Select top-K contributing features per sparse unit

    Example:
        >>> # With FeatureExtractor results
        >>> selector = FeatureSelector(method='activation_frequency', n_features=100)
        >>> selector.fit(sparse_features)
        >>> X_selected = selector.transform(sparse_features)
        >>>
        >>> # Get which features were selected
        >>> selected_idx = selector.get_selected_indices()

    Args:
        method: Selection method
        n_features: Number of features to select (or fraction if < 1)
        threshold: Activation threshold for frequency method
    """

    def __init__(
        self,
        method: str = "activation_frequency",
        n_features: Union[int, float] = 100,
        threshold: float = 0.0,
    ):
        self.method = method
        self.n_features = n_features
        self.threshold = threshold

        self._selected_indices: Optional[np.ndarray] = None
        self._feature_scores: Optional[np.ndarray] = None
        self._n_features_in: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FeatureSelector":
        """
        Fit selector to sparse features.

        Args:
            X: Sparse feature matrix (n_samples, n_sparse_features)
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        X = np.asarray(X)
        self._n_features_in = X.shape[1]

        # Determine number of features to select
        if self.n_features < 1:
            n_select = int(self._n_features_in * self.n_features)
        else:
            n_select = int(self.n_features)
        n_select = min(n_select, self._n_features_in)

        # Compute feature scores based on method
        if self.method == "activation_frequency":
            # Fraction of samples where feature is active
            self._feature_scores = (X > self.threshold).mean(axis=0)

        elif self.method == "mean_activation":
            # Mean activation value
            self._feature_scores = np.mean(X, axis=0)

        elif self.method == "variance":
            # Variance of activation
            self._feature_scores = np.var(X, axis=0)

        elif self.method == "max_activation":
            # Maximum activation value
            self._feature_scores = np.max(X, axis=0)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Select top features
        self._selected_indices = np.argsort(self._feature_scores)[-n_select:][::-1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features.

        Args:
            X: Feature matrix

        Returns:
            Selected features
        """
        if self._selected_indices is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

        return X[:, self._selected_indices]

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        if self._selected_indices is None:
            raise RuntimeError("Selector not fitted")
        return self._selected_indices.copy()

    def get_feature_scores(self) -> np.ndarray:
        """Get scores for all features."""
        if self._feature_scores is None:
            raise RuntimeError("Selector not fitted")
        return self._feature_scores.copy()

    def get_feature_ranking(self) -> np.ndarray:
        """Get feature indices sorted by score (descending)."""
        if self._feature_scores is None:
            raise RuntimeError("Selector not fitted")
        return np.argsort(self._feature_scores)[::-1]


class SupervisedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select features based on correlation with target.

    Uses correlation or mutual information to select features
    most predictive of the target variable.
    """

    def __init__(
        self,
        method: str = "correlation",
        n_features: Union[int, float] = 100,
        task: str = "classification",
    ):
        """
        Initialize selector.

        Args:
            method: 'correlation', 'mutual_info', or 'f_score'
            n_features: Number of features to select
            task: 'classification' or 'regression'
        """
        self.method = method
        self.n_features = n_features
        self.task = task

        self._selected_indices: Optional[np.ndarray] = None
        self._feature_scores: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedFeatureSelector":
        """
        Fit selector to features and target.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_features_in = X.shape[1]

        # Determine number to select
        if self.n_features < 1:
            n_select = int(n_features_in * self.n_features)
        else:
            n_select = int(self.n_features)
        n_select = min(n_select, n_features_in)

        # Compute scores
        if self.method == "correlation":
            # Absolute correlation with target
            scores = []
            for j in range(n_features_in):
                if np.std(X[:, j]) > 0:
                    corr = np.abs(np.corrcoef(X[:, j], y)[0, 1])
                    scores.append(corr if not np.isnan(corr) else 0)
                else:
                    scores.append(0)
            self._feature_scores = np.array(scores)

        elif self.method == "mutual_info":
            if self.task == "classification":
                from sklearn.feature_selection import mutual_info_classif

                self._feature_scores = mutual_info_classif(X, y)
            else:
                from sklearn.feature_selection import mutual_info_regression

                self._feature_scores = mutual_info_regression(X, y)

        elif self.method == "f_score":
            if self.task == "classification":
                from sklearn.feature_selection import f_classif

                self._feature_scores, _ = f_classif(X, y)
            else:
                from sklearn.feature_selection import f_regression

                self._feature_scores, _ = f_regression(X, y)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Handle NaNs
        self._feature_scores = np.nan_to_num(self._feature_scores, 0)

        # Select top features
        self._selected_indices = np.argsort(self._feature_scores)[-n_select:][::-1]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features."""
        if self._selected_indices is None:
            raise RuntimeError("Selector not fitted")
        return X[:, self._selected_indices]

    def get_selected_indices(self) -> np.ndarray:
        """Get selected feature indices."""
        if self._selected_indices is None:
            raise RuntimeError("Selector not fitted")
        return self._selected_indices.copy()


# =============================================================================
# End-to-End Classifier
# =============================================================================


class SparseFeatureClassifier(BaseEstimator, ClassifierMixin):
    """
    End-to-end classifier using VAE + sparse AE features.

    This classifier:
    1. Extracts latent features from a trained VAE
    2. Applies sparse autoencoder for interpretability
    3. Uses a downstream classifier for prediction

    Example:
        >>> # Train VAE and sparse AE first
        >>> vae = MicrobiomeVAE(input_dim=500, latent_dim=50)
        >>> # ... train VAE ...
        >>>
        >>> # Create end-to-end classifier
        >>> clf = SparseFeatureClassifier(
        ...     vae=vae,
        ...     sparse_ae=sparse_ae,  # Optional
        ...     classifier='logistic'
        ... )
        >>>
        >>> # Fit and predict
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)

    Args:
        vae: Trained VAE model
        sparse_ae: Optional trained sparse AE
        classifier: Classifier type or sklearn classifier instance
        feature_selector: Optional feature selector
        device: PyTorch device
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        sparse_ae: Optional[torch.nn.Module] = None,
        classifier: Union[str, BaseEstimator] = "logistic",
        feature_selector: Optional[FeatureSelector] = None,
        device: Optional[torch.device] = None,
        scale_features: bool = True,
    ):
        self.vae = vae
        self.sparse_ae = sparse_ae
        self.classifier = classifier
        self.feature_selector = feature_selector
        self.device = device or torch.device("cpu")
        self.scale_features = scale_features

        self._scaler: Optional[StandardScaler] = None
        self._classifier_fitted: Optional[BaseEstimator] = None
        self._classes: Optional[np.ndarray] = None

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features through VAE and optional sparse AE."""
        self.vae.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            # Get latent features from VAE
            mu, _ = self.vae.encode(X_tensor)
            features = mu.cpu().numpy()

            # Apply sparse AE if available
            if self.sparse_ae is not None:
                self.sparse_ae.eval()
                features_tensor = torch.FloatTensor(features).to(self.device)
                features = self.sparse_ae.get_sparse_features(features_tensor).cpu().numpy()

        # Apply feature selection if available
        if self.feature_selector is not None and hasattr(
            self.feature_selector, "_selected_indices"
        ):
            if self.feature_selector._selected_indices is not None:
                features = self.feature_selector.transform(features)

        return features

    def _get_classifier(self) -> BaseEstimator:
        """Get classifier instance."""
        if isinstance(self.classifier, str):
            classifiers = {
                "logistic": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(n_estimators=100),
                "svm": SVC(probability=True),
            }
            if self.classifier not in classifiers:
                raise ValueError(f"Unknown classifier: {self.classifier}")
            return classifiers[self.classifier]
        return self.classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseFeatureClassifier":
        """
        Fit the classifier.

        Args:
            X: Training data (original features)
            y: Training labels

        Returns:
            self
        """
        # Extract features
        features = self._extract_features(X)

        # Fit feature selector if provided
        if self.feature_selector is not None:
            self.feature_selector.fit(features, y)
            features = self.feature_selector.transform(features)

        # Scale features
        if self.scale_features:
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features)

        # Fit classifier
        self._classifier_fitted = self._get_classifier()
        self._classifier_fitted.fit(features, y)
        self._classes = np.unique(y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test data (original features)

        Returns:
            Predicted labels
        """
        if self._classifier_fitted is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        features = self._extract_features(X)

        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)

        if self._scaler is not None:
            features = self._scaler.transform(features)

        return self._classifier_fitted.predict(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test data

        Returns:
            Class probabilities
        """
        if self._classifier_fitted is None:
            raise RuntimeError("Classifier not fitted")

        features = self._extract_features(X)

        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)

        if self._scaler is not None:
            features = self._scaler.transform(features)

        return self._classifier_fitted.predict_proba(features)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Test data
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return (predictions == y).mean()

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from classifier (if available).

        Returns:
            Feature importance array or None
        """
        if self._classifier_fitted is None:
            return None

        if hasattr(self._classifier_fitted, "feature_importances_"):
            return self._classifier_fitted.feature_importances_
        elif hasattr(self._classifier_fitted, "coef_"):
            return np.abs(self._classifier_fitted.coef_).mean(axis=0)

        return None


# =============================================================================
# Pipeline Integration
# =============================================================================


class VAEFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer wrapper for VAE feature extraction.

    Allows easy integration into sklearn pipelines.

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>>
        >>> pipe = Pipeline([
        ...     ('q2_mechinterp', VAEFeatureTransformer(vae)),
        ...     ('scaler', StandardScaler()),
        ...     ('classifier', LogisticRegression())
        ... ])
        >>>
        >>> pipe.fit(X_train, y_train)
        >>> pipe.score(X_test, y_test)
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        sparse_ae: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        use_mean: bool = True,
    ):
        """
        Initialize transformer.

        Args:
            vae: Trained VAE model
            sparse_ae: Optional sparse AE
            device: PyTorch device
            use_mean: Use latent mean (True) or sample (False)
        """
        self.vae = vae
        self.sparse_ae = sparse_ae
        self.device = device or torch.device("cpu")
        self.use_mean = use_mean

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "VAEFeatureTransformer":
        """Fit (no-op, models should be pre-trained)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to VAE features.

        Args:
            X: Input data

        Returns:
            Feature matrix
        """
        self.vae.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)

            mu, logvar = self.vae.encode(X_tensor)

            if self.use_mean:
                features = mu
            else:
                features = self.vae.reparameterize(mu, logvar)

            features = features.cpu().numpy()

            if self.sparse_ae is not None:
                self.sparse_ae.eval()
                features_tensor = torch.FloatTensor(features).to(self.device)
                features = self.sparse_ae.get_sparse_features(features_tensor).cpu().numpy()

        return features


# =============================================================================
# Ensemble Methods
# =============================================================================


class VAEEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using multiple VAE models.

    Combines predictions from multiple VAEs trained with
    different seeds or architectures.
    """

    def __init__(
        self,
        vaes: List[torch.nn.Module],
        classifier: Union[str, BaseEstimator] = "logistic",
        ensemble_method: str = "average",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ensemble.

        Args:
            vaes: List of trained VAE models
            classifier: Base classifier
            ensemble_method: 'average' or 'concatenate'
            device: PyTorch device
        """
        self.vaes = vaes
        self.classifier = classifier
        self.ensemble_method = ensemble_method
        self.device = device or torch.device("cpu")

        self._classifiers: List[BaseEstimator] = []
        self._scalers: List[StandardScaler] = []

    def _get_classifier(self) -> BaseEstimator:
        if isinstance(self.classifier, str):
            classifiers = {
                "logistic": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(n_estimators=100),
            }
            return classifiers.get(self.classifier, LogisticRegression())
        return self.classifier

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VAEEnsembleClassifier":
        """Fit ensemble classifier."""
        self._classifiers = []
        self._scalers = []

        for vae in self.vaes:
            vae.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                mu, _ = vae.encode(X_tensor)
                features = mu.cpu().numpy()

            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            clf = self._get_classifier()
            clf.fit(features, y)

            self._classifiers.append(clf)
            self._scalers.append(scaler)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get averaged probability predictions."""
        all_probs = []

        for vae, clf, scaler in zip(self.vaes, self._classifiers, self._scalers):
            vae.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                mu, _ = vae.encode(X_tensor)
                features = mu.cpu().numpy()

            features = scaler.transform(features)
            probs = clf.predict_proba(features)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return (self.predict(X) == y).mean()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "FeatureSelector",
    "SupervisedFeatureSelector",
    "SparseFeatureClassifier",
    "VAEFeatureTransformer",
    "VAEEnsembleClassifier",
]
