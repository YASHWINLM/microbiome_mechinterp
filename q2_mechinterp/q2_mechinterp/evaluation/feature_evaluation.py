"""
Feature evaluation utilities using downstream ML models.

This module provides:
- FeatureEvaluator for cross-validated assessment
- ModelRegistry for available ML models
- Feature quality metrics (reconstruction, sparsity, etc.)

Design Principles:
- Model-agnostic: Works with any sklearn-compatible model
- Extensible: Easy to add new models and metrics
- Reproducible: Fixed random states for all operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
import time


# =============================================================================
# Result Containers
# =============================================================================


@dataclass
class EvaluationResult:
    """
    Container for single evaluation result.

    Attributes:
        model_name: Name of the ML model used
        feature_set: Name of the feature set evaluated
        cv_scores: Cross-validation scores per fold
        mean_score: Mean score across folds
        std_score: Standard deviation across folds
        metric_name: Name of the scoring metric
        n_features: Number of features in the set
        train_time: Total training time
        extra_info: Additional information
    """

    model_name: str
    feature_set: str
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    metric_name: str
    n_features: int
    train_time: float = 0.0
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult({self.feature_set}/{self.model_name}: "
            f"{self.mean_score:.4f}±{self.std_score:.4f} {self.metric_name})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_set": self.feature_set,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "metric_name": self.metric_name,
            "n_features": self.n_features,
            "train_time": self.train_time,
            "cv_scores": list(self.cv_scores),
            **self.extra_info,
        }


@dataclass
class ComparisonResult:
    """
    Container for feature comparison results.

    Attributes:
        results_df: DataFrame with all evaluation results
        best_by_model: Best feature set for each model
        best_overall: Best (feature_set, model) combination
        statistical_tests: Pairwise statistical significance tests
    """

    results_df: pd.DataFrame
    best_by_model: Dict[str, str]
    best_overall: Tuple[str, str]
    statistical_tests: Optional[Dict[str, Dict[str, float]]] = None

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Feature Comparison Summary",
            "=" * 40,
            f"Best overall: {self.best_overall[0]} + {self.best_overall[1]}",
            "",
            "Best by model:",
        ]
        for model, feature_set in self.best_by_model.items():
            lines.append(f"  {model}: {feature_set}")
        return "\n".join(lines)


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """
    Registry of available ML models for evaluation.

    Models are specified as (module_path, class_name, default_params).
    """

    CLASSIFIERS = {
        "logistic": (
            "sklearn.linear_model",
            "LogisticRegression",
            {"max_iter": 1000, "solver": "lbfgs"},
        ),
        "rf": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100, "n_jobs": -1}),
        "svm": ("sklearn.svm", "SVC", {"kernel": "rbf", "probability": True}),
        "knn": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
        "mlp": (
            "sklearn.neural_network",
            "MLPClassifier",
            {"hidden_layer_sizes": (100,), "max_iter": 500},
        ),
        "xgboost": (
            "xgboost",
            "XGBClassifier",
            {"use_label_encoder": False, "eval_metric": "logloss", "verbosity": 0},
        ),
        "lightgbm": ("lightgbm", "LGBMClassifier", {"verbose": -1, "n_jobs": -1}),
        "gradient_boosting": (
            "sklearn.ensemble",
            "GradientBoostingClassifier",
            {"n_estimators": 100},
        ),
    }

    REGRESSORS = {
        "ridge": ("sklearn.linear_model", "Ridge", {"alpha": 1.0}),
        "lasso": ("sklearn.linear_model", "Lasso", {"alpha": 1.0, "max_iter": 1000}),
        "elastic": (
            "sklearn.linear_model",
            "ElasticNet",
            {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000},
        ),
        "rf": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 100, "n_jobs": -1}),
        "svr": ("sklearn.svm", "SVR", {"kernel": "rbf"}),
        "xgboost": ("xgboost", "XGBRegressor", {"verbosity": 0}),
        "lightgbm": ("lightgbm", "LGBMRegressor", {"verbose": -1, "n_jobs": -1}),
        "gradient_boosting": (
            "sklearn.ensemble",
            "GradientBoostingRegressor",
            {"n_estimators": 100},
        ),
    }

    @classmethod
    def get_model(
        cls,
        name: str,
        task: str = "classification",
        random_state: Optional[int] = None,
        **override_params,
    ):
        """
        Get instantiated model by name.

        Args:
            name: Model name
            task: 'classification' or 'regression'
            random_state: Random seed
            **override_params: Override default parameters

        Returns:
            Instantiated sklearn-compatible model
        """
        registry = cls.CLASSIFIERS if task == "classification" else cls.REGRESSORS

        if name not in registry:
            raise ValueError(
                f"Unknown {task} model: {name}. " f"Available: {list(registry.keys())}"
            )

        module_path, class_name, default_params = registry[name]

        import importlib

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Could not import {module_path}. " f"Install required package: {e}")

        model_class = getattr(module, class_name)

        params = {**default_params, **override_params}

        if random_state is not None and name not in ["knn", "svr", "svm"]:
            params["random_state"] = random_state

        return model_class(**params)

    @classmethod
    def list_models(cls, task: str = "classification") -> List[str]:
        """List available models for a task."""
        registry = cls.CLASSIFIERS if task == "classification" else cls.REGRESSORS
        return list(registry.keys())


# =============================================================================
# Feature Evaluator
# =============================================================================


class FeatureEvaluator:
    """
    Evaluate feature quality using downstream ML models.

    Example:
        >>> evaluator = FeatureEvaluator(task='classification', cv=5)
        >>> result = evaluator.evaluate(X_latent, y, model='logistic')
        >>> print(f"Accuracy: {result.mean_score:.3f}")

    Attributes:
        task: 'classification' or 'regression'
        cv: Number of cross-validation folds
        scoring: Scoring metric
        random_state: Random seed
    """

    def __init__(
        self,
        task: str = "classification",
        cv: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            task: 'classification' or 'regression'
            cv: Number of cross-validation folds
            scoring: Scoring metric (None for task default)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            verbose: Whether to print progress
        """
        if task not in ["classification", "regression"]:
            raise ValueError(f"task must be 'classification' or 'regression', got {task}")

        self.task = task
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        if scoring is None:
            self.scoring = "accuracy" if task == "classification" else "neg_mean_squared_error"
        else:
            self.scoring = scoring

        self._results: List[EvaluationResult] = []

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "logistic",
        feature_set_name: str = "features",
        scale: bool = True,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single feature set with a single model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector
            model_name: Name of model to use
            feature_set_name: Name for the feature set
            scale: Whether to standardize features
            model_params: Additional model parameters

        Returns:
            EvaluationResult with cross-validation scores
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        model_params = model_params or {}
        model = ModelRegistry.get_model(
            model_name, task=self.task, random_state=self.random_state, **model_params
        )

        if self.task == "classification":
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        start_time = time.time()
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
        except Exception as e:
            warnings.warn(f"Evaluation failed for {model_name}: {e}")
            scores = np.array([np.nan] * self.cv)

        train_time = time.time() - start_time

        result = EvaluationResult(
            model_name=model_name,
            feature_set=feature_set_name,
            cv_scores=scores,
            mean_score=np.nanmean(scores),
            std_score=np.nanstd(scores),
            metric_name=self.scoring,
            n_features=X.shape[1],
            train_time=train_time,
        )

        self._results.append(result)

        if self.verbose:
            print(f"  {model_name}: {result.mean_score:.4f} ± {result.std_score:.4f}")

        return result

    def compare_features(
        self,
        feature_sets: Dict[str, np.ndarray],
        y: np.ndarray,
        models: Optional[List[str]] = None,
        scale_features: bool = True,
    ) -> ComparisonResult:
        """
        Compare multiple feature sets with multiple models.

        Args:
            feature_sets: Dict mapping feature set names to feature matrices
            y: Target vector
            models: List of model names (None for defaults)
            scale_features: Whether to standardize features

        Returns:
            ComparisonResult with summary DataFrame and best models
        """
        if models is None:
            models = ["logistic", "rf"] if self.task == "classification" else ["ridge", "rf"]

        results = []

        for feat_name, X in feature_sets.items():
            if self.verbose:
                print(f"\nEvaluating: {feat_name} (shape: {X.shape})")

            for model_name in models:
                try:
                    result = self.evaluate(
                        X,
                        y,
                        model_name=model_name,
                        feature_set_name=feat_name,
                        scale=scale_features,
                    )
                    results.append(result.to_dict())
                except Exception as e:
                    if self.verbose:
                        print(f"  {model_name}: FAILED - {e}")

        results_df = pd.DataFrame(results)

        best_by_model = {}
        if not results_df.empty:
            for model in models:
                model_results = results_df[results_df["model_name"] == model]
                if not model_results.empty:
                    best_idx = model_results["mean_score"].idxmax()
                    best_by_model[model] = model_results.loc[best_idx, "feature_set"]

        best_overall = ("", "")
        if not results_df.empty:
            best_idx = results_df["mean_score"].idxmax()
            best_overall = (
                results_df.loc[best_idx, "feature_set"],
                results_df.loc[best_idx, "model_name"],
            )

        return ComparisonResult(
            results_df=results_df, best_by_model=best_by_model, best_overall=best_overall
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as a DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self._results])

    def clear_results(self) -> None:
        """Clear stored results."""
        self._results = []


# =============================================================================
# Feature Quality Metrics
# =============================================================================


def calculate_feature_quality_metrics(
    original: np.ndarray,
    latent: np.ndarray,
    reconstructed: np.ndarray,
    sparse: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate various feature quality metrics.

    Args:
        original: Original input data (n_samples, n_original_features)
        latent: Latent representations (n_samples, latent_dim)
        reconstructed: Reconstructed data (n_samples, n_original_features)
        sparse: Optional sparse features (n_samples, sparse_dim)

    Returns:
        Dictionary with quality metrics
    """
    metrics = {}

    # Reconstruction quality
    metrics["reconstruction_mse"] = float(np.mean((original - reconstructed) ** 2))
    metrics["reconstruction_mae"] = float(np.mean(np.abs(original - reconstructed)))

    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - original.mean()) ** 2)
    metrics["reconstruction_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Per-feature correlation
    correlations = []
    for j in range(original.shape[1]):
        if np.std(original[:, j]) > 0 and np.std(reconstructed[:, j]) > 0:
            corr = np.corrcoef(original[:, j], reconstructed[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    metrics["mean_feature_correlation"] = float(np.mean(correlations)) if correlations else 0.0

    # Latent space metrics
    metrics["latent_dim"] = latent.shape[1]
    metrics["latent_variance_total"] = float(np.var(latent))
    metrics["latent_variance_per_dim"] = float(np.mean(np.var(latent, axis=0)))
    metrics["compression_ratio"] = original.shape[1] / latent.shape[1]

    dim_variances = np.var(latent, axis=0)
    threshold = np.mean(dim_variances) * 0.1
    metrics["active_latent_dims"] = int(np.sum(dim_variances > threshold))
    metrics["latent_utilization"] = metrics["active_latent_dims"] / latent.shape[1]

    # Sparse features metrics
    if sparse is not None:
        metrics["sparse_dim"] = sparse.shape[1]
        metrics["expansion_ratio"] = sparse.shape[1] / latent.shape[1]
        metrics["sparsity"] = float((sparse == 0).mean())
        metrics["mean_active_per_sample"] = float((sparse > 0).sum(axis=1).mean())
        metrics["std_active_per_sample"] = float((sparse > 0).sum(axis=1).std())

        activation_freq = (sparse > 0).mean(axis=0)
        metrics["mean_activation_frequency"] = float(activation_freq.mean())
        metrics["max_activation_frequency"] = float(activation_freq.max())
        metrics["min_nonzero_activation_freq"] = (
            float(activation_freq[activation_freq > 0].min())
            if (activation_freq > 0).any()
            else 0.0
        )

        metrics["dead_features"] = int((activation_freq == 0).sum())
        metrics["dead_feature_ratio"] = metrics["dead_features"] / sparse.shape[1]

    return metrics


def calculate_downstream_utility(
    features: np.ndarray,
    labels: np.ndarray,
    task: str = "classification",
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Quick utility assessment using simple models.

    Args:
        features: Feature matrix
        labels: Target labels
        task: 'classification' or 'regression'
        cv: Cross-validation folds
        random_state: Random seed

    Returns:
        Dictionary with utility metrics
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    metrics = {}

    if task == "classification":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=random_state)
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        scores = cross_val_score(model, features_scaled, labels, cv=cv_split, scoring="accuracy")
        metrics["accuracy"] = float(scores.mean())
        metrics["accuracy_std"] = float(scores.std())

        scores = cross_val_score(
            model, features_scaled, labels, cv=cv_split, scoring="balanced_accuracy"
        )
        metrics["balanced_accuracy"] = float(scores.mean())
    else:
        from sklearn.linear_model import Ridge

        model = Ridge()
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        scores = cross_val_score(model, features_scaled, labels, cv=cv_split, scoring="r2")
        metrics["r2"] = float(scores.mean())
        metrics["r2_std"] = float(scores.std())

        scores = cross_val_score(
            model, features_scaled, labels, cv=cv_split, scoring="neg_mean_squared_error"
        )
        metrics["mse"] = float(-scores.mean())

    return metrics


def calculate_representation_similarity(
    features1: np.ndarray, features2: np.ndarray, method: str = "cka"
) -> float:
    """
    Calculate similarity between two feature representations.

    Args:
        features1: First feature matrix (n_samples, n_features1)
        features2: Second feature matrix (n_samples, n_features2)
        method: 'cka' (Centered Kernel Alignment) or 'procrustes'

    Returns:
        Similarity score (0 to 1)
    """
    if len(features1) != len(features2):
        raise ValueError("Feature matrices must have same number of samples")

    if method == "cka":

        def _centering_matrix(n):
            return np.eye(n) - np.ones((n, n)) / n

        n = len(features1)
        H = _centering_matrix(n)

        K1 = features1 @ features1.T
        K2 = features2 @ features2.T

        HK1H = H @ K1 @ H
        HK2H = H @ K2 @ H

        hsic = np.trace(HK1H @ HK2H)
        norm1 = np.sqrt(np.trace(HK1H @ HK1H))
        norm2 = np.sqrt(np.trace(HK2H @ HK2H))

        if norm1 > 0 and norm2 > 0:
            return float(hsic / (norm1 * norm2))
        return 0.0

    elif method == "procrustes":
        from scipy.spatial import procrustes

        min_dim = min(features1.shape[1], features2.shape[1])

        if features1.shape[1] > min_dim or features2.shape[1] > min_dim:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min_dim)
            features1 = pca.fit_transform(features1)
            features2 = pca.fit_transform(features2)

        _, _, disparity = procrustes(features1, features2)
        return float(1 - disparity)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'cka' or 'procrustes'")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "EvaluationResult",
    "ComparisonResult",
    "FeatureEvaluator",
    "ModelRegistry",
    "calculate_feature_quality_metrics",
    "calculate_downstream_utility",
    "calculate_representation_similarity",
]
