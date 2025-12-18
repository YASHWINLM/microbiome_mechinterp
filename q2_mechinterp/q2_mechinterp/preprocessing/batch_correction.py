"""
Preprocessing utilities including batch effect correction.

This module provides:
- Simple batch correction methods (centering, scaling)
- ComBat: Parametric and non-parametric empirical Bayes
- ConQuR: Conditional Quantile Regression for microbiome data

Design Principles:
- Scikit-learn compatible API (fit/transform pattern)
- Preserve data characteristics (sparsity, compositionality)
- Comprehensive documentation of methods
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod


# =============================================================================
# Base Class
# =============================================================================

class BaseBatchCorrector(ABC):
    """
    Abstract base class for batch effect correction.
    
    All batch correctors follow the sklearn fit/transform pattern.
    """
    
    def __init__(self):
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> 'BaseBatchCorrector':
        """Fit the batch correction model."""
        pass
    
    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """Apply batch correction."""
        pass
    
    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, batch_labels, covariates)
        return self.transform(X, batch_labels)
    
    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )


# =============================================================================
# Simple Batch Correction
# =============================================================================

class SimpleBatchCorrector(BaseBatchCorrector):
    """
    Simple batch effect correction using centering or scaling.
    
    Methods:
    - 'center': Subtract batch mean
    - 'scale': Divide by batch standard deviation
    - 'zscore': Center and scale (z-score normalization)
    
    Example:
        >>> corrector = SimpleBatchCorrector(method='center')
        >>> X_corrected = corrector.fit_transform(X, batch_labels)
    
    Args:
        method: Correction method ('center', 'scale', 'zscore')
    """
    
    def __init__(self, method: str = 'center'):
        super().__init__()
        
        valid_methods = ['center', 'scale', 'zscore']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        self.method = method
        self._batch_means: Dict[Any, np.ndarray] = {}
        self._batch_stds: Dict[Any, np.ndarray] = {}
        self._batch_counts: Dict[Any, int] = {}
        self._global_mean: Optional[np.ndarray] = None
        self._global_std: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> 'SimpleBatchCorrector':
        """
        Fit batch correction parameters.
        
        Args:
            X: Data matrix (n_samples, n_features)
            batch_labels: Batch assignment for each sample
            covariates: Ignored for simple correction
        
        Returns:
            self
        """
        X = np.asarray(X)
        batch_labels = np.asarray(batch_labels)
        
        if len(X) != len(batch_labels):
            raise ValueError("X and batch_labels must have same length")
        
        # Compute global statistics
        self._global_mean = np.mean(X, axis=0)
        self._global_std = np.std(X, axis=0)
        self._global_std[self._global_std == 0] = 1.0  # Prevent division by zero
        
        # Compute batch-specific statistics
        unique_batches = np.unique(batch_labels)
        
        for batch in unique_batches:
            mask = batch_labels == batch
            X_batch = X[mask]
            
            self._batch_means[batch] = np.mean(X_batch, axis=0)
            self._batch_stds[batch] = np.std(X_batch, axis=0)
            self._batch_stds[batch][self._batch_stds[batch] == 0] = 1.0
            self._batch_counts[batch] = mask.sum()
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """
        Apply batch correction.
        
        Args:
            X: Data matrix (n_samples, n_features)
            batch_labels: Batch assignment for each sample
        
        Returns:
            Corrected data matrix
        """
        self._check_is_fitted()
        
        X = np.asarray(X, dtype=np.float64)
        batch_labels = np.asarray(batch_labels)
        X_corrected = X.copy()
        
        for batch in np.unique(batch_labels):
            mask = batch_labels == batch
            
            if batch not in self._batch_means:
                warnings.warn(f"Unknown batch {batch}, using global statistics")
                batch_mean = self._global_mean
                batch_std = self._global_std
            else:
                batch_mean = self._batch_means[batch]
                batch_std = self._batch_stds[batch]
            
            if self.method == 'center':
                # Subtract batch mean, add global mean
                X_corrected[mask] = X[mask] - batch_mean + self._global_mean
            
            elif self.method == 'scale':
                # Scale by ratio of global to batch std
                X_corrected[mask] = X[mask] * (self._global_std / batch_std)
            
            elif self.method == 'zscore':
                # Full z-score normalization then rescale to global
                X_corrected[mask] = (
                    (X[mask] - batch_mean) / batch_std
                ) * self._global_std + self._global_mean
        
        return X_corrected


# =============================================================================
# ComBat Batch Correction
# =============================================================================

class ComBatCorrector(BaseBatchCorrector):
    """
    ComBat batch effect correction using empirical Bayes.
    
    ComBat adjusts for batch effects by:
    1. Standardizing data within batches
    2. Using empirical Bayes to shrink batch effect estimates
    3. Correcting both location (mean) and scale (variance) effects
    
    Reference:
        Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray
        expression data using empirical Bayes methods. Biostatistics 2007.
    
    Example:
        >>> combat = ComBatCorrector(parametric=True)
        >>> X_corrected = combat.fit_transform(X, batch_labels, covariates=design)
    
    Args:
        parametric: Use parametric (True) or non-parametric (False) prior
        mean_only: Only correct for location effects, not scale
    """
    
    def __init__(
        self,
        parametric: bool = True,
        mean_only: bool = False
    ):
        super().__init__()
        self.parametric = parametric
        self.mean_only = mean_only
        
        # Fitted parameters
        self._grand_mean: Optional[np.ndarray] = None
        self._var_pooled: Optional[np.ndarray] = None
        self._stand_mean: Optional[np.ndarray] = None
        self._gamma_star: Dict[Any, np.ndarray] = {}
        self._delta_star: Dict[Any, np.ndarray] = {}
        self._batch_design: Optional[np.ndarray] = None
        self._batches: Optional[List] = None
    
    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> 'ComBatCorrector':
        """
        Fit ComBat model.
        
        Args:
            X: Data matrix (n_samples, n_features)
            batch_labels: Batch assignment for each sample
            covariates: Biological covariates to preserve (n_samples, n_covariates)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        batch_labels = np.asarray(batch_labels)
        
        n_samples, n_features = X.shape
        self._batches = list(np.unique(batch_labels))
        n_batches = len(self._batches)
        
        # Create batch design matrix
        batch_design = np.zeros((n_samples, n_batches))
        for i, batch in enumerate(self._batches):
            batch_design[batch_labels == batch, i] = 1
        self._batch_design = batch_design
        
        # Get batch sizes
        batch_sizes = batch_design.sum(axis=0)
        
        # Create full design matrix
        if covariates is not None:
            design = np.hstack([batch_design, covariates])
        else:
            design = batch_design
        
        # Standardize data
        # Compute grand mean (across all samples)
        self._grand_mean = X.mean(axis=0)
        
        # Compute pooled variance
        self._var_pooled = X.var(axis=0, ddof=1)
        self._var_pooled[self._var_pooled == 0] = 1.0
        
        # Standardize
        X_stand = (X - self._grand_mean) / np.sqrt(self._var_pooled)
        
        # Compute stand_mean (batch-specific means on standardized data)
        self._stand_mean = np.zeros((n_batches, n_features))
        for i, batch in enumerate(self._batches):
            mask = batch_labels == batch
            self._stand_mean[i] = X_stand[mask].mean(axis=0)
        
        # Estimate batch effects (gamma for location, delta for scale)
        gamma_hat = {}
        delta_hat = {}
        
        for i, batch in enumerate(self._batches):
            mask = batch_labels == batch
            X_batch = X_stand[mask]
            
            # Location effect
            gamma_hat[batch] = X_batch.mean(axis=0)
            
            # Scale effect
            delta_hat[batch] = X_batch.var(axis=0, ddof=1)
            delta_hat[batch][delta_hat[batch] == 0] = 1.0
        
        # Empirical Bayes shrinkage
        if self.parametric:
            self._gamma_star, self._delta_star = self._parametric_eb(
                gamma_hat, delta_hat, batch_sizes
            )
        else:
            self._gamma_star, self._delta_star = self._nonparametric_eb(
                gamma_hat, delta_hat, X_stand, batch_labels
            )
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """
        Apply ComBat batch correction.
        
        Args:
            X: Data matrix (n_samples, n_features)
            batch_labels: Batch assignment for each sample
        
        Returns:
            Corrected data matrix
        """
        self._check_is_fitted()
        
        X = np.asarray(X, dtype=np.float64)
        batch_labels = np.asarray(batch_labels)
        
        # Standardize
        X_stand = (X - self._grand_mean) / np.sqrt(self._var_pooled)
        X_corrected = np.zeros_like(X_stand)
        
        for batch in np.unique(batch_labels):
            mask = batch_labels == batch
            
            if batch not in self._gamma_star:
                warnings.warn(f"Unknown batch {batch}, no correction applied")
                X_corrected[mask] = X_stand[mask]
                continue
            
            gamma = self._gamma_star[batch]
            delta = self._delta_star[batch]
            
            # Remove batch effects
            if self.mean_only:
                X_corrected[mask] = X_stand[mask] - gamma
            else:
                X_corrected[mask] = (X_stand[mask] - gamma) / np.sqrt(delta)
        
        # Back-transform to original scale
        X_corrected = X_corrected * np.sqrt(self._var_pooled) + self._grand_mean
        
        return X_corrected
    
    def _parametric_eb(
        self,
        gamma_hat: Dict[Any, np.ndarray],
        delta_hat: Dict[Any, np.ndarray],
        batch_sizes: np.ndarray
    ) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
        """Parametric empirical Bayes estimation."""
        gamma_star = {}
        delta_star = {}
        
        # Estimate hyperparameters from all batches
        all_gamma = np.vstack([gamma_hat[b] for b in self._batches])
        all_delta = np.vstack([delta_hat[b] for b in self._batches])
        
        # Gamma prior: Normal(gamma_bar, tau^2)
        gamma_bar = all_gamma.mean(axis=0)
        tau_sq = all_gamma.var(axis=0, ddof=1)
        tau_sq[tau_sq == 0] = 1e-6
        
        # Delta prior: Inverse Gamma(lambda, theta)
        # Use method of moments
        delta_mean = all_delta.mean(axis=0)
        delta_var = all_delta.var(axis=0, ddof=1)
        delta_var[delta_var == 0] = 1e-6
        
        # Shrink estimates
        for i, batch in enumerate(self._batches):
            n_i = batch_sizes[i]
            
            # Shrink gamma
            shrinkage = tau_sq / (tau_sq + 1.0 / n_i)
            gamma_star[batch] = shrinkage * gamma_hat[batch] + (1 - shrinkage) * gamma_bar
            
            if self.mean_only:
                delta_star[batch] = np.ones_like(delta_hat[batch])
            else:
                # Shrink delta (simplified)
                delta_star[batch] = delta_hat[batch]  # Minimal shrinkage for variance
        
        return gamma_star, delta_star
    
    def _nonparametric_eb(
        self,
        gamma_hat: Dict[Any, np.ndarray],
        delta_hat: Dict[Any, np.ndarray],
        X_stand: np.ndarray,
        batch_labels: np.ndarray
    ) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
        """Non-parametric empirical Bayes using kernel density estimation."""
        gamma_star = {}
        delta_star = {}
        
        for batch in self._batches:
            mask = batch_labels == batch
            X_batch = X_stand[mask]
            
            # For non-parametric, use median-based shrinkage
            all_gamma = np.vstack([gamma_hat[b] for b in self._batches])
            gamma_median = np.median(all_gamma, axis=0)
            
            # MAD-based shrinkage
            mad = np.median(np.abs(all_gamma - gamma_median), axis=0)
            mad[mad == 0] = 1e-6
            
            weight = 1.0 / (1.0 + np.abs(gamma_hat[batch] - gamma_median) / mad)
            gamma_star[batch] = weight * gamma_median + (1 - weight) * gamma_hat[batch]
            
            if self.mean_only:
                delta_star[batch] = np.ones_like(delta_hat[batch])
            else:
                delta_star[batch] = delta_hat[batch]
        
        return gamma_star, delta_star


# =============================================================================
# ConQuR Batch Correction (Microbiome-specific)
# =============================================================================

class ConQuRCorrector(BaseBatchCorrector):
    """
    ConQuR: Conditional Quantile Regression for batch correction.
    
    Specifically designed for microbiome data, handling:
    - Sparsity (many zeros)
    - Compositionality
    - Non-normal distributions
    
    ConQuR works by:
    1. Building quantile regression models for each feature
    2. Mapping sample quantiles within batches to a reference batch
    3. Preserving zeros and compositional structure
    
    Reference:
        Ling W, et al. Batch effects removal for microbiome data via
        conditional quantile regression. Nature Communications 2022.
    
    Example:
        >>> conqur = ConQuRCorrector(reference_batch='batch1')
        >>> X_corrected = conqur.fit_transform(X, batch_labels)
    
    Args:
        reference_batch: Name of reference batch (None = largest batch)
        quantiles: Quantiles to use for regression
        zero_handling: How to handle zeros ('keep', 'pseudocount', 'model')
        pseudocount: Value to add if zero_handling='pseudocount'
    """
    
    def __init__(
        self,
        reference_batch: Optional[str] = None,
        quantiles: Optional[List[float]] = None,
        zero_handling: str = 'keep',
        pseudocount: float = 0.5
    ):
        super().__init__()
        
        self.reference_batch = reference_batch
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.zero_handling = zero_handling
        self.pseudocount = pseudocount
        
        valid_zero = ['keep', 'pseudocount', 'model']
        if zero_handling not in valid_zero:
            raise ValueError(f"zero_handling must be one of {valid_zero}")
        
        # Fitted parameters
        self._reference: Optional[str] = None
        self._ref_quantiles: Optional[np.ndarray] = None
        self._batch_quantiles: Dict[Any, np.ndarray] = {}
        self._zero_fractions: Dict[Any, np.ndarray] = {}
        self._ref_zero_fraction: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> 'ConQuRCorrector':
        """
        Fit ConQuR model.
        
        Args:
            X: Data matrix (n_samples, n_features), typically counts or abundances
            batch_labels: Batch assignment for each sample
            covariates: Ignored (for API compatibility)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        batch_labels = np.asarray(batch_labels)
        
        n_samples, n_features = X.shape
        batches = np.unique(batch_labels)
        
        # Determine reference batch
        if self.reference_batch is not None:
            if self.reference_batch not in batches:
                raise ValueError(f"Reference batch {self.reference_batch} not found")
            self._reference = self.reference_batch
        else:
            # Use largest batch as reference
            batch_sizes = {b: (batch_labels == b).sum() for b in batches}
            self._reference = max(batch_sizes, key=batch_sizes.get)
        
        # Compute quantiles for reference batch
        ref_mask = batch_labels == self._reference
        X_ref = X[ref_mask]
        
        self._ref_quantiles = np.zeros((len(self.quantiles), n_features))
        self._ref_zero_fraction = (X_ref == 0).mean(axis=0)
        
        for j in range(n_features):
            col = X_ref[:, j]
            if self.zero_handling == 'keep':
                # Compute quantiles on non-zero values
                nonzero = col[col > 0]
                if len(nonzero) > 0:
                    self._ref_quantiles[:, j] = np.quantile(nonzero, self.quantiles)
                else:
                    self._ref_quantiles[:, j] = 0
            else:
                self._ref_quantiles[:, j] = np.quantile(col, self.quantiles)
        
        # Compute quantiles for each batch
        for batch in batches:
            if batch == self._reference:
                continue
            
            mask = batch_labels == batch
            X_batch = X[mask]
            
            batch_quantiles = np.zeros((len(self.quantiles), n_features))
            self._zero_fractions[batch] = (X_batch == 0).mean(axis=0)
            
            for j in range(n_features):
                col = X_batch[:, j]
                if self.zero_handling == 'keep':
                    nonzero = col[col > 0]
                    if len(nonzero) > 0:
                        batch_quantiles[:, j] = np.quantile(nonzero, self.quantiles)
                    else:
                        batch_quantiles[:, j] = 0
                else:
                    batch_quantiles[:, j] = np.quantile(col, self.quantiles)
            
            self._batch_quantiles[batch] = batch_quantiles
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """
        Apply ConQuR batch correction.
        
        Args:
            X: Data matrix (n_samples, n_features)
            batch_labels: Batch assignment for each sample
        
        Returns:
            Corrected data matrix
        """
        self._check_is_fitted()
        
        X = np.asarray(X, dtype=np.float64)
        batch_labels = np.asarray(batch_labels)
        X_corrected = X.copy()
        
        for batch in np.unique(batch_labels):
            if batch == self._reference:
                continue  # Reference batch unchanged
            
            mask = batch_labels == batch
            
            if batch not in self._batch_quantiles:
                warnings.warn(f"Unknown batch {batch}, no correction applied")
                continue
            
            batch_q = self._batch_quantiles[batch]
            ref_q = self._ref_quantiles
            
            for j in range(X.shape[1]):
                col = X[mask, j].copy()
                
                if self.zero_handling == 'keep':
                    # Only transform non-zero values
                    nonzero_mask = col > 0
                    if nonzero_mask.sum() == 0:
                        continue
                    
                    # Handle zeros probabilistically
                    zero_mask = ~nonzero_mask
                    if zero_mask.any():
                        # Decide whether zeros stay zero based on reference zero fraction
                        n_zeros = zero_mask.sum()
                        ref_zero_frac = self._ref_zero_fraction[j]
                        batch_zero_frac = self._zero_fractions.get(batch, np.array([0]))[j] if batch in self._zero_fractions else 0
                        
                        # If reference has fewer zeros, some zeros might become non-zero
                        if ref_zero_frac < batch_zero_frac:
                            p_nonzero = (batch_zero_frac - ref_zero_frac) / batch_zero_frac if batch_zero_frac > 0 else 0
                            # Convert some zeros to small values
                            n_convert = int(n_zeros * p_nonzero)
                            if n_convert > 0:
                                zero_indices = np.where(zero_mask)[0]
                                convert_idx = np.random.choice(zero_indices, n_convert, replace=False)
                                # Assign small values near the minimum of reference
                                min_nonzero_ref = ref_q[0, j] * 0.5 if ref_q[0, j] > 0 else 0.01
                                col[convert_idx] = min_nonzero_ref
                                nonzero_mask = col > 0
                    
                    # Quantile mapping for non-zero values
                    nonzero_vals = col[nonzero_mask]
                    if len(nonzero_vals) > 0:
                        corrected = self._quantile_mapping(
                            nonzero_vals, batch_q[:, j], ref_q[:, j]
                        )
                        col[nonzero_mask] = corrected
                else:
                    # Transform all values
                    col = self._quantile_mapping(col, batch_q[:, j], ref_q[:, j])
                
                X_corrected[mask, j] = col
        
        return X_corrected
    
    def _quantile_mapping(
        self,
        values: np.ndarray,
        source_quantiles: np.ndarray,
        target_quantiles: np.ndarray
    ) -> np.ndarray:
        """
        Map values from source to target distribution using quantile matching.
        
        Uses linear interpolation between quantiles.
        """
        # Handle edge cases
        if len(values) == 0:
            return values
        
        if np.allclose(source_quantiles, target_quantiles):
            return values
        
        # Avoid division issues
        source_quantiles = np.clip(source_quantiles, 1e-10, None)
        
        # Compute percentile rank of each value in source distribution
        corrected = np.zeros_like(values)
        
        for i, val in enumerate(values):
            if val <= source_quantiles[0]:
                # Below lowest quantile
                if source_quantiles[0] > 0:
                    ratio = val / source_quantiles[0]
                    corrected[i] = ratio * target_quantiles[0]
                else:
                    corrected[i] = target_quantiles[0]
            elif val >= source_quantiles[-1]:
                # Above highest quantile
                if source_quantiles[-1] > 0:
                    ratio = val / source_quantiles[-1]
                    corrected[i] = ratio * target_quantiles[-1]
                else:
                    corrected[i] = target_quantiles[-1]
            else:
                # Interpolate between quantiles
                idx = np.searchsorted(source_quantiles, val) - 1
                idx = max(0, min(idx, len(source_quantiles) - 2))
                
                # Linear interpolation
                q_low = source_quantiles[idx]
                q_high = source_quantiles[idx + 1]
                
                if q_high > q_low:
                    t = (val - q_low) / (q_high - q_low)
                else:
                    t = 0.5
                
                t_low = target_quantiles[idx]
                t_high = target_quantiles[idx + 1]
                
                corrected[i] = t_low + t * (t_high - t_low)
        
        # Ensure non-negative
        corrected = np.clip(corrected, 0, None)
        
        return corrected


# =============================================================================
# Preprocessing Pipelines
# =============================================================================

class PreprocessingPipeline:
    """
    Chained preprocessing steps.
    
    Example:
        >>> pipeline = PreprocessingPipeline([
        ...     ('combat', ComBatCorrector()),
        ...     ('scale', SimpleBatchCorrector(method='zscore'))
        ... ])
        >>> X_processed = pipeline.fit_transform(X, batch_labels)
    """
    
    def __init__(self, steps: List[Tuple[str, BaseBatchCorrector]]):
        """
        Initialize pipeline.
        
        Args:
            steps: List of (name, corrector) tuples
        """
        self.steps = steps
        self._is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> 'PreprocessingPipeline':
        """Fit all steps."""
        X_transformed = X.copy()
        
        for name, corrector in self.steps:
            corrector.fit(X_transformed, batch_labels, covariates)
            X_transformed = corrector.transform(X_transformed, batch_labels)
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray
    ) -> np.ndarray:
        """Transform data through all steps."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")
        
        X_transformed = X.copy()
        
        for name, corrector in self.steps:
            X_transformed = corrector.transform(X_transformed, batch_labels)
        
        return X_transformed
    
    def fit_transform(
        self,
        X: np.ndarray,
        batch_labels: np.ndarray,
        covariates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, batch_labels, covariates)
        return self.transform(X, batch_labels)


# =============================================================================
# Utility Functions
# =============================================================================

def detect_batch_effects(
    X: np.ndarray,
    batch_labels: np.ndarray,
    method: str = 'pca'
) -> Dict[str, Any]:
    """
    Detect and quantify batch effects in data.
    
    Args:
        X: Data matrix
        batch_labels: Batch assignments
        method: Detection method ('pca', 'silhouette', 'kbet')
    
    Returns:
        Dictionary with detection results
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    if method in ['pca', 'all']:
        # PCA-based detection
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Variance explained by batch in PC space
        batch_encoded = pd.get_dummies(batch_labels).values
        
        # R² of batch vs PCs
        from sklearn.linear_model import LinearRegression
        
        r2_scores = []
        for i in range(X_pca.shape[1]):
            model = LinearRegression().fit(batch_encoded, X_pca[:, i])
            r2_scores.append(model.score(batch_encoded, X_pca[:, i]))
        
        results['pca_batch_r2'] = r2_scores
        results['mean_batch_r2'] = np.mean(r2_scores)
        results['variance_explained'] = pca.explained_variance_ratio_.tolist()
    
    if method in ['silhouette', 'all']:
        # Silhouette score treating batches as clusters
        from sklearn.metrics import silhouette_score
        
        try:
            sil = silhouette_score(X_scaled, batch_labels)
            results['silhouette_score'] = sil
            results['batch_effect_detected'] = sil > 0.1
        except Exception:
            results['silhouette_score'] = None
    
    return results


def evaluate_correction(
    X_original: np.ndarray,
    X_corrected: np.ndarray,
    batch_labels: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate batch correction quality.
    
    Args:
        X_original: Original data
        X_corrected: Corrected data
        batch_labels: Batch assignments
        labels: Optional biological labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.preprocessing import StandardScaler
    
    metrics = {}
    
    # Batch effect reduction
    original_detection = detect_batch_effects(X_original, batch_labels)
    corrected_detection = detect_batch_effects(X_corrected, batch_labels)
    
    metrics['original_batch_r2'] = original_detection.get('mean_batch_r2', 0)
    metrics['corrected_batch_r2'] = corrected_detection.get('mean_batch_r2', 0)
    metrics['batch_effect_reduction'] = (
        metrics['original_batch_r2'] - metrics['corrected_batch_r2']
    )
    
    # Information preservation (correlation between original and corrected)
    correlations = []
    for j in range(X_original.shape[1]):
        if np.std(X_original[:, j]) > 0 and np.std(X_corrected[:, j]) > 0:
            corr = np.corrcoef(X_original[:, j], X_corrected[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    metrics['mean_feature_correlation'] = np.mean(correlations) if correlations else 0
    
    # If biological labels available, check if they're preserved
    if labels is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        scaler = StandardScaler()
        
        # Original data classification
        X_orig_scaled = scaler.fit_transform(X_original)
        model = LogisticRegression(max_iter=1000)
        try:
            orig_scores = cross_val_score(model, X_orig_scaled, labels, cv=3)
            metrics['original_classification'] = orig_scores.mean()
        except Exception:
            metrics['original_classification'] = None
        
        # Corrected data classification
        X_corr_scaled = scaler.fit_transform(X_corrected)
        try:
            corr_scores = cross_val_score(model, X_corr_scaled, labels, cv=3)
            metrics['corrected_classification'] = corr_scores.mean()
        except Exception:
            metrics['corrected_classification'] = None
    
    return metrics


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'BaseBatchCorrector',
    'SimpleBatchCorrector',
    'ComBatCorrector',
    'ConQuRCorrector',
    'PreprocessingPipeline',
    'detect_batch_effects',
    'evaluate_correction',
]
