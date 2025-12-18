"""
Data augmentation for VAE training.

This module provides domain-specific augmentation strategies:
- Microbiome: Preserves sparsity and compositionality
- Transcriptomics: Expression-aware noise and dropout

Design Principles:
- Preserve data characteristics (sparsity, ranges)
- On-the-fly augmentation for training efficiency
- Composable augmentation strategies
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional, List, Callable, Tuple, Union
from dataclasses import dataclass
import warnings


# =============================================================================
# Base Augmentor
# =============================================================================


class BaseAugmentor:
    """
    Base class for data augmentors.

    Subclasses implement specific augmentation strategies.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentation."""
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        """Reset random state."""
        self.rng = np.random.RandomState(seed)


# =============================================================================
# Microbiome Augmentation
# =============================================================================


class MicrobiomeAugmentor(BaseAugmentor):
    """
    Augmentation strategies for microbiome data.

    Designed to preserve:
    - Sparsity patterns (many zeros)
    - Compositional structure (counts sum to total)
    - Non-negative values

    Example:
        >>> augmentor = MicrobiomeAugmentor()
        >>>
        >>> # Single augmentation
        >>> x_aug = augmentor.subsample(x, depth=0.8)
        >>>
        >>> # Combined augmentation
        >>> x_aug = augmentor.apply(
        ...     x,
        ...     methods=['subsample', 'multiplicative_noise'],
        ...     params={'depth': 0.8, 'noise_scale': 0.1}
        ... )
    """

    def __init__(self, random_state: Optional[int] = None, preserve_zeros: bool = True):
        """
        Initialize augmentor.

        Args:
            random_state: Random seed
            preserve_zeros: Whether to preserve zero values
        """
        super().__init__(random_state)
        self.preserve_zeros = preserve_zeros

    def subsample(self, x: np.ndarray, depth: float = 0.8) -> np.ndarray:
        """
        Rarefaction-style subsampling.

        Simulates sequencing at a lower depth while preserving
        relative abundances and sparsity.

        Args:
            x: Data matrix (samples x features) - can be counts or proportions
            depth: Fraction of original depth to sample (0-1)

        Returns:
            Subsampled data
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        for i in range(x.shape[0]):
            # Normalize to proportions
            row = x[i]
            total = row.sum()

            if total == 0:
                continue

            probs = row / total

            # Sample new total
            new_total = int(total * depth)
            if new_total == 0:
                continue

            # Multinomial sampling
            new_counts = self.rng.multinomial(new_total, probs)

            # Rescale back to original scale
            result[i] = new_counts * (total / new_total)

        return result.astype(np.float32)

    def multiplicative_noise(self, x: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Add multiplicative (lognormal) noise.

        Preserves zeros and relative magnitudes better than
        additive noise for sparse compositional data.

        Args:
            x: Data matrix
            noise_scale: Scale of lognormal noise (sigma)

        Returns:
            Augmented data
        """
        x = np.asarray(x, dtype=np.float64)

        # Generate lognormal multipliers
        noise = self.rng.lognormal(0, noise_scale, x.shape)

        result = x * noise

        # Preserve zeros if requested
        if self.preserve_zeros:
            result[x == 0] = 0

        return result.astype(np.float32)

    def dropout(self, x: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """
        Random feature dropout.

        Simulates technical dropouts (features below detection limit).

        Args:
            x: Data matrix
            dropout_rate: Fraction of non-zero values to drop

        Returns:
            Data with additional zeros
        """
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()

        # Only dropout non-zero values
        nonzero_mask = x != 0
        dropout_mask = self.rng.random(x.shape) < dropout_rate

        result[nonzero_mask & dropout_mask] = 0

        return result.astype(np.float32)

    def permute_within_sample(self, x: np.ndarray, permute_fraction: float = 0.1) -> np.ndarray:
        """
        Randomly permute a fraction of features within each sample.

        Args:
            x: Data matrix
            permute_fraction: Fraction of features to permute

        Returns:
            Data with permuted features
        """
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()

        n_features = x.shape[1]
        n_permute = int(n_features * permute_fraction)

        if n_permute < 2:
            return result.astype(np.float32)

        for i in range(x.shape[0]):
            # Select random features to permute
            perm_idx = self.rng.choice(n_features, n_permute, replace=False)
            # Shuffle their values
            result[i, perm_idx] = self.rng.permutation(result[i, perm_idx])

        return result.astype(np.float32)

    def __call__(self, x: np.ndarray, method: str = "multiplicative_noise", **kwargs) -> np.ndarray:
        """
        Apply a single augmentation.

        Args:
            x: Data matrix
            method: Augmentation method name
            **kwargs: Method-specific parameters

        Returns:
            Augmented data
        """
        methods = {
            "subsample": self.subsample,
            "multiplicative_noise": self.multiplicative_noise,
            "dropout": self.dropout,
            "permute": self.permute_within_sample,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

        return methods[method](x, **kwargs)

    def apply(self, x: np.ndarray, methods: List[str], params: Optional[dict] = None) -> np.ndarray:
        """
        Apply multiple augmentations in sequence.

        Args:
            x: Data matrix
            methods: List of method names
            params: Dict of method-specific parameters

        Returns:
            Augmented data
        """
        params = params or {}
        result = x.copy()

        for method in methods:
            method_params = {
                k.replace(f"{method}_", ""): v
                for k, v in params.items()
                if k.startswith(method) or k in ["depth", "noise_scale", "dropout_rate"]
            }
            result = self(result, method=method, **method_params)

        return result


# =============================================================================
# Transcriptomics Augmentation
# =============================================================================


class TranscriptomicsAugmentor(BaseAugmentor):
    """
    Augmentation strategies for gene expression data.

    Designed for RNA-seq or microarray data with:
    - Continuous expression values
    - Technical noise patterns
    - Gene-level characteristics

    Example:
        >>> augmentor = TranscriptomicsAugmentor()
        >>>
        >>> # Single augmentation
        >>> x_aug = augmentor.gaussian_noise(x, noise_level=0.1)
        >>>
        >>> # Mixup for regularization
        >>> x_mix, y_mix = augmentor.mixup(x1, y1, x2, y2, alpha=0.2)
    """

    def gaussian_noise(
        self, x: np.ndarray, noise_level: float = 0.1, per_feature: bool = True
    ) -> np.ndarray:
        """
        Add Gaussian noise.

        Args:
            x: Data matrix (samples x genes)
            noise_level: Noise scale relative to feature std
            per_feature: Scale noise by each feature's std

        Returns:
            Noisy data
        """
        x = np.asarray(x, dtype=np.float64)

        if per_feature:
            stds = np.std(x, axis=0, keepdims=True)
            stds[stds == 0] = 1.0
            noise = self.rng.normal(0, noise_level, x.shape) * stds
        else:
            noise = self.rng.normal(0, noise_level * np.std(x), x.shape)

        return (x + noise).astype(np.float32)

    def gene_dropout(
        self, x: np.ndarray, dropout_rate: float = 0.1, expression_dependent: bool = True
    ) -> np.ndarray:
        """
        Expression-dependent gene dropout.

        Lower expressed genes have higher dropout probability,
        simulating technical limitations.

        Args:
            x: Data matrix
            dropout_rate: Base dropout rate
            expression_dependent: Whether dropout depends on expression level

        Returns:
            Data with gene dropout
        """
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()

        if expression_dependent:
            # Compute dropout probabilities (higher for low expression)
            mean_expr = np.mean(x, axis=0, keepdims=True)
            max_expr = np.max(mean_expr)

            if max_expr > 0:
                # Inverse relationship: low expression = high dropout
                dropout_probs = dropout_rate * (1 - mean_expr / max_expr)
                dropout_probs = np.clip(dropout_probs, 0, 0.5)
            else:
                dropout_probs = np.full_like(x, dropout_rate)

            dropout_mask = self.rng.random(x.shape) < dropout_probs
        else:
            dropout_mask = self.rng.random(x.shape) < dropout_rate

        result[dropout_mask] = 0

        return result.astype(np.float32)

    def sample_scaling(
        self, x: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Random per-sample scaling.

        Simulates library size variation.

        Args:
            x: Data matrix
            scale_range: (min, max) scaling factors

        Returns:
            Scaled data
        """
        x = np.asarray(x, dtype=np.float64)

        scales = self.rng.uniform(scale_range[0], scale_range[1], (x.shape[0], 1))

        return (x * scales).astype(np.float32)

    def mixup(
        self, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation.

        Creates convex combinations of samples and their labels.

        Reference:
            Zhang et al. "mixup: Beyond Empirical Risk Minimization" ICLR 2018

        Args:
            x1, y1: First set of samples and labels
            x2, y2: Second set of samples and labels
            alpha: Beta distribution parameter

        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        # Sample mixing coefficient
        lam = self.rng.beta(alpha, alpha)

        # Mix features
        x_mixed = lam * x1 + (1 - lam) * x2

        # Mix labels (for soft labels)
        y_mixed = lam * y1 + (1 - lam) * y2

        return x_mixed.astype(np.float32), y_mixed.astype(np.float32)

    def cutout(self, x: np.ndarray, cutout_fraction: float = 0.1) -> np.ndarray:
        """
        Random gene cutout (masking).

        Sets random genes to zero for regularization.

        Args:
            x: Data matrix
            cutout_fraction: Fraction of genes to mask

        Returns:
            Data with masked genes
        """
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()

        n_cutout = int(x.shape[1] * cutout_fraction)
        cutout_idx = self.rng.choice(x.shape[1], n_cutout, replace=False)

        result[:, cutout_idx] = 0

        return result.astype(np.float32)

    def __call__(self, x: np.ndarray, method: str = "gaussian_noise", **kwargs) -> np.ndarray:
        """Apply a single augmentation."""
        methods = {
            "gaussian_noise": self.gaussian_noise,
            "gene_dropout": self.gene_dropout,
            "sample_scaling": self.sample_scaling,
            "cutout": self.cutout,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

        return methods[method](x, **kwargs)


# =============================================================================
# Augmented DataLoader
# =============================================================================


class AugmentedDataset(Dataset):
    """
    Dataset wrapper that applies augmentation on-the-fly.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        augmentor: Optional[BaseAugmentor] = None,
        augment_prob: float = 0.5,
        augment_method: str = "gaussian_noise",
        augment_kwargs: Optional[dict] = None,
    ):
        """
        Initialize dataset.

        Args:
            data: Feature matrix
            labels: Optional labels
            augmentor: Augmentor instance
            augment_prob: Probability of applying augmentation
            augment_method: Default augmentation method
            augment_kwargs: Arguments for augmentation
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = labels
        self.augmentor = augmentor
        self.augment_prob = augment_prob
        self.augment_method = augment_method
        self.augment_kwargs = augment_kwargs or {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        x = self.data[idx : idx + 1]  # Keep 2D for augmentor

        if self.augmentor is not None and np.random.random() < self.augment_prob:
            x = self.augmentor(x, method=self.augment_method, **self.augment_kwargs)

        x = torch.FloatTensor(x.squeeze())

        if self.labels is not None:
            y = self.labels[idx]
            if isinstance(y, np.ndarray):
                y = torch.LongTensor([y]).squeeze()
            else:
                y = torch.tensor(y)
            return x, y

        return x


class AugmentedDataLoader:
    """
    DataLoader wrapper that applies augmentation.

    Example:
        >>> augmentor = MicrobiomeAugmentor()
        >>> loader = AugmentedDataLoader(
        ...     data=X_train,
        ...     augmentor=augmentor,
        ...     augment_prob=0.5,
        ...     batch_size=32
        ... )
        >>>
        >>> for epoch in range(epochs):
        ...     for batch in loader:
        ...         # batch is augmented with prob 0.5
        ...         train_step(batch)
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        augmentor: Optional[BaseAugmentor] = None,
        augment_prob: float = 0.5,
        augment_method: str = "gaussian_noise",
        augment_kwargs: Optional[dict] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ):
        """
        Initialize augmented data loader.

        Args:
            data: Feature matrix
            labels: Optional labels
            augmentor: Augmentor instance
            augment_prob: Probability of augmentation
            augment_method: Augmentation method
            augment_kwargs: Augmentation parameters
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: DataLoader workers
        """
        dataset = AugmentedDataset(
            data=data,
            labels=labels,
            augmentor=augmentor,
            augment_prob=augment_prob,
            augment_method=augment_method,
            augment_kwargs=augment_kwargs,
        )

        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


# =============================================================================
# Composition
# =============================================================================


class ComposedAugmentor:
    """
    Compose multiple augmentations with individual probabilities.

    Example:
        >>> composed = ComposedAugmentor([
        ...     (MicrobiomeAugmentor().subsample, 0.3, {'depth': 0.8}),
        ...     (MicrobiomeAugmentor().multiplicative_noise, 0.5, {'noise_scale': 0.1}),
        ... ])
        >>> x_aug = composed(x)
    """

    def __init__(
        self, augmentations: List[Tuple[Callable, float, dict]], random_state: Optional[int] = None
    ):
        """
        Initialize composed augmentor.

        Args:
            augmentations: List of (function, probability, kwargs) tuples
            random_state: Random seed
        """
        self.augmentations = augmentations
        self.rng = np.random.RandomState(random_state)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentations with their probabilities."""
        result = x.copy()

        for func, prob, kwargs in self.augmentations:
            if self.rng.random() < prob:
                result = func(result, **kwargs)

        return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BaseAugmentor",
    "MicrobiomeAugmentor",
    "TranscriptomicsAugmentor",
    "AugmentedDataset",
    "AugmentedDataLoader",
    "ComposedAugmentor",
]
