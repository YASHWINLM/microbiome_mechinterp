"""
Data augmentation for VAE training.

This module provides domain-specific augmentation strategies:
- MicrobiomeAugmentor: Preserves sparsity and compositionality
- TranscriptomicsAugmentor: Expression-aware noise and dropout
- AugmentedDataLoader: On-the-fly augmentation during training

Example:
    >>> from q2_mechinterp.augmentation import (
    ...     MicrobiomeAugmentor,
    ...     AugmentedDataLoader,
    ...     ComposedAugmentor
    ... )
    >>>
    >>> # Single augmentor
    >>> augmentor = MicrobiomeAugmentor(random_state=42)
    >>> x_aug = augmentor.subsample(x, depth=0.8)
    >>>
    >>> # Augmented data loader for training
    >>> loader = AugmentedDataLoader(
    ...     data=X_train,
    ...     augmentor=augmentor,
    ...     augment_prob=0.5,
    ...     batch_size=32
    ... )
    >>>
    >>> # Composed augmentations
    >>> composed = ComposedAugmentor([
    ...     (augmentor.subsample, 0.3, {'depth': 0.8}),
    ...     (augmentor.multiplicative_noise, 0.5, {'noise_scale': 0.1}),
    ... ])
"""

from q2_mechinterp.augmentation.augmentation import (
    BaseAugmentor,
    MicrobiomeAugmentor,
    TranscriptomicsAugmentor,
    AugmentedDataset,
    AugmentedDataLoader,
    ComposedAugmentor,
)

__all__ = [
    "BaseAugmentor",
    "MicrobiomeAugmentor",
    "TranscriptomicsAugmentor",
    "AugmentedDataset",
    "AugmentedDataLoader",
    "ComposedAugmentor",
]