"""
Visualization utilities for VAE analysis.
"""

from q2_mechinterp.visualization.plots import (
    plot_training_curves,
    plot_latent_space,
    plot_feature_importance,
    plot_sparse_analysis,
    plot_reconstruction_comparison,
    plot_loss_components,
    plot_latent_correlation,
    plot_activation_histogram,
    plot_feature_activation_per_sample,
    plot_taxonomy_importance,
    plot_elbow,
    plot_sample_distances,
)

__all__ = [
    "plot_training_curves",
    "plot_latent_space",
    "plot_feature_importance",
    "plot_sparse_analysis",
    "plot_reconstruction_comparison",
    "plot_loss_components",
    "plot_latent_correlation",
    "plot_activation_histogram",
    "plot_feature_activation_per_sample",
    "plot_taxonomy_importance",
    "plot_elbow",
    "plot_sample_distances",
]
