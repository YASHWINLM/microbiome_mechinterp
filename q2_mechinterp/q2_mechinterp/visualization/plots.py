"""
Visualization utilities for VAE analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Any
import warnings


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch (optional).
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', alpha=0.7)
    
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(epochs, val_losses, label='Validation Loss', alpha=0.7)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_latent_space(
    latent_features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    method: str = 'tsne',
    title: str = "Latent Space Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Visualize latent space using dimensionality reduction.
    
    Args:
        latent_features: Latent feature matrix (samples x latent_dim).
        labels: Sample labels for coloring (optional).
        label_names: Names for label categories.
        method: Reduction method ('tsne', 'umap', 'pca').
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
        **kwargs: Additional arguments for reduction method.
    
    Returns:
        Matplotlib figure.
    """
    # Reduce to 2D
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            random_state=kwargs.get('random_state', 42),
            perplexity=kwargs.get('perplexity', 30)
        )
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                random_state=kwargs.get('random_state', 42),
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1)
            )
        except ImportError:
            warnings.warn("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    coords = reducer.fit_transform(latent_features)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names[label] if label_names is not None else f"Class {label}"
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[i]], label=name, alpha=0.7, s=30
            )
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30)
    
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_feature_importance(
    importance_df,
    top_n: int = 20,
    title: str = "Top Feature Importance",
    figsize: Tuple[int, int] = (12, 8),
    feature_col: str = 'Feature',
    importance_col: str = 'Importance',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot top feature importance scores.
    
    Args:
        importance_df: DataFrame with feature importance.
        top_n: Number of top features to plot.
        title: Plot title.
        figsize: Figure size.
        feature_col: Column name for features.
        importance_col: Column name for importance scores.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n)
    
    # Truncate long feature names
    feature_names = [
        f"{name[:30]}..." if len(str(name)) > 30 else str(name)
        for name in top_features[feature_col]
    ]
    
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, top_features[importance_col], alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_sparse_analysis(
    mapping_results: Dict[str, Any],
    approach_name: str = "Features",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize sparse feature mapping analysis.
    
    Args:
        mapping_results: Results from FeatureExtractor.map_sparse_to_original().
        approach_name: Name for the approach (e.g., "MetaG", "MetaT").
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Feature activation distribution
    activations = mapping_results['feature_activations']
    axes[0, 0].hist(activations, bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title(f'{approach_name} Feature Activation Distribution')
    axes[0, 0].set_xlabel('Mean Activation')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Top important features
    importance_df = mapping_results['sparse_importance_df']
    feature_col = 'Feature' if 'Feature' in importance_df.columns else importance_df.columns[0]
    importance_col = 'Importance' if 'Importance' in importance_df.columns else importance_df.columns[1]
    
    top_20 = importance_df.head(20)
    feature_names = [
        f"{str(name)[:15]}..." if len(str(name)) > 15 else str(name)
        for name in top_20[feature_col]
    ]
    
    axes[0, 1].barh(range(len(top_20)), top_20[importance_col])
    axes[0, 1].set_yticks(range(len(top_20)))
    axes[0, 1].set_yticklabels(feature_names)
    axes[0, 1].set_title(f'Top 20 Most Important {approach_name} Features')
    axes[0, 1].set_xlabel('Importance Score')
    
    # 3. Sparsity pie chart
    sparsity = mapping_results['sparsity']
    axes[1, 0].pie(
        [sparsity, 100 - sparsity],
        labels=[f'Zero ({sparsity:.1f}%)', f'Non-zero ({100-sparsity:.1f}%)'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightcoral', 'lightgreen']
    )
    axes[1, 0].set_title(f'{approach_name} Feature Sparsity')
    
    # 4. Importance distribution (log scale)
    importance_values = importance_df[importance_col]
    axes[1, 1].hist(importance_values, bins=50, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title(f'{approach_name} Feature Importance Distribution')
    axes[1, 1].set_xlabel('Importance Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 5,
    n_features: int = 50,
    title: str = "Original vs Reconstructed",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare original and reconstructed samples.
    
    Args:
        original: Original data matrix.
        reconstructed: Reconstructed data matrix.
        feature_names: Feature names (optional).
        n_samples: Number of samples to compare.
        n_features: Number of features to show.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
    
    # Select random samples
    sample_idx = np.random.choice(len(original), min(n_samples, len(original)), replace=False)
    
    # Select features with highest variance for visualization
    feature_var = np.var(original, axis=0)
    top_feature_idx = np.argsort(feature_var)[-n_features:]
    
    for i, idx in enumerate(sample_idx):
        # Original
        axes[i, 0].bar(range(n_features), original[idx, top_feature_idx], alpha=0.7)
        axes[i, 0].set_title(f'Sample {idx} - Original')
        if i == n_samples - 1:
            axes[i, 0].set_xlabel('Feature Index')
        
        # Reconstructed
        axes[i, 1].bar(range(n_features), reconstructed[idx, top_feature_idx], alpha=0.7)
        axes[i, 1].set_title(f'Sample {idx} - Reconstructed')
        if i == n_samples - 1:
            axes[i, 1].set_xlabel('Feature Index')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_loss_components(
    history: Dict[str, List[float]],
    title: str = "Loss Components",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot individual loss components over training.
    
    Args:
        history: Dictionary with 'recon_loss', 'kl_loss', 'l1_loss' lists.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    components = ['recon_loss', 'kl_loss', 'l1_loss']
    available = [c for c in components if c in history and len(history[c]) > 0]
    
    if len(available) == 0:
        print("No loss component history available")
        return None
    
    fig, axes = plt.subplots(1, len(available), figsize=figsize)
    if len(available) == 1:
        axes = [axes]
    
    titles = {
        'recon_loss': 'Reconstruction Loss',
        'kl_loss': 'KL Divergence',
        'l1_loss': 'L1 Regularization'
    }
    
    for i, component in enumerate(available):
        epochs = range(1, len(history[component]) + 1)
        axes[i].plot(epochs, history[component], alpha=0.7)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(titles.get(component, component))
        axes[i].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_latent_correlation(
    latent_features: np.ndarray,
    title: str = "Latent Feature Correlations",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation heatmap of latent features.
    
    Args:
        latent_features: Latent feature matrix (samples x latent_dim).
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap for heatmap.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    corr_matrix = np.corrcoef(latent_features.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')
    
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Latent Dimension')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_activation_histogram(
    sparse_features: np.ndarray,
    title: str = "Sparse Feature Activations",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of sparse feature activations.
    
    Args:
        sparse_features: Sparse feature matrix (samples x sparse_dim).
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # All activations
    all_activations = sparse_features.flatten()
    axes[0].hist(all_activations, bins=100, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Activation Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('All Activations')
    axes[0].set_yscale('log')
    
    # Non-zero activations only
    nonzero_activations = all_activations[all_activations != 0]
    if len(nonzero_activations) > 0:
        axes[1].hist(nonzero_activations, bins=50, alpha=0.7, color='coral')
        axes[1].set_xlabel('Activation Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Non-zero Activations Only')
    else:
        axes[1].text(0.5, 0.5, 'No non-zero activations', 
                     ha='center', va='center', transform=axes[1].transAxes)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_feature_activation_per_sample(
    sparse_features: np.ndarray,
    sample_labels: Optional[np.ndarray] = None,
    title: str = "Active Features per Sample",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot number of active (non-zero) features per sample.
    
    Args:
        sparse_features: Sparse feature matrix (samples x sparse_dim).
        sample_labels: Optional labels for coloring.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    active_counts = (sparse_features != 0).sum(axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if sample_labels is not None:
        unique_labels = np.unique(sample_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = sample_labels == label
            ax.scatter(np.where(mask)[0], active_counts[mask], 
                      c=[colors[i]], label=f'Class {label}', alpha=0.7)
        ax.legend()
    else:
        ax.scatter(range(len(active_counts)), active_counts, alpha=0.7)
    
    ax.axhline(y=np.mean(active_counts), color='red', linestyle='--', 
               label=f'Mean: {np.mean(active_counts):.1f}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Number of Active Features')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_taxonomy_importance(
    importance_df,
    level: str = 'Genus',
    top_n: int = 15,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance aggregated by taxonomic level.
    
    Args:
        importance_df: DataFrame with 'Feature', 'Importance', and taxonomy columns.
        level: Taxonomic level to aggregate by ('Genus', 'Family', 'Species').
        top_n: Number of top taxa to show.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    if level not in importance_df.columns:
        print(f"Warning: {level} column not found in importance_df")
        return None
    
    # Aggregate importance by taxonomic level
    importance_col = 'Importance' if 'Importance' in importance_df.columns else importance_df.columns[1]
    agg_importance = importance_df.groupby(level)[importance_col].sum().sort_values(ascending=False)
    
    # Remove unknown/unclassified
    agg_importance = agg_importance[~agg_importance.index.isin(['Unknown', 'Unclassified', ''])]
    
    top_taxa = agg_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_taxa))
    bars = ax.barh(y_pos, top_taxa.values, alpha=0.8)
    
    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(top_taxa)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_taxa.index)
    ax.invert_yaxis()
    ax.set_xlabel('Aggregated Importance Score')
    ax.set_title(title or f'Top {top_n} {level} by Importance')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_elbow(
    data: np.ndarray,
    max_components: int = 50,
    title: str = "Variance Explained by Components",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot elbow curve for choosing number of latent dimensions.
    
    Args:
        data: Input data matrix (samples x features).
        max_components: Maximum number of components to evaluate.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    from sklearn.decomposition import PCA
    
    n_components = min(max_components, min(data.shape) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Explained')
    axes[0].set_title('Individual Variance')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(range(1, n_components + 1), cumulative_variance, 'o-', alpha=0.7)
    axes[1].axhline(y=0.9, color='red', linestyle='--', label='90% variance')
    axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Find components for 90% and 95% variance
    n_90 = np.argmax(cumulative_variance >= 0.9) + 1
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Components for 90% variance: {n_90}")
    print(f"Components for 95% variance: {n_95}")
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_sample_distances(
    latent_features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    metric: str = 'euclidean',
    title: str = "Sample Distance Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot pairwise distance matrix between samples in latent space.
    
    Args:
        latent_features: Latent feature matrix (samples x latent_dim).
        labels: Sample labels for ordering (optional).
        metric: Distance metric ('euclidean', 'cosine', 'correlation').
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (optional).
    
    Returns:
        Matplotlib figure.
    """
    from scipy.spatial.distance import pdist, squareform
    
    distances = squareform(pdist(latent_features, metric=metric))
    
    # Optionally reorder by labels
    if labels is not None:
        order = np.argsort(labels)
        distances = distances[order][:, order]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(distances, cmap='viridis')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric.capitalize()} Distance')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig
