#!/usr/bin/env python3
"""
Microbiome Feature Engineering with VAEs
=========================================

This script demonstrates how to use the `q2_mechinterp` package to extract
interpretable features from microbiome data using Variational Autoencoders
and Sparse Autoencoders.

Usage:
    python microbiome_example.py

Requirements:
    pip install q2-mechinterp[all]
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# q2_mechinterp imports
from q2_mechinterp.microbiome import MicrobiomeVAE, MicrobiomeDataProcessor
from q2_mechinterp.training import VAETrainer
from q2_mechinterp.extraction import FeatureExtractor
from q2_mechinterp.core.utils import get_device, set_seed
from q2_mechinterp.core.logging import StructuredLogger, timer, print_model_summary
from q2_mechinterp.visualization import (
    plot_training_curves,
    plot_latent_space,
    plot_feature_importance,
    plot_reconstruction_comparison,
)
from q2_mechinterp.evaluation import FeatureEvaluator


def generate_synthetic_microbiome_data(
    n_samples: int = 200,
    n_features: int = 500,
    n_classes: int = 2,
    sparsity: float = 0.7,
    random_state: int = 42,
) -> tuple:
    """Generate synthetic microbiome-like data for demonstration."""
    np.random.seed(random_state)

    # Simulate sparse microbiome counts
    counts = np.random.lognormal(mean=2, sigma=2, size=(n_samples, n_features))
    counts = counts * (np.random.random((n_samples, n_features)) > sparsity)
    counts = counts.astype(int)

    # Create feature names
    genera = [
        "Bacteroides",
        "Prevotella",
        "Faecalibacterium",
        "Ruminococcus",
        "Blautia",
        "Lachnospira",
        "Roseburia",
        "Coprococcus",
    ]

    feature_names = []
    taxonomy_records = []

    for i in range(n_features):
        genus = np.random.choice(genera)
        feature_id = f"OTU_{i:04d}"
        taxonomy = (
            f"k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;"
            f"f__Lachnospiraceae;g__{genus};s__species_{i}"
        )
        feature_names.append(feature_id)
        taxonomy_records.append({"genome_id": feature_id, "taxonomy": taxonomy})

    # Create labels with signal
    labels = np.random.choice(["healthy", "disease"], n_samples)
    disease_features = np.random.choice(n_features, n_features // 10, replace=False)
    disease_mask = labels == "disease"
    counts[disease_mask][:, disease_features] *= 2

    # Create DataFrames
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    data_df = pd.DataFrame(counts, columns=feature_names, index=sample_ids)
    metadata_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": labels,
            "age": np.random.randint(20, 70, n_samples),
        }
    ).set_index("sample_id")
    taxonomy_df = pd.DataFrame(taxonomy_records)

    return data_df, metadata_df, taxonomy_df


def main():
    """Main function demonstrating the VAE feature extraction workflow."""

    # =========================================================================
    # Configuration
    # =========================================================================

    OUTPUT_DIR = Path("./microbiome_vae_outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    logger = StructuredLogger("microbiome_vae", log_dir=OUTPUT_DIR)

    # =========================================================================
    # 1. Load and Preprocess Data
    # =========================================================================

    print("\n" + "=" * 60)
    print("Step 1: Loading and Preprocessing Data")
    print("=" * 60)

    data_df, metadata_df, taxonomy_df = generate_synthetic_microbiome_data(
        n_samples=200, n_features=500, sparsity=0.7
    )

    print(f"Generated data shape: {data_df.shape}")
    print(f"Sparsity: {(data_df == 0).sum().sum() / data_df.size:.1%}")
    print(f"Label distribution:\n{metadata_df['group'].value_counts()}")

    # Initialize processor
    processor = MicrobiomeDataProcessor(data_paths={}, device=device)
    processor.load_data_from_dataframes(
        metag_df=data_df, metadata_df=metadata_df, taxonomy_df=taxonomy_df
    )

    # Apply log transformation
    print("Applying log transformation...")
    data_array = data_df.values.astype(float)
    data_array = np.log1p(data_array)
    data_array = data_array - data_array.mean(axis=1, keepdims=True)

    processor.table_metag_rclr = pd.DataFrame(
        data_array, index=data_df.index, columns=data_df.columns
    )
    processor.table_metag_filtered = processor.table_metag_rclr.copy()

    # Filter features
    processor.filter_features(min_prevalence=0.1, top_variance_pct=0.5)
    n_features = processor.table_metag_filtered.shape[1]
    print(f"Features after filtering: {n_features}")

    # Prepare datasets
    processor.prepare_datasets(label_column="group", test_size=0.2, stratify=True, random_state=42)

    # Get data and split
    X, y = processor.get_data_for_training(data_type="metag")

    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Create DataLoaders
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val)), batch_size=batch_size, shuffle=False
    )

    input_dim = X_train.shape[1]
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    logger.log_params(
        {
            "n_samples": len(data_df),
            "n_features_original": data_df.shape[1],
            "n_features_filtered": n_features,
        }
    )

    # =========================================================================
    # 2. Train Variational Autoencoder
    # =========================================================================

    print("\n" + "=" * 60)
    print("Step 2: Training Variational Autoencoder")
    print("=" * 60)

    processed_data = processor.metag_scaled
    if processed_data is None:
        processed_data = processor.table_metag_filtered.values

    vae = MicrobiomeVAE(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        dropout_rate=0.3,
        use_layer_norm=True,
    )

    print_model_summary(vae)

    trainer = VAETrainer(
        model=vae, device=device, learning_rate=1e-3, weight_decay=1e-5, gradient_clip=1.0
    )

    with timer("VAE Training"):
        train_losses, val_losses = trainer.train(
            train_loader,
            val_loader,
            epochs=100,
            beta=0.01,
            alpha=0.001,
            patience=15,
            verbose=10,
            save_path=None,
        )

    plot_training_curves(
        train_losses,
        val_losses,
        title="VAE Training Progress",
        save_path=OUTPUT_DIR / "training_curves.png",
    )

    torch.save(vae.state_dict(), OUTPUT_DIR / "microbiome_vae.pt")

    # =========================================================================
    # 3. Extract Latent Features
    # =========================================================================

    print("\n" + "=" * 60)
    print("Step 3: Extracting Latent Features")
    print("=" * 60)

    extractor = FeatureExtractor(vae, device)
    latent_features = extractor.get_latent_features(processed_data)
    print(f"Latent features shape: {latent_features.shape}")

    labels = metadata_df["group"].values
    label_encoder = {label: i for i, label in enumerate(np.unique(labels))}
    encoded_labels = np.array([label_encoder[l] for l in labels])

    plot_latent_space(
        latent_features,
        labels=encoded_labels,
        label_names=list(label_encoder.keys()),
        method="pca",
        title="VAE Latent Space",
        save_path=OUTPUT_DIR / "latent_space.png",
    )

    # =========================================================================
    # 4. Train Sparse Autoencoder
    # =========================================================================

    print("\n" + "=" * 60)
    print("Step 4: Training Sparse Autoencoder")
    print("=" * 60)

    with timer("Sparse AE Training"):
        extractor.train_sparse_autoencoder(
            latent_features,
            sparse_multiplier=3,
            l1_reg=0.01,
            epochs=100,
            lr=1e-3,
            batch_size=32,
            verbose=10,
        )

    sparse_features = extractor.get_sparse_features(latent_features)
    print(f"Sparse features shape: {sparse_features.shape}")
    print(f"Sparsity: {(sparse_features == 0).mean():.1%}")

    np.save(OUTPUT_DIR / "latent_features.npy", latent_features)
    np.save(OUTPUT_DIR / "sparse_features.npy", sparse_features)

    # =========================================================================
    # 5. Reconstruction Quality
    # =========================================================================

    print("\n" + "=" * 60)
    print("Step 5: Evaluating Reconstruction Quality")
    print("=" * 60)

    reconstructions = extractor.get_reconstruction(processed_data)

    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(processed_data, reconstructions)
    r2 = r2_score(processed_data.flatten(), reconstructions.flatten())

    print(f"Reconstruction MSE: {mse:.4f}")
    print(f"Reconstruction R²: {r2:.4f}")

    plot_reconstruction_comparison(
        original=processed_data,
        reconstructed=reconstructions,
        n_samples=3,
        n_features=30,
        title="Original vs Reconstructed Profiles",
        save_path=OUTPUT_DIR / "reconstruction_comparison.png",
    )

    # =========================================================================
    # 6. Downstream Task Evaluation
    # =========================================================================

    print("\n" + "=" * 60)
    print("Step 6: Evaluating Features on Downstream Classification")
    print("=" * 60)

    evaluator = FeatureEvaluator(task="classification", cv=5, scoring="accuracy", verbose=True)

    feature_sets = {
        "original": processed_data,
        "vae_latent": latent_features,
        "sparse_ae": sparse_features,
    }

    comparison_result = evaluator.compare_features(
        feature_sets=feature_sets,
        y=encoded_labels,
        models=["logistic", "rf", "knn"],
        scale_features=True,
    )

    print("\nFeature Comparison Results:")
    print(comparison_result.results_df.to_string())
    comparison_result.results_df.to_csv(OUTPUT_DIR / "feature_comparison.csv", index=False)

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR.absolute()}")

    results_df = comparison_result.results_df
    best_idx = results_df["mean_score"].idxmax()
    print(f"\nBest downstream performance:")
    print(f"  Feature set: {results_df.loc[best_idx, 'feature_set']}")
    print(f"  Model: {results_df.loc[best_idx, 'model_name']}")
    print(f"  Accuracy: {results_df.loc[best_idx, 'mean_score']:.2%}")


if __name__ == "__main__":
    main()
