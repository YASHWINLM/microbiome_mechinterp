#!/usr/bin/env python3
"""
Updated IBD Analysis with Data Cleaning for VAE Training
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from microbiome_mechinterp import (
    MicrobiomeDataProcessor,
    VAE,
    MicrobiomeVAETrainer,
    FeatureExtractor,
    Visualizer,
)
from microbiome_mechinterp.utils import get_device


# Data cleaning function
def clean_data_for_vae(data, max_value=10.0):
    """Clean data to prevent NaN issues"""
    data = data.copy()
    # Replace NaN/Inf
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    # Clip extreme values
    data = np.clip(data, -max_value, max_value)
    return data


# Setup
device = get_device()
data_paths = {
    "metag_biom": "../data/metaG/222424_72996_analysis_Metagenomic_Woltkav014DatabasescratchqpwoltkaWoLr2WoLr2BIOMnonebiom.biom",
    "metat_biom": "../data/metaT/222425_72996_analysis_Metatranscriptomic_Woltkav014DatabasescratchqpwoltkaWoLr2WoLr2BIOMnonebiom.biom",
    "metadata": "../data/metadata/72996_72996_analysis_mapping.txt",
    "lineages": "../data/lineages.txt",
}

print("=" * 60)
print("STEP 1: LOADING AND PROCESSING DATA")
print("=" * 60)

processor = MicrobiomeDataProcessor(data_paths, device=device)
processor.load_data()
processor.apply_rclr_transformation()
processor.filter_features()
processor.prepare_datasets()

print(f"\nFinal dataset shapes:")
print(f"  MetaG scaled: {processor.metag_scaled.shape}")
print(f"  MetaT scaled: {processor.metat_scaled.shape}")

# IMPORTANT: Clean data before VAE training
print("\n" + "=" * 60)
print("STEP 1.5: CLEANING DATA FOR VAE")
print("=" * 60)

print("\nBefore cleaning:")
print(
    f"  MetaG range: [{processor.metag_scaled.min():.2f}, {processor.metag_scaled.max():.2f}]"
)
print(f"  MetaG NaN count: {np.isnan(processor.metag_scaled).sum()}")
print(f"  MetaG Inf count: {np.isinf(processor.metag_scaled).sum()}")

metag_clean = clean_data_for_vae(processor.metag_scaled, max_value=10.0)
metat_clean = clean_data_for_vae(processor.metat_scaled, max_value=10.0)

print("\nAfter cleaning:")
print(f"  MetaG range: [{metag_clean.min():.2f}, {metag_clean.max():.2f}]")
print(f"  MetaG NaN count: {np.isnan(metag_clean).sum()}")
print(f"  MetaG Inf count: {np.isinf(metag_clean).sum()}")

# Split for validation
train_metag, val_metag = train_test_split(metag_clean, test_size=0.15, random_state=42)
train_metat, val_metat = train_test_split(metat_clean, test_size=0.15, random_state=42)

print(f"\nTrain/Val split:")
print(f"  MetaG train: {train_metag.shape}, val: {val_metag.shape}")
print(f"  MetaT train: {train_metat.shape}, val: {val_metat.shape}")

print("\n" + "=" * 60)
print("STEP 2: TRAINING VAE")
print("=" * 60)

# Train MetaG VAE
print("\n[MetaG VAE]")
vae_metag = VAE(input_dim=train_metag.shape[1], latent_dim=50).to(device)
trainer_metag = MicrobiomeVAETrainer(vae_metag, device=device, lr=1e-4)

train_losses_metag, val_losses_metag, best_epoch_metag = trainer_metag.train(
    data=train_metag,
    val_data=val_metag,
    epochs=100,
    batch_size=32,
    beta=0.01,  # Conservative
    alpha=0.001,  # Conservative
    noise_factor=0.01,
    patience=15,
    min_delta=0.0001,
)

print(f"\nMetaG VAE training completed")
print(f"  Best epoch: {best_epoch_metag}")
print(f"  Final train loss: {train_losses_metag[-1]:.6f}")
if val_losses_metag:
    print(f"  Final val loss: {val_losses_metag[-1]:.6f}")

# Train MetaT VAE
print("\n[MetaT VAE]")
vae_metat = VAE(input_dim=train_metat.shape[1], latent_dim=30).to(device)
trainer_metat = MicrobiomeVAETrainer(vae_metat, device=device, lr=1e-4)

train_losses_metat, val_losses_metat, best_epoch_metat = trainer_metat.train(
    data=train_metat,
    val_data=val_metat,
    epochs=100,
    batch_size=32,
    beta=0.01,
    alpha=0.001,
    noise_factor=0.01,
    patience=15,
)

print(f"\nMetaT VAE training completed")
print(f"  Best epoch: {best_epoch_metat}")

print("\n" + "=" * 60)
print("STEP 3: EXTRACTING FEATURES")
print("=" * 60)

# Extract latent features (use clean data)
extractor_metag = FeatureExtractor(vae_metag, device)
latent_metag = extractor_metag.get_latent_features(metag_clean)

print(f"\nLatent MetaG shape: {latent_metag.shape}")
print(f"  Range: [{latent_metag.min():.3f}, {latent_metag.max():.3f}]")
print(f"  NaN count: {np.isnan(latent_metag).sum()}")

if np.isnan(latent_metag).any() or np.isinf(latent_metag).any():
    print("\n⚠️  Warning: Latent features contain NaN/Inf!")
    print("  This means the VAE is still producing invalid outputs")
    print("  Suggestions:")
    print("  1. Try training on CPU instead of MPS")
    print("  2. Further reduce beta/alpha")
    print("  3. Increase data clipping (max_value=5)")
else:
    print("\n✓ Latent features are valid!")

    # Train sparse autoencoder
    print("\nTraining Sparse Autoencoder...")
    extractor_metag.train_sparse_autoencoder(
        latent_metag, sparse_multiplier=3, epochs=100, learning_rate=0.001, l1_reg=0.01
    )

    # Get sparse features
    sparse_features_metag = extractor_metag.get_sparse_features(latent_metag)
    print(f"Sparse features shape: {sparse_features_metag.shape}")

    # Continue with rest of pipeline...
    print("\n✓ Feature extraction completed!")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
