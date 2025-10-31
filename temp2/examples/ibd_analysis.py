#!/usr/bin/env python3
"""
Complete IBD analysis example using microbiome-mechinterp package
"""
from microbiome_mechinterp import (
    MicrobiomeDataProcessor,
    VAE,
    MicrobiomeVAETrainer,
    FeatureExtractor,
    Visualizer,
    FeatureClassifier,
    map_sparse_features_to_otus,
)
from microbiome_mechinterp.utils import get_device
import numpy as np

# Setup
data_paths = {
    "metag_biom": "../data/metaG/222424_72996_analysis_Metagenomic_Woltkav014DatabasescratchqpwoltkaWoLr2WoLr2BIOMnonebiom.biom",
    "metat_biom": "../data/metaT/222425_72996_analysis_Metatranscriptomic_Woltkav014DatabasescratchqpwoltkaWoLr2WoLr2BIOMnonebiom.biom",
    "metadata": "../data/metadata/72996_72996_analysis_mapping.txt",
    "lineages": "../data/lineages.txt",
}

device = get_device()

# 1. Load and process data
print("\n" + "=" * 60)
print("STEP 1: LOADING AND PROCESSING DATA")
print("=" * 60)
processor = MicrobiomeDataProcessor(data_paths, device=device)
processor.load_data()
processor.apply_rclr_transformation()
processor.filter_features()
processor.prepare_datasets()

# 2. Train VAE on MetaG data
print("\n" + "=" * 60)
print("STEP 2: TRAINING VAE")
print("=" * 60)
vae_metag = VAE(input_dim=processor.metag_scaled.shape[1], latent_dim=50).to(device)
trainer_metag = MicrobiomeVAETrainer(vae_metag, device=device)
train_losses, val_losses, best_epoch = trainer_metag.train(
    data=processor.metag_scaled, epochs=100, patience=10
)

# 3. Extract features
print("\n" + "=" * 60)
print("STEP 3: EXTRACTING FEATURES")
print("=" * 60)
extractor_metag = FeatureExtractor(vae_metag, device)
latent_metag = extractor_metag.get_latent_features(processor.metag_scaled)
sae_losses = extractor_metag.train_sparse_autoencoder(latent_metag, sparse_multiplier=3)
sparse_features_metag = extractor_metag.get_sparse_features(latent_metag)

# 4. Map sparse features to OTUs
print("\n" + "=" * 60)
print("STEP 4: MAPPING SPARSE FEATURES TO OTUs")
print("=" * 60)
mapping_results = map_sparse_features_to_otus(
    extractor_metag,
    sparse_features_metag,
    processor.metag_data.columns,
    approach_name="MetaG",
)

# 5. Visualize
print("\n" + "=" * 60)
print("STEP 5: VISUALIZATION")
print("=" * 60)
viz = Visualizer(save_dir="figs")
viz.visualize_latent_space(
    latent_metag, processor.encoded_labels, processor.label_names, "MetaG Latent"
)
viz.visualize_sparse_features(
    sparse_features_metag,
    processor.encoded_labels,
    processor.label_names,
    "MetaG Sparse",
)
viz.plot_training_curves(train_losses, title="VAE Training MetaG")
viz.plot_training_curves(sae_losses, title="Sparse AE Training MetaG")

# 6. Evaluate with Random Forest
print("\n" + "=" * 60)
print("STEP 6: CLASSIFICATION EVALUATION")
print("=" * 60)
classifier = FeatureClassifier()
accuracy, f1, rf = classifier.evaluate_features(
    sparse_features_metag, processor.encoded_labels, n_folds=5
)

print("\n" + "=" * 60)
print("IBD ANALYSIS COMPLETE!")
print("=" * 60)
print(f"Final Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Sparsity: {mapping_results['sparsity']:.2f}%")
print(f"\nOutput files generated:")
print(f"  - sparse_importance_metag.csv")
print(f"  - figs/umap_metag_latent.png")
print(f"  - figs/sparse_umap_metag_sparse.png")
print(f"  - figs/training_vae_training_metag.png")
print(f"  - figs/training_sparse_ae_training_metag.png")
print(f"  - figs/sparse_mapping_analysis_metag.png")
