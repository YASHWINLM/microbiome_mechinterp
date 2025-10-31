# Microbiome Mechinterp

Advanced microbiome analysis using Variational Autoencoders (VAE) and Sparse Autoencoders (SAE) for feature engineering and mechanistic interpretability.

## Features

- **VAE-based feature extraction** from microbiome data
- **Sparse Autoencoders** for interpretable feature learning
- **OTU/Gene mapping** - trace sparse features back to original biological features
- **QIIME2 integration** for sample classification
- **Comprehensive visualization** with UMAP embeddings
- **Taxonomy-aware analysis** with species/genus level insights

## Installation

```bash
# Create conda environment with QIIME2
conda create -n qiime2-microbiome -c qiime2 -c conda-forge python=3.9 qiime2 gemelli
conda activate qiime2-microbiome

# Install PyTorch
conda install pytorch -c pytorch

# Install package
pip install -e .
```

## Quick Start

### Command Line

```bash
microbiome-mechinterp \
    --metag-biom data/metaG/table.biom \
    --metat-biom data/metaT/table.biom \
    --metadata data/metadata.txt \
    --lineages data/lineages.txt \
    --latent-dim 50 \
    --epochs 100
```

### Python API

```python
from microbiome_mechinterp import (
    MicrobiomeDataProcessor, VAE, MicrobiomeVAETrainer,
    FeatureExtractor, Visualizer, map_sparse_features_to_otus
)
from microbiome_mechinterp.utils import get_device

# Setup
device = get_device()
data_paths = {
    'metag_biom': 'data/metaG/table.biom',
    'metat_biom': 'data/metaT/table.biom',
    'metadata': 'data/metadata.txt',
    'lineages': 'data/lineages.txt'
}

# Process data
processor = MicrobiomeDataProcessor(data_paths, device=device)
processor.load_data()
processor.apply_rclr_transformation()
processor.filter_features()
processor.prepare_datasets()

# Train VAE
vae = VAE(input_dim=processor.metag_scaled.shape[1], latent_dim=50).to(device)
trainer = MicrobiomeVAETrainer(vae, device=device)
trainer.train(processor.metag_scaled, epochs=100)

# Extract and analyze features
extractor = FeatureExtractor(vae, device)
latent = extractor.get_latent_features(processor.metag_scaled)
extractor.train_sparse_autoencoder(latent, sparse_multiplier=3)
sparse_features = extractor.get_sparse_features(latent)

# Map back to OTUs
mapping_results = map_sparse_features_to_otus(
    extractor, sparse_features, processor.metag_data.columns, "MetaG"
)

# Visualize
viz = Visualizer(save_dir='figs')
viz.visualize_latent_space(latent, processor.encoded_labels, processor.label_names)
viz.visualize_sparse_features(sparse_features, processor.encoded_labels, processor.label_names)
```

## Output Files

The package generates several output files:

### CSV Files

- `sparse_importance_*.csv` - OTU importance scores from sparse features
- `sparse_importance_with_taxonomy_*.csv` - Enhanced with taxonomy information

### Visualizations (in `figs/` directory)

- `umap_*.png` - UMAP embeddings of latent and sparse features
- `sparse_mapping_analysis_*.png` - Feature mapping visualizations
- `training_*.png` - Training curves for VAE and Sparse AE

### QIIME2 Artifacts (in `qiime2_results/` directory)

- `q2_accuracy_*.qzv` - Classification accuracy results
- `q2_predictions_*.qzv` - Sample predictions
- `q2_feature_importance_*.qza` - Feature importance rankings
- `q2_heatmap_*.qzv` - Feature heatmaps

## Example Analysis

See `examples/ibd_analysis.py` for a complete IBD classification pipeline.

## Citation

## License

MIT License
