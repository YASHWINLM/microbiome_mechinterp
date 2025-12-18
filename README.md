# Q2-Mechinterp

A Python package for VAE-based feature engineering, primarily designed for microbiome (metagenomics/metatranscriptomics) data with additional support for transcriptomics (RNA-seq) data.

## Features

- **Variational Autoencoders (VAEs)** for dimensionality reduction and latent feature learning
- **Sparse Autoencoders** for interpretable feature extraction from VAE latent spaces
- **Microbiome-specific processing**: BIOM table support, RCLR transformation, taxonomy integration
- **Transcriptomics support**: Gene expression processing, conditional VAEs
- **Multi-omics integration**: Combined metagenomics + metatranscriptomics analysis
- **Feature importance analysis**: Map sparse features back to original features
- **Comprehensive visualization**: Training curves, latent spaces, feature importance

## Installation

### Basic Installation

```bash
pip install q2-mechinterp
```

### With Microbiome Support

```bash
pip install q2-mechinterp[microbiome]
```

### With Visualization

```bash
pip install q2-mechinterp[visualization]
```

### Full Installation

```bash
pip install q2-mechinterp[full]
```

### Development Installation

```bash
git clone https://github.com/l1joseph/microbiome_mechinterp.git
cd microbiome_mechinterp/temp2/q2-mechinterp
pip install -e ".[dev,full]"
```

## Quick Start

### Microbiome Data Analysis

```python
from q2_mechinterp.microbiome import MicrobiomeVAE, MicrobiomeDataProcessor
from q2_mechinterp.training import VAETrainer
from q2_mechinterp.extraction import FeatureExtractor
from q2_mechinterp.core.utils import get_device

# Setup
device = get_device()

# Load and process data
processor = MicrobiomeDataProcessor(
    metagenomics_path="metagenomics.biom",
    metatranscriptomics_path="metatranscriptomics.biom",  # optional
    metadata_path="metadata.csv"
)
processor.load_data()
processor.apply_rclr_transformation()
processor.filter_features(prevalence_threshold=0.1, variance_percentile=10)
train_loader, val_loader, test_loader = processor.prepare_datasets(
    batch_size=32,
    test_size=0.2,
    val_size=0.1
)

# Create and train VAE
vae = MicrobiomeVAE(
    input_dim=processor.n_features,
    latent_dim=50,
    hidden_dims=[256, 128],
    dropout=0.2,
    beta=0.01  # Lower beta for sparse microbiome data
)

trainer = VAETrainer(
    model=vae,
    device=device,
    learning_rate=1e-3,
    weight_decay=1e-5
)

train_losses, val_losses = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping_patience=10
)

# Extract features
extractor = FeatureExtractor(vae, device)

# Get latent features
latent_features = extractor.get_latent_features(processor.data)

# Train sparse autoencoder for interpretable features
extractor.train_sparse_autoencoder(
    latent_features,
    expansion_factor=3,
    l1_lambda=0.001,
    epochs=50
)

# Get sparse features
sparse_features = extractor.get_sparse_features(latent_features)

# Analyze feature importance
importance_df = extractor.analyze_feature_importance(
    feature_names=processor.feature_names,
    taxonomy_df=processor.taxonomy_df  # optional
)
```

### Transcriptomics Data Analysis

```python
from q2_mechinterp.transcriptomics import TranscriptomicsVAE, TranscriptomicsDataProcessor
from q2_mechinterp.training import VAETrainer
from q2_mechinterp.extraction import FeatureExtractor

# Load and process data
processor = TranscriptomicsDataProcessor()
processor.load_expression_data("gene_counts.csv")
processor.filter_genes(
    min_counts=10,
    min_samples_pct=0.1,
    top_pct=0.02  # Keep top 2% most variable genes
)
processor.normalize(method="log1p")
train_loader, val_loader, test_loader = processor.prepare_datasets(batch_size=32)

# Create VAE (with optional conditioning on sample labels)
vae = TranscriptomicsVAE(
    input_dim=processor.n_genes,
    latent_dim=50,
    hidden_dims=[512, 256],
    beta=1.0
)

# Training and feature extraction same as microbiome...
```

### Multi-Omics Analysis

```python
from q2_mechinterp.microbiome import MultiOmicsVAE

# For combined metagenomics + metatranscriptomics
vae = MultiOmicsVAE(
    mg_input_dim=500,   # metagenomics features
    mt_input_dim=500,   # metatranscriptomics features
    latent_dim=50,
    hidden_dims=[256, 128]
)

# Train with paired data
trainer = VAETrainer(vae, device)
# ... training code
```

## Package Structure

```
q2_mechinterp/
├── __init__.py              # Main package exports
├── core/                    # Shared components
│   ├── vae.py              # BaseVAE, ConcreteVAE
│   ├── sparse.py           # Sparse autoencoder variants
│   └── utils.py            # Utilities (device, seeding, etc.)
├── microbiome/             # Microbiome-specific
│   ├── vae.py              # MicrobiomeVAE, MultiOmicsVAE
│   ├── processor.py        # Data loading and preprocessing
│   └── taxonomy.py         # Taxonomy utilities
├── transcriptomics/        # Transcriptomics-specific
│   ├── vae.py              # TranscriptomicsVAE
│   └── processor.py        # Gene expression processing
├── training/               # Training utilities
│   └── trainer.py          # VAETrainer with early stopping
├── extraction/             # Feature extraction
│   └── extractor.py        # FeatureExtractor, sparse features
└── visualization/          # Plotting utilities
    └── plots.py            # Training curves, latent space, etc.
```

## Model Architectures

### MicrobiomeVAE

- Designed for sparse, compositional microbiome data
- Lower default beta (0.01) to prevent posterior collapse
- RCLR transformation support
- Taxonomy-aware feature importance

### TranscriptomicsVAE

- Optimized for gene expression data
- Residual connections for better gradient flow
- Support for conditional generation (sample labels)
- Higher default beta (1.0)

### Sparse Autoencoders

Three variants for interpretable feature extraction:

- `SparseAutoencoder`: L1 regularization
- `TopKSparseAutoencoder`: Hard top-k activation constraint
- `DeepSparseAutoencoder`: Multi-layer hierarchical features

## Visualization

```python
from q2_mechinterp.visualization import (
    plot_training_curves,
    plot_latent_space,
    plot_feature_importance,
    plot_sparse_analysis,
    plot_reconstruction_comparison
)

# Training progress
plot_training_curves(train_losses, val_losses)

# Latent space visualization
plot_latent_space(latent_features, labels=sample_labels, method='umap')

# Feature importance
plot_feature_importance(importance_df, top_n=20)

# Sparse feature analysis
plot_sparse_analysis(sparse_features)
```

## Key Parameters

### VAE Training

- `latent_dim`: Dimension of latent space (typically 20-100)
- `beta`: KL divergence weight (0.01-1.0, lower for sparse data)
- `hidden_dims`: Encoder/decoder layer sizes
- `dropout`: Regularization (0.1-0.3)

### Sparse Autoencoder

- `expansion_factor`: Ratio of sparse to latent features (2-5x)
- `l1_lambda`: Sparsity penalty (0.0001-0.01)
- `top_k`: For TopK variant, number of active features

### Training

- `learning_rate`: 1e-4 to 1e-3
- `early_stopping_patience`: Epochs to wait (10-20)
- `gradient_clip`: Max gradient norm (1.0)

## Requirements

**Core:**

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, Pandas, scikit-learn, SciPy

**Microbiome (optional):**

- biom-format

**Visualization (optional):**

- Matplotlib, Seaborn, UMAP

## Citation

If you use this package in your research, please cite:

```bibtex
@software{q2_mechinterp,
  author = {Joseph, Leo},
  title = {Q2 Mechinterp: VAE-based Feature Engineering for Microbiome and Transcriptomics Data},
  year = {2024},
  url = {https://github.com/l1joseph/microbiome_mechinterp}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
