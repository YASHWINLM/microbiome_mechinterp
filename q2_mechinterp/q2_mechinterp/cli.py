"""
Command-line interface for VAE feature extraction.

Provides commands for:
- Training VAE models from config files
- Extracting features from trained models
- Evaluating feature quality

Usage:
    q2-mechinterp train config.yaml
    q2-mechinterp extract --model checkpoint.pt --data data.csv
    q2-mechinterp evaluate --features latent.npy --labels labels.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML, JSON, or TOML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = path.suffix.lower()

    if ext in [".yaml", ".yml"]:
        try:
            import yaml

            with open(path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")

    elif ext == ".json":
        import json

        with open(path, "r") as f:
            return json.load(f)

    elif ext == ".toml":
        try:
            import tomllib

            with open(path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import toml

                with open(path, "r") as f:
                    return toml.load(f)
            except ImportError:
                raise ImportError("toml package required. Install with: pip install toml")

    else:
        raise ValueError(f"Unknown config format: {ext}")


def cmd_train(args: argparse.Namespace) -> int:
    """Train VAE from configuration file."""
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from q2_mechinterp.core.protocols import PipelineConfig
    from q2_mechinterp.core.logging import StructuredLogger, timer

    print(f"Loading config from: {args.config}")
    config = PipelineConfig.load(args.config)

    # Setup
    device = config.get_device()
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Setup logging
    logger = StructuredLogger(
        config.experiment_name, log_dir=config.output_dir if args.log else None
    )
    logger.log_config(config)

    # Load data
    print(f"Loading data from: {args.data}")
    data_path = Path(args.data)

    if data_path.suffix == ".npy":
        data = np.load(data_path)
    elif data_path.suffix in [".csv", ".tsv"]:
        sep = "\t" if data_path.suffix == ".tsv" else ","
        df = pd.read_csv(data_path, sep=sep, index_col=0)
        data = df.values.astype(np.float32)
    else:
        raise ValueError(f"Unknown data format: {data_path.suffix}")

    print(f"Data shape: {data.shape}")

    # Update input_dim in config
    config.vae.input_dim = data.shape[1]

    # Split data
    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(
        data, test_size=config.data.val_size + config.data.test_size, random_state=config.seed
    )

    if config.data.test_size > 0:
        val_data, test_data = train_test_split(
            val_data,
            test_size=config.data.test_size / (config.data.val_size + config.data.test_size),
            random_state=config.seed,
        )
    else:
        test_data = None

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_data)),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_data)),
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    # Create model
    print("Creating model...")

    # Determine model class based on config or default
    model_type = getattr(config, "model_type", "base")

    if model_type == "microbiome":
        from q2_mechinterp.microbiome import MicrobiomeVAE

        model = MicrobiomeVAE(
            input_dim=config.vae.input_dim,
            latent_dim=config.vae.latent_dim,
            hidden_dims=config.vae.hidden_dims,
            dropout_rate=config.vae.dropout_rate,
        ).to(device)
    elif model_type == "transcriptomics":
        from q2_mechinterp.transcriptomics import TranscriptomicsVAE

        model = TranscriptomicsVAE(
            input_dim=config.vae.input_dim,
            latent_dim=config.vae.latent_dim,
            hidden_dims=config.vae.hidden_dims,
            dropout_rate=config.vae.dropout_rate,
        ).to(device)
    else:
        from q2_mechinterp.core import ConcreteVAE

        model = ConcreteVAE(
            input_dim=config.vae.input_dim,
            latent_dim=config.vae.latent_dim,
            hidden_dims=config.vae.hidden_dims,
            dropout_rate=config.vae.dropout_rate,
        ).to(device)

    print(f"Model: {model.__class__.__name__}")
    print(f"  Input dim: {config.vae.input_dim}")
    print(f"  Latent dim: {config.vae.latent_dim}")
    print(f"  Hidden dims: {config.vae.hidden_dims}")

    # Train
    from q2_mechinterp.training import VAETrainer

    trainer = VAETrainer(model, device, lr=config.training.learning_rate)

    print("\nTraining...")
    with timer("Training", logger):
        train_losses, val_losses = trainer.train(
            train_loader,
            val_loader,
            epochs=config.training.epochs,
            beta=config.training.beta,
            alpha=config.training.alpha,
            patience=config.training.early_stopping_patience,
            verbose=10,
        )

    # Save model
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{config.experiment_name}_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        checkpoint_path,
    )
    print(f"Model saved to: {checkpoint_path}")

    # Save training history
    logger.save_history()

    print("\nTraining complete!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Best val loss: {min(val_losses):.4f}")

    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract features from trained model."""
    import numpy as np
    import pandas as pd
    import torch

    from q2_mechinterp.core.protocols import PipelineConfig
    from q2_mechinterp.extraction import FeatureExtractor

    # Load checkpoint
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu")

    config_dict = checkpoint.get("config", {})

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Recreate model
    vae_config = config_dict.get("vae", {})
    model_type = config_dict.get("model_type", "base")

    if model_type == "microbiome":
        from q2_mechinterp.microbiome import MicrobiomeVAE

        model = MicrobiomeVAE(
            input_dim=vae_config["input_dim"],
            latent_dim=vae_config.get("latent_dim", 50),
            hidden_dims=vae_config.get("hidden_dims", [256, 128]),
        )
    elif model_type == "transcriptomics":
        from q2_mechinterp.transcriptomics import TranscriptomicsVAE

        model = TranscriptomicsVAE(
            input_dim=vae_config["input_dim"],
            latent_dim=vae_config.get("latent_dim", 50),
            hidden_dims=vae_config.get("hidden_dims", [512, 256]),
        )
    else:
        from q2_mechinterp.core import ConcreteVAE

        model = ConcreteVAE(
            input_dim=vae_config["input_dim"],
            latent_dim=vae_config.get("latent_dim", 50),
            hidden_dims=vae_config.get("hidden_dims", [256, 128]),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from: {args.data}")
    data_path = Path(args.data)

    if data_path.suffix == ".npy":
        data = np.load(data_path)
        sample_names = None
    elif data_path.suffix in [".csv", ".tsv"]:
        sep = "\t" if data_path.suffix == ".tsv" else ","
        df = pd.read_csv(data_path, sep=sep, index_col=0)
        data = df.values.astype(np.float32)
        sample_names = list(df.index)
    else:
        raise ValueError(f"Unknown format: {data_path.suffix}")

    print(f"Data shape: {data.shape}")

    # Extract features
    extractor = FeatureExtractor(model, device)

    print("Extracting latent features...")
    latent_features = extractor.get_latent_features(data)
    print(f"Latent features shape: {latent_features.shape}")

    # Optionally extract sparse features
    sparse_features = None
    if args.sparse:
        print("Training sparse autoencoder...")
        sparse_ae, sparse_features = extractor.train_sparse_ae(
            latent_features, expansion_factor=args.expansion_factor, epochs=args.sparse_epochs
        )
        print(f"Sparse features shape: {sparse_features.shape}")

    # Save features
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as numpy
    np.save(output_dir / "latent_features.npy", latent_features)
    print(f"Saved latent features to: {output_dir / 'latent_features.npy'}")

    if sparse_features is not None:
        np.save(output_dir / "sparse_features.npy", sparse_features)
        print(f"Saved sparse features to: {output_dir / 'sparse_features.npy'}")

    # Optionally save as CSV with sample names
    if sample_names is not None and args.csv:
        latent_df = pd.DataFrame(
            latent_features,
            index=sample_names,
            columns=[f"latent_{i}" for i in range(latent_features.shape[1])],
        )
        latent_df.to_csv(output_dir / "latent_features.csv")
        print(f"Saved latent CSV to: {output_dir / 'latent_features.csv'}")

    print("\nExtraction complete!")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate feature quality."""
    import numpy as np
    import pandas as pd

    from q2_mechinterp.evaluation import FeatureEvaluator

    # Load features
    print(f"Loading features from: {args.features}")
    features_path = Path(args.features)

    if features_path.suffix == ".npy":
        features = np.load(features_path)
    elif features_path.suffix == ".csv":
        df = pd.read_csv(features_path, index_col=0)
        features = df.values
    else:
        raise ValueError(f"Unknown format: {features_path.suffix}")

    print(f"Features shape: {features.shape}")

    # Load labels
    print(f"Loading labels from: {args.labels}")
    labels_path = Path(args.labels)

    if labels_path.suffix == ".npy":
        labels = np.load(labels_path)
    elif labels_path.suffix == ".csv":
        df = pd.read_csv(labels_path, index_col=0)
        labels = df.iloc[:, 0].values
    else:
        raise ValueError(f"Unknown format: {labels_path.suffix}")

    # Determine task type
    n_unique = len(np.unique(labels))
    task = "classification" if n_unique < 20 else "regression"
    print(f"Task: {task} ({n_unique} unique values)")

    # Evaluate
    evaluator = FeatureEvaluator(task=task, cv=args.cv)

    models = args.models.split(",") if args.models else ["logistic", "rf"]

    print("\nEvaluating features...")
    results = evaluator.compare_features(
        feature_sets={"input_features": features}, y=labels, models=models
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(results.results_df.to_string())

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.results_df.to_csv(output_path)
        print(f"\nResults saved to: {output_path}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="q2-mechinterp", description="VAE feature extraction for omics data"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train VAE model")
    train_parser.add_argument("config", help="Path to config file (YAML/JSON/TOML)")
    train_parser.add_argument("--data", required=True, help="Path to data file")
    train_parser.add_argument("--log", action="store_true", help="Enable logging")
    train_parser.add_argument("--device", help="Device (cpu/cuda/mps)")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract features")
    extract_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    extract_parser.add_argument("--data", required=True, help="Path to data file")
    extract_parser.add_argument("--output", "-o", default="./features", help="Output directory")
    extract_parser.add_argument("--sparse", action="store_true", help="Extract sparse features")
    extract_parser.add_argument(
        "--expansion-factor", type=int, default=3, help="Sparse AE expansion"
    )
    extract_parser.add_argument(
        "--sparse-epochs", type=int, default=50, help="Sparse AE training epochs"
    )
    extract_parser.add_argument("--csv", action="store_true", help="Also save as CSV")
    extract_parser.add_argument("--device", help="Device (cpu/cuda/mps)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate features")
    eval_parser.add_argument("--features", required=True, help="Path to features file")
    eval_parser.add_argument("--labels", required=True, help="Path to labels file")
    eval_parser.add_argument("--cv", type=int, default=5, help="CV folds")
    eval_parser.add_argument("--models", help="Models to use (comma-separated)")
    eval_parser.add_argument("--output", "-o", help="Output path for results CSV")

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == "train":
            return cmd_train(args)
        elif args.command == "extract":
            return cmd_extract(args)
        elif args.command == "evaluate":
            return cmd_evaluate(args)
        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
