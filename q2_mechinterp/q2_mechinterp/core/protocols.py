"""
Protocol classes and configuration dataclasses.

This module defines:
1. Protocol classes for consistent interfaces across VAE variants
2. Configuration dataclasses for experiment reproducibility
3. Type aliases for cleaner type hints

Design Principles:
- Protocols enable duck typing with static type checking
- Dataclasses are immutable where possible
- All configs support YAML/TOML/JSON serialization
- Configs validate themselves on creation
"""

from typing import (
    Protocol, TypeVar, Optional, Dict, List, Tuple, Any, 
    Union, runtime_checkable, Callable, Sequence
)
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader


# =============================================================================
# Type Aliases
# =============================================================================

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, List[List[float]], torch.Tensor]
DeviceLike = Union[str, torch.device]
HiddenDims = List[int]

# Callback type for training hooks
TrainingCallback = Callable[[int, Dict[str, float]], None]


# =============================================================================
# Protocol Classes
# =============================================================================

@runtime_checkable
class VAEProtocol(Protocol):
    """
    Protocol defining the interface for all VAE models.
    
    Any class implementing encode(), decode(), forward(), and
    reparameterize() methods with the correct signatures is
    considered a valid VAE.
    
    Example:
        >>> def train_any_vae(model: VAEProtocol, data: torch.Tensor):
        ...     recon, mu, logvar = model(data)
        ...     # ... training logic
    """
    
    input_dim: int
    latent_dim: int
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space, returning (mu, logvar)."""
        ...
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        ...
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, mu, logvar)."""
        ...
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        ...


@runtime_checkable
class DataProcessorProtocol(Protocol):
    """
    Protocol defining the interface for data processors.
    
    Data processors handle loading, preprocessing, and splitting
    of data for VAE training.
    """
    
    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        ...
    
    @property
    def n_features(self) -> int:
        """Number of features after preprocessing."""
        ...
    
    def get_data_for_training(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get processed data and optional labels."""
        ...
    
    def get_feature_names(self) -> List[str]:
        """Get current feature names."""
        ...
    
    def prepare_datasets(
        self,
        batch_size: int = 32,
        **kwargs
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """Create train/val/test DataLoaders."""
        ...


@runtime_checkable
class SparseAEProtocol(Protocol):
    """Protocol for sparse autoencoders."""
    
    latent_dim: int
    sparse_dim: int
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode."""
        ...
    
    def get_sparse_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations."""
        ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for model trainers."""
    
    model: torch.nn.Module
    device: torch.device
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        **kwargs
    ) -> Tuple[List[float], List[float]]:
        """Train the model."""
        ...
    
    def validate(self, val_loader: DataLoader, **kwargs) -> float:
        """Validate model on validation set."""
        ...


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class VAEConfig:
    """
    Configuration for VAE model architecture.
    
    Attributes:
        input_dim: Dimension of input features
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        use_layer_norm: Whether to use layer normalization on input
        use_batch_norm: Whether to use batch normalization
        activation: Activation function name
    
    Example:
        >>> config = VAEConfig(input_dim=500, latent_dim=50)
        >>> config.save('config.yaml')
        >>> loaded = VAEConfig.load('config.yaml')
    """
    input_dim: int
    latent_dim: int = 50
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.2
    use_layer_norm: bool = True
    use_batch_norm: bool = True
    activation: str = 'leaky_relu'
    
    def __post_init__(self):
        """Validate configuration after creation."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        
        valid_activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'tanh']
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VAEConfig":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def save(self, path: PathLike, format: Optional[str] = None) -> None:
        """Save to YAML, TOML, or JSON file."""
        _save_config(self.to_dict(), path, format)
    
    @classmethod
    def load(cls, path: PathLike) -> "VAEConfig":
        """Load from YAML, TOML, or JSON file."""
        return cls.from_dict(_load_config(path))


@dataclass
class TrainingConfig:
    """
    Configuration for training process.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        beta: KL divergence weight (beta-VAE)
        alpha: L1 regularization weight on latent space
        gradient_clip: Maximum gradient norm for clipping
        noise_factor: Input noise for regularization
        early_stopping_patience: Epochs to wait before stopping
        min_delta: Minimum improvement to reset patience
        scheduler_type: LR scheduler type
        scheduler_patience: Scheduler patience (for plateau)
        scheduler_factor: LR reduction factor
        mixed_precision: Whether to use mixed precision training
        num_workers: DataLoader workers
    """
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta: float = 0.01
    alpha: float = 0.001
    gradient_clip: float = 1.0
    noise_factor: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    scheduler_type: str = 'plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Performance
    mixed_precision: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        valid_schedulers = ['plateau', 'cosine', 'step', 'exponential', 'none']
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"scheduler_type must be one of {valid_schedulers}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def save(self, path: PathLike, format: Optional[str] = None) -> None:
        _save_config(self.to_dict(), path, format)
    
    @classmethod
    def load(cls, path: PathLike) -> "TrainingConfig":
        return cls.from_dict(_load_config(path))


@dataclass
class SparseAEConfig:
    """
    Configuration for Sparse Autoencoder.
    
    Attributes:
        expansion_factor: Ratio of sparse_dim to latent_dim
        l1_coefficient: L1 regularization coefficient
        epochs: Training epochs for sparse AE
        learning_rate: Learning rate
        batch_size: Batch size
        use_batch_norm: Whether to use batch normalization
        sparse_type: Type of sparse AE ('standard', 'topk', 'deep')
        topk_k: K value for TopK sparse AE
    """
    expansion_factor: int = 3
    l1_coefficient: float = 0.001
    epochs: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 32
    use_batch_norm: bool = True
    sparse_type: str = 'standard'
    topk_k: Optional[int] = None
    
    def __post_init__(self):
        if self.expansion_factor < 1:
            raise ValueError(f"expansion_factor must be >= 1, got {self.expansion_factor}")
        if self.l1_coefficient < 0:
            raise ValueError(f"l1_coefficient must be non-negative")
        
        valid_types = ['standard', 'topk', 'deep']
        if self.sparse_type not in valid_types:
            raise ValueError(f"sparse_type must be one of {valid_types}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SparseAEConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def save(self, path: PathLike, format: Optional[str] = None) -> None:
        _save_config(self.to_dict(), path, format)
    
    @classmethod
    def load(cls, path: PathLike) -> "SparseAEConfig":
        return cls.from_dict(_load_config(path))


@dataclass
class DataConfig:
    """
    Configuration for data processing.
    
    Attributes:
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        stratify: Whether to stratify splits by labels
        
        # Microbiome-specific
        transformation: Data transformation method
        min_prevalence: Minimum prevalence threshold
        top_variance_pct: Top variance percentile to keep
        
        # Transcriptomics-specific
        top_genes_pct: Top genes percentage to keep
        min_expression: Minimum expression threshold
        log_transform: Whether to apply log transformation
    """
    # Split parameters
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    stratify: bool = True
    
    # Microbiome-specific
    transformation: str = 'rclr'
    min_prevalence: float = 0.1
    top_variance_pct: float = 10.0
    
    # Transcriptomics-specific
    top_genes_pct: float = 0.02
    min_expression: float = 0.0
    log_transform: bool = True
    
    # General
    max_features: Optional[int] = None
    standardize: bool = True
    standardize_method: str = 'standard'
    
    def __post_init__(self):
        if not 0 <= self.test_size < 1:
            raise ValueError(f"test_size must be in [0, 1), got {self.test_size}")
        if not 0 <= self.val_size < 1:
            raise ValueError(f"val_size must be in [0, 1), got {self.val_size}")
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
        
        valid_transforms = ['rclr', 'clr', 'log1p', 'none', 'standard']
        if self.transformation not in valid_transforms:
            raise ValueError(f"transformation must be one of {valid_transforms}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def save(self, path: PathLike, format: Optional[str] = None) -> None:
        _save_config(self.to_dict(), path, format)
    
    @classmethod
    def load(cls, path: PathLike) -> "DataConfig":
        return cls.from_dict(_load_config(path))


@dataclass
class EvaluationConfig:
    """
    Configuration for feature evaluation.
    
    Attributes:
        task: 'classification' or 'regression'
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        models: List of model names to use
        n_jobs: Number of parallel jobs
    """
    task: str = 'classification'
    cv_folds: int = 5
    scoring: Optional[str] = None
    models: List[str] = field(default_factory=lambda: ['logistic', 'rf'])
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.task not in ['classification', 'regression']:
            raise ValueError(f"task must be 'classification' or 'regression'")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration aggregating all component configs.
    
    Provides a single point of configuration for reproducible experiments.
    
    Attributes:
        vae: VAE model configuration
        training: Training configuration
        sparse_ae: Sparse autoencoder configuration
        data: Data processing configuration
        evaluation: Evaluation configuration
        output_dir: Directory for outputs
        experiment_name: Name for experiment tracking
        seed: Global random seed
        device: Compute device ('auto', 'cpu', 'cuda', 'mps')
    
    Example:
        >>> config = PipelineConfig.load('experiment.yaml')
        >>> # Or create with defaults and override
        >>> config = PipelineConfig(
        ...     vae=VAEConfig(input_dim=500),
        ...     experiment_name='my_experiment'
        ... )
    """
    vae: VAEConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sparse_ae: SparseAEConfig = field(default_factory=SparseAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Pipeline settings
    output_dir: str = './outputs'
    experiment_name: str = 'vae_experiment'
    seed: int = 42
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device not in ['auto', 'cpu', 'cuda', 'mps']:
            raise ValueError(f"device must be 'auto', 'cpu', 'cuda', or 'mps'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert full pipeline config to dictionary."""
        return {
            'vae': self.vae.to_dict(),
            'training': self.training.to_dict(),
            'sparse_ae': self.sparse_ae.to_dict(),
            'data': self.data.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        # Handle nested configs
        vae_dict = d.get('vae', {})
        if not vae_dict.get('input_dim'):
            raise ValueError("vae.input_dim is required")
        
        return cls(
            vae=VAEConfig.from_dict(vae_dict),
            training=TrainingConfig.from_dict(d.get('training', {})),
            sparse_ae=SparseAEConfig.from_dict(d.get('sparse_ae', {})),
            data=DataConfig.from_dict(d.get('data', {})),
            evaluation=EvaluationConfig.from_dict(d.get('evaluation', {})),
            output_dir=d.get('output_dir', './outputs'),
            experiment_name=d.get('experiment_name', 'vae_experiment'),
            seed=d.get('seed', 42),
            device=d.get('device', 'auto'),
        )
    
    def save(self, path: PathLike, format: Optional[str] = None) -> None:
        """Save full config to file."""
        _save_config(self.to_dict(), path, format)
    
    @classmethod
    def load(cls, path: PathLike) -> "PipelineConfig":
        """Load full config from file."""
        return cls.from_dict(_load_config(path))
    
    def get_device(self) -> torch.device:
        """Get resolved torch device."""
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(self.device)
    
    def get_output_path(self, filename: str) -> Path:
        """Get path for output file."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename


# =============================================================================
# Config I/O Utilities
# =============================================================================

def _save_config(data: Dict[str, Any], path: PathLike, format: Optional[str] = None) -> None:
    """Save configuration dictionary to file."""
    path = Path(path)
    
    # Determine format from extension if not specified
    if format is None:
        ext = path.suffix.lower()
        if ext in ['.yaml', '.yml']:
            format = 'yaml'
        elif ext == '.toml':
            format = 'toml'
        elif ext == '.json':
            format = 'json'
        else:
            format = 'yaml'  # Default
    
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'yaml':
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    elif format == 'toml':
        try:
            import toml
            with open(path, 'w') as f:
                toml.dump(data, f)
        except ImportError:
            raise ImportError("toml package required. Install with: pip install toml")
    
    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    else:
        raise ValueError(f"Unknown format: {format}. Use 'yaml', 'toml', or 'json'")


def _load_config(path: PathLike) -> Dict[str, Any]:
    """Load configuration dictionary from file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    ext = path.suffix.lower()
    
    if ext in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
    
    elif ext == '.toml':
        # Try tomllib (Python 3.11+) first, then toml package
        try:
            import tomllib
            with open(path, 'rb') as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import toml
                with open(path, 'r') as f:
                    return toml.load(f)
            except ImportError:
                raise ImportError("toml package required. Install with: pip install toml")
    
    elif ext == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    
    else:
        raise ValueError(f"Unknown file extension: {ext}. Use .yaml, .yml, .toml, or .json")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two config dictionaries.
    
    Values in override take precedence over base.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration
        override: Override values
    
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_config(
    input_dim: int,
    domain: str = 'microbiome',
    **overrides
) -> PipelineConfig:
    """
    Create a default configuration for a given domain.
    
    Args:
        input_dim: Number of input features
        domain: 'microbiome' or 'transcriptomics'
        **overrides: Override any config values
    
    Returns:
        PipelineConfig with domain-appropriate defaults
    """
    if domain == 'microbiome':
        vae_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=50,
            hidden_dims=[256, 128],
            dropout_rate=0.2
        )
        training_config = TrainingConfig(
            beta=0.01,  # Lower for sparse data
            alpha=0.001
        )
        data_config = DataConfig(
            transformation='rclr',
            min_prevalence=0.1
        )
    
    elif domain == 'transcriptomics':
        vae_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=50,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.2
        )
        training_config = TrainingConfig(
            beta=1.0,  # Standard for dense data
            alpha=0.1
        )
        data_config = DataConfig(
            transformation='log1p',
            top_genes_pct=0.02,
            log_transform=True
        )
    
    else:
        raise ValueError(f"Unknown domain: {domain}. Use 'microbiome' or 'transcriptomics'")
    
    config = PipelineConfig(
        vae=vae_config,
        training=training_config,
        data=data_config
    )
    
    # Apply overrides
    if overrides:
        config_dict = config.to_dict()
        merged = merge_configs(config_dict, overrides)
        config = PipelineConfig.from_dict(merged)
    
    return config
