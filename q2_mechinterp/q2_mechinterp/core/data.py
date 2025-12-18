"""
Unified base class for data processors.

This module provides a common interface for all data processors (microbiome, 
transcriptomics, etc.) to reduce code duplication and ensure consistent behavior.

Design Principles:
- Single Responsibility: Each method does one thing well
- Open/Closed: Extend via subclassing, not modification
- Interface Segregation: Optional methods via mixins
- Dependency Inversion: Depend on abstractions (protocols)
"""

from abc import ABC, abstractmethod
from typing import (
    Tuple, Optional, List, Dict, Any, Union, 
    Callable, TypeVar, Generic, Iterator
)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dataclasses import dataclass, field
import warnings

# Type aliases for clarity
ArrayLike = Union[np.ndarray, pd.DataFrame, List[List[float]]]
PathLike = Union[str, Path]
T = TypeVar('T', bound='BaseDataProcessor')


@dataclass
class DataSplit:
    """Container for train/val/test split indices and data."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: Optional[np.ndarray] = None
    
    @property
    def n_train(self) -> int:
        return len(self.train_idx)
    
    @property
    def n_val(self) -> int:
        return len(self.val_idx)
    
    @property
    def n_test(self) -> int:
        return len(self.test_idx) if self.test_idx is not None else 0


@dataclass
class ProcessorState:
    """Immutable state snapshot for reproducibility and checkpointing."""
    data_shape: Tuple[int, int]
    n_features_original: int
    n_features_filtered: int
    transformations_applied: List[str] = field(default_factory=list)
    split_info: Optional[Dict[str, int]] = None
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_shape': self.data_shape,
            'n_features_original': self.n_features_original,
            'n_features_filtered': self.n_features_filtered,
            'transformations_applied': list(self.transformations_applied),
            'split_info': self.split_info,
            'random_state': self.random_state
        }


class BaseDataProcessor(ABC):
    """
    Abstract base class for all data processors.
    
    Provides common functionality for:
    - Data loading from files and DataFrames
    - Train/val/test splitting with stratification
    - DataLoader creation with configurable parameters
    - Feature filtering by prevalence and variance
    - State management for reproducibility
    
    Subclasses implement domain-specific transformations via the
    abstract `transform()` method.
    
    Example:
        >>> class MyProcessor(BaseDataProcessor):
        ...     def transform(self, method='standard'):
        ...         # Apply domain-specific transformation
        ...         self._data = (self._data - self._data.mean()) / self._data.std()
        ...         self._transformations.append(f'standardize:{method}')
        ...         return self
        ...
        >>> processor = MyProcessor()
        >>> processor.load_from_numpy(data)
        >>> processor.transform()
        >>> train_loader, val_loader, test_loader = processor.prepare_datasets()
    
    Attributes:
        device: PyTorch device for tensor operations
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        random_state: int = 42
    ):
        """
        Initialize processor.
        
        Args:
            device: PyTorch device (auto-detected if None)
            random_state: Random seed for reproducibility
        """
        self.device = device or torch.device('cpu')
        self.random_state = random_state
        
        # Core data storage
        self._data: Optional[np.ndarray] = None
        self._raw_data: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._sample_names: List[str] = []
        
        # Labels
        self._labels: Optional[np.ndarray] = None
        self._encoded_labels: Optional[np.ndarray] = None
        self._label_names: Optional[List[str]] = None
        self._label_encoder: Optional[Any] = None
        
        # Metadata
        self._metadata: Optional[pd.DataFrame] = None
        
        # Split tracking
        self._split: Optional[DataSplit] = None
        
        # Transformation history (for reproducibility)
        self._transformations: List[str] = []
        
        # Feature mask (for filtering)
        self._feature_mask: Optional[np.ndarray] = None
        self._original_feature_names: List[str] = []
    
    # =========================================================================
    # Properties (read-only access to internal state)
    # =========================================================================
    
    @property
    def data(self) -> np.ndarray:
        """Processed data matrix (samples × features)."""
        if self._data is None:
            raise ValueError(
                "Data not loaded. Call load_from_dataframe(), "
                "load_from_numpy(), or a domain-specific load method first."
            )
        return self._data
    
    @property
    def raw_data(self) -> Optional[np.ndarray]:
        """Original unprocessed data (if preserved)."""
        return self._raw_data
    
    @property
    def feature_names(self) -> List[str]:
        """Current feature names after any filtering."""
        return self._feature_names
    
    @property
    def sample_names(self) -> List[str]:
        """Sample identifiers."""
        return self._sample_names
    
    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return self._data.shape[0] if self._data is not None else 0
    
    @property
    def n_features(self) -> int:
        """Number of features after filtering."""
        return self._data.shape[1] if self._data is not None else 0
    
    @property
    def n_features_original(self) -> int:
        """Original number of features before filtering."""
        return len(self._original_feature_names) if self._original_feature_names else self.n_features
    
    @property
    def labels(self) -> Optional[np.ndarray]:
        """Raw labels (before encoding)."""
        return self._labels
    
    @property
    def encoded_labels(self) -> Optional[np.ndarray]:
        """Integer-encoded labels for classification."""
        return self._encoded_labels
    
    @property
    def label_names(self) -> Optional[List[str]]:
        """Unique label values in encoding order."""
        return self._label_names
    
    @property
    def n_classes(self) -> int:
        """Number of unique classes (0 if no labels)."""
        return len(self._label_names) if self._label_names else 0
    
    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Sample metadata DataFrame."""
        return self._metadata
    
    @property
    def transformations(self) -> List[str]:
        """List of transformations applied (for reproducibility)."""
        return list(self._transformations)
    
    @property
    def is_loaded(self) -> bool:
        """Whether data has been loaded."""
        return self._data is not None
    
    @property
    def has_labels(self) -> bool:
        """Whether labels are available."""
        return self._encoded_labels is not None
    
    @property
    def has_split(self) -> bool:
        """Whether data has been split."""
        return self._split is not None
    
    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================
    
    @abstractmethod
    def transform(self: T, **kwargs) -> T:
        """
        Apply domain-specific transformations.
        
        Subclasses implement this with their specific preprocessing
        (e.g., RCLR for microbiome, log-normalization for RNA-seq).
        
        Returns:
            self for method chaining
        """
        pass
    
    # =========================================================================
    # Data Loading Methods
    # =========================================================================
    
    def load_from_dataframe(
        self: T,
        data_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        label_column: Optional[str] = None,
        preserve_raw: bool = True
    ) -> T:
        """
        Load data from pandas DataFrames.
        
        Args:
            data_df: Data matrix (samples × features), index = sample IDs
            metadata_df: Optional metadata with sample info
            label_column: Column in metadata to use as labels
            preserve_raw: Whether to keep a copy of raw data
        
        Returns:
            self for method chaining
        
        Raises:
            ValueError: If data_df is empty or contains non-numeric data
            TypeError: If inputs are not DataFrames
        """
        # Validate inputs
        self._validate_dataframe(data_df, "data_df")
        
        # Store data
        self._data = data_df.values.astype(np.float32)
        if preserve_raw:
            self._raw_data = self._data.copy()
        
        self._feature_names = list(data_df.columns)
        self._original_feature_names = list(data_df.columns)
        self._sample_names = list(data_df.index)
        
        # Handle metadata and labels
        if metadata_df is not None:
            self._validate_dataframe(metadata_df, "metadata_df", allow_non_numeric=True)
            self._metadata = metadata_df.copy()
            
            if label_column and label_column in metadata_df.columns:
                # Align metadata with data
                common_samples = list(set(self._sample_names) & set(metadata_df.index))
                if len(common_samples) < len(self._sample_names):
                    warnings.warn(
                        f"Only {len(common_samples)}/{len(self._sample_names)} "
                        f"samples have metadata. Others will have no labels."
                    )
                self._set_labels_from_metadata(metadata_df, label_column, common_samples)
        
        self._transformations.append('load_from_dataframe')
        return self
    
    def load_from_numpy(
        self: T,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_names: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        preserve_raw: bool = True
    ) -> T:
        """
        Load data from numpy arrays.
        
        Args:
            data: Data matrix (samples × features)
            feature_names: Optional feature names (auto-generated if None)
            sample_names: Optional sample names (auto-generated if None)
            labels: Optional labels array
            preserve_raw: Whether to keep a copy of raw data
        
        Returns:
            self for method chaining
        
        Raises:
            ValueError: If data is invalid (wrong dims, contains NaN/Inf)
        """
        # Validate
        data = self._validate_array(data, "data", ndim=2)
        
        # Store
        self._data = data.astype(np.float32)
        if preserve_raw:
            self._raw_data = self._data.copy()
        
        self._feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
        self._original_feature_names = list(self._feature_names)
        self._sample_names = sample_names or [f"sample_{i}" for i in range(data.shape[0])]
        
        if labels is not None:
            self._set_labels(labels)
        
        self._transformations.append('load_from_numpy')
        return self
    
    def load_from_file(
        self: T,
        filepath: PathLike,
        sep: str = '\t',
        index_col: int = 0,
        transpose: bool = False,
        **kwargs
    ) -> T:
        """
        Load data from CSV/TSV file.
        
        Args:
            filepath: Path to data file
            sep: Column separator
            index_col: Column to use as index
            transpose: Whether to transpose (if features are rows)
            **kwargs: Additional arguments passed to pd.read_csv
        
        Returns:
            self for method chaining
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath, sep=sep, index_col=index_col, **kwargs)
        
        if transpose:
            df = df.T
        
        return self.load_from_dataframe(df)
    
    # =========================================================================
    # Label Handling
    # =========================================================================
    
    def _set_labels(self, labels: np.ndarray) -> None:
        """Set and encode labels."""
        self._labels = np.asarray(labels)
        
        # Get unique labels preserving order of first appearance
        _, idx = np.unique(self._labels, return_index=True)
        unique_sorted = self._labels[np.sort(idx)]
        self._label_names = list(np.unique(self._labels))  # Alphabetically sorted
        
        # Create encoding
        label_to_int = {label: i for i, label in enumerate(self._label_names)}
        self._encoded_labels = np.array([label_to_int[l] for l in self._labels])
    
    def _set_labels_from_metadata(
        self,
        metadata_df: pd.DataFrame,
        label_column: str,
        sample_order: List[str]
    ) -> None:
        """Extract and set labels from metadata."""
        # Reorder metadata to match sample order
        labels = []
        for sample in self._sample_names:
            if sample in metadata_df.index:
                labels.append(metadata_df.loc[sample, label_column])
            else:
                labels.append(None)
        
        # Filter out None values and their corresponding indices
        valid_mask = [l is not None for l in labels]
        if not all(valid_mask):
            warnings.warn(
                f"{sum(not v for v in valid_mask)} samples have missing labels"
            )
        
        labels = [l for l in labels if l is not None]
        self._set_labels(np.array(labels))
    
    def set_labels(self: T, labels: np.ndarray, label_names: Optional[List[str]] = None) -> T:
        """
        Manually set labels for the dataset.
        
        Args:
            labels: Label array (length must match n_samples)
            label_names: Optional explicit ordering of label names
        
        Returns:
            self for method chaining
        """
        if len(labels) != self.n_samples:
            raise ValueError(
                f"Labels length ({len(labels)}) doesn't match n_samples ({self.n_samples})"
            )
        
        self._set_labels(labels)
        
        if label_names is not None:
            self._label_names = list(label_names)
            # Re-encode with new ordering
            label_to_int = {label: i for i, label in enumerate(self._label_names)}
            self._encoded_labels = np.array([label_to_int[l] for l in self._labels])
        
        return self
    
    # =========================================================================
    # Feature Filtering
    # =========================================================================
    
    def filter_features(
        self: T,
        prevalence_threshold: float = 0.0,
        variance_percentile: float = 0.0,
        min_value: Optional[float] = None,
        max_features: Optional[int] = None,
        feature_names_to_keep: Optional[List[str]] = None,
        inplace: bool = True
    ) -> T:
        """
        Filter features by prevalence, variance, or explicit selection.
        
        Filtering operations are applied in order:
        1. Explicit feature selection (if provided)
        2. Prevalence threshold
        3. Variance percentile
        4. Max features limit
        
        Args:
            prevalence_threshold: Min fraction of samples with non-zero values
            variance_percentile: Remove bottom X% by variance (0-100)
            min_value: Optional minimum max-value threshold
            max_features: Maximum number of features to keep
            feature_names_to_keep: Explicit list of features to keep
            inplace: If False, return filtered copy without modifying self
        
        Returns:
            self (or copy if inplace=False)
        """
        if self._data is None:
            raise ValueError("Data not loaded")
        
        # Start with all features
        mask = np.ones(self.n_features, dtype=bool)
        
        # 1. Explicit feature selection
        if feature_names_to_keep is not None:
            keep_set = set(feature_names_to_keep)
            mask &= np.array([f in keep_set for f in self._feature_names])
        
        # 2. Prevalence filter (fraction of samples with non-zero values)
        if prevalence_threshold > 0:
            prevalence = (self._data > 0).mean(axis=0)
            mask &= (prevalence >= prevalence_threshold)
        
        # 3. Variance filter (on remaining features)
        if variance_percentile > 0:
            variances = self._data[:, mask].var(axis=0)
            if len(variances) > 0:
                var_threshold = np.percentile(variances, variance_percentile)
                # Create variance mask for all features
                all_variances = self._data.var(axis=0)
                mask &= (all_variances >= var_threshold)
        
        # 4. Min value filter
        if min_value is not None:
            max_vals = self._data.max(axis=0)
            mask &= (max_vals >= min_value)
        
        # 5. Max features limit (keep highest variance)
        if max_features is not None and mask.sum() > max_features:
            masked_indices = np.where(mask)[0]
            variances = self._data[:, masked_indices].var(axis=0)
            top_var_indices = np.argsort(variances)[-max_features:]
            new_mask = np.zeros_like(mask)
            new_mask[masked_indices[top_var_indices]] = True
            mask = new_mask
        
        # Apply filter
        n_before = self.n_features
        
        if inplace:
            self._apply_feature_mask(mask)
            self._transformations.append(
                f'filter_features(prev={prevalence_threshold},var_pct={variance_percentile},'
                f'kept={self.n_features}/{n_before})'
            )
            return self
        else:
            # Return filtered copy
            copy = self._shallow_copy()
            copy._apply_feature_mask(mask)
            return copy
    
    def _apply_feature_mask(self, mask: np.ndarray) -> None:
        """Apply boolean mask to filter features."""
        self._data = self._data[:, mask]
        self._feature_names = [f for f, m in zip(self._feature_names, mask) if m]
        self._feature_mask = mask if self._feature_mask is None else self._feature_mask & mask
    
    def _shallow_copy(self: T) -> T:
        """Create a shallow copy for non-inplace operations."""
        import copy
        return copy.copy(self)
    
    # =========================================================================
    # Train/Val/Test Splitting
    # =========================================================================
    
    def split_data(
        self: T,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: Optional[int] = None
    ) -> T:
        """
        Split data into train/val/test sets.
        
        Args:
            test_size: Fraction for test set (0 to skip test split)
            val_size: Fraction for validation (from remaining after test)
            stratify: Whether to stratify by labels (if available)
            random_state: Random seed (uses self.random_state if None)
        
        Returns:
            self for method chaining
        """
        from sklearn.model_selection import train_test_split
        
        if self._data is None:
            raise ValueError("Data not loaded")
        
        random_state = random_state or self.random_state
        indices = np.arange(self.n_samples)
        
        # Determine stratification
        stratify_labels = None
        if stratify and self._encoded_labels is not None:
            stratify_labels = self._encoded_labels
        
        # First split: separate test set
        if test_size > 0:
            idx_trainval, idx_test = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )
            stratify_trainval = stratify_labels[idx_trainval] if stratify_labels is not None else None
        else:
            idx_trainval = indices
            idx_test = None
            stratify_trainval = stratify_labels
        
        # Second split: separate validation from training
        if val_size > 0:
            # Adjust val_size to be fraction of trainval
            adjusted_val_size = val_size / (1 - test_size) if test_size > 0 else val_size
            adjusted_val_size = min(adjusted_val_size, 0.5)  # Cap at 50%
            
            idx_train, idx_val = train_test_split(
                idx_trainval,
                test_size=adjusted_val_size,
                random_state=random_state,
                stratify=stratify_trainval
            )
        else:
            idx_train = idx_trainval
            idx_val = np.array([], dtype=int)
        
        # Store split
        self._split = DataSplit(
            train_idx=idx_train,
            val_idx=idx_val,
            test_idx=idx_test
        )
        
        self._transformations.append(
            f'split_data(train={len(idx_train)},val={len(idx_val)},'
            f'test={len(idx_test) if idx_test is not None else 0})'
        )
        
        return self
    
    # =========================================================================
    # DataLoader Creation
    # =========================================================================
    
    def prepare_datasets(
        self: T,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        include_labels: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train/val/test DataLoaders.
        
        If data hasn't been split, performs split first.
        
        Args:
            batch_size: Batch size for all loaders
            test_size: Fraction for test set
            val_size: Fraction for validation
            stratify: Whether to stratify by labels
            num_workers: DataLoader workers
            pin_memory: Pin memory for GPU (auto-detected if None)
            include_labels: Whether to include labels in TensorDataset
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split if not already done
        if self._split is None:
            self.split_data(test_size=test_size, val_size=val_size, stratify=stratify)
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = self.device.type == 'cuda'
        
        # Create loaders
        train_loader = self._create_loader(
            self._split.train_idx,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            include_labels=include_labels
        )
        
        val_loader = self._create_loader(
            self._split.val_idx,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            include_labels=include_labels
        ) if len(self._split.val_idx) > 0 else None
        
        test_loader = self._create_loader(
            self._split.test_idx,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            include_labels=include_labels
        ) if self._split.test_idx is not None and len(self._split.test_idx) > 0 else None
        
        return train_loader, val_loader, test_loader
    
    def _create_loader(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        pin_memory: bool = False,
        include_labels: bool = False
    ) -> DataLoader:
        """Create a DataLoader for specified indices."""
        if len(indices) == 0:
            return None
        
        X = torch.FloatTensor(self._data[indices])
        
        if include_labels and self._encoded_labels is not None:
            y = torch.LongTensor(self._encoded_labels[indices])
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)
        
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(indices)),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    # =========================================================================
    # Data Access Methods
    # =========================================================================
    
    def get_data_for_training(
        self
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get processed data and labels.
        
        Returns:
            Tuple of (data, labels) where labels may be None
        """
        return self._data, self._encoded_labels
    
    def get_split_data(
        self,
        split: str = 'train'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get data for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
        
        Returns:
            Tuple of (X, y) for the split
        """
        if self._split is None:
            raise ValueError("Data not split. Call split_data() or prepare_datasets() first.")
        
        idx_map = {
            'train': self._split.train_idx,
            'val': self._split.val_idx,
            'test': self._split.test_idx
        }
        
        if split not in idx_map:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")
        
        idx = idx_map[split]
        if idx is None or len(idx) == 0:
            raise ValueError(f"Split '{split}' is empty or not available")
        
        X = self._data[idx]
        y = self._encoded_labels[idx] if self._encoded_labels is not None else None
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get current feature names."""
        return list(self._feature_names)
    
    def get_sample_names(self, split: Optional[str] = None) -> List[str]:
        """
        Get sample names, optionally for a specific split.
        
        Args:
            split: Optional split ('train', 'val', 'test')
        
        Returns:
            List of sample names
        """
        if split is None:
            return list(self._sample_names)
        
        if self._split is None:
            raise ValueError("Data not split")
        
        idx_map = {
            'train': self._split.train_idx,
            'val': self._split.val_idx,
            'test': self._split.test_idx
        }
        
        idx = idx_map.get(split)
        if idx is None:
            raise ValueError(f"Unknown split: {split}")
        
        return [self._sample_names[i] for i in idx]
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def get_state(self) -> ProcessorState:
        """
        Get current processor state for reproducibility.
        
        Returns:
            ProcessorState dataclass with current configuration
        """
        split_info = None
        if self._split is not None:
            split_info = {
                'n_train': self._split.n_train,
                'n_val': self._split.n_val,
                'n_test': self._split.n_test
            }
        
        return ProcessorState(
            data_shape=(self.n_samples, self.n_features),
            n_features_original=self.n_features_original,
            n_features_filtered=self.n_features,
            transformations_applied=list(self._transformations),
            split_info=split_info,
            random_state=self.random_state
        )
    
    def reset(self: T) -> T:
        """
        Reset to raw data state, clearing all transformations.
        
        Returns:
            self for method chaining
        """
        if self._raw_data is not None:
            self._data = self._raw_data.copy()
            self._feature_names = list(self._original_feature_names)
            self._feature_mask = None
            self._split = None
            self._transformations = ['reset']
        else:
            raise ValueError("Raw data not preserved. Cannot reset.")
        return self
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def _validate_array(
        self,
        arr: ArrayLike,
        name: str = "array",
        ndim: Optional[int] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> np.ndarray:
        """Validate and convert array-like to numpy array."""
        if arr is None:
            raise ValueError(f"{name} cannot be None")
        
        # Convert to numpy
        if isinstance(arr, pd.DataFrame):
            arr = arr.values
        elif isinstance(arr, list):
            arr = np.array(arr)
        elif not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be array-like, got {type(arr)}")
        
        # Check dimensions
        if ndim is not None and arr.ndim != ndim:
            raise ValueError(
                f"{name} must be {ndim}D, got {arr.ndim}D with shape {arr.shape}"
            )
        
        # Check for NaN/Inf
        if not allow_nan and np.isnan(arr).any():
            n_nan = np.isnan(arr).sum()
            raise ValueError(f"{name} contains {n_nan} NaN values")
        
        if not allow_inf and np.isinf(arr).any():
            n_inf = np.isinf(arr).sum()
            raise ValueError(f"{name} contains {n_inf} infinite values")
        
        return arr
    
    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        name: str = "dataframe",
        allow_non_numeric: bool = False
    ) -> None:
        """Validate pandas DataFrame."""
        if df is None:
            raise ValueError(f"{name} cannot be None")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValueError(f"{name} is empty")
        
        if not allow_non_numeric:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) != len(df.columns):
                non_numeric = set(df.columns) - set(numeric_cols)
                raise ValueError(
                    f"{name} has non-numeric columns: {list(non_numeric)[:5]}..."
                )
    
    # =========================================================================
    # Magic Methods
    # =========================================================================
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "empty"
        split_str = ""
        if self._split:
            split_str = f", split=({self._split.n_train}/{self._split.n_val}/{self._split.n_test})"
        
        return (
            f"{self.__class__.__name__}("
            f"status={status}, "
            f"shape={self._data.shape if self._data is not None else None}"
            f"{split_str})"
        )
    
    def __len__(self) -> int:
        return self.n_samples


# =============================================================================
# Mixin Classes for Optional Functionality
# =============================================================================

class ScalerMixin:
    """Mixin providing standardization/normalization capabilities."""
    
    _scaler: Optional[Any] = None
    
    def standardize(
        self,
        method: str = 'standard',
        **kwargs
    ) -> 'ScalerMixin':
        """
        Standardize/normalize features.
        
        Args:
            method: 'standard' (z-score), 'minmax', 'robust', 'quantile'
            **kwargs: Additional arguments for scaler
        
        Returns:
            self for method chaining
        """
        from sklearn.preprocessing import (
            StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
        )
        
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'quantile': QuantileTransformer
        }
        
        if method not in scalers:
            raise ValueError(f"Unknown method: {method}. Use one of {list(scalers.keys())}")
        
        self._scaler = scalers[method](**kwargs)
        self._data = self._scaler.fit_transform(self._data)
        self._transformations.append(f'standardize:{method}')
        
        return self
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data
        
        Returns:
            Data in original scale
        """
        if self._scaler is None:
            raise ValueError("No scaler fitted. Call standardize() first.")
        return self._scaler.inverse_transform(data)


class BatchInfoMixin:
    """Mixin for handling batch information in multi-study data."""
    
    _batch_labels: Optional[np.ndarray] = None
    _batch_names: Optional[List[str]] = None
    
    def set_batch_labels(
        self,
        batch_labels: np.ndarray,
        batch_column: Optional[str] = None
    ) -> 'BatchInfoMixin':
        """
        Set batch labels for batch effect correction.
        
        Args:
            batch_labels: Array of batch assignments
            batch_column: If provided, extract from metadata instead
        
        Returns:
            self for method chaining
        """
        if batch_column is not None and self._metadata is not None:
            if batch_column in self._metadata.columns:
                batch_labels = self._metadata[batch_column].values
        
        self._batch_labels = np.asarray(batch_labels)
        self._batch_names = list(np.unique(self._batch_labels))
        
        return self
    
    @property
    def batch_labels(self) -> Optional[np.ndarray]:
        return self._batch_labels
    
    @property
    def n_batches(self) -> int:
        return len(self._batch_names) if self._batch_names else 0
