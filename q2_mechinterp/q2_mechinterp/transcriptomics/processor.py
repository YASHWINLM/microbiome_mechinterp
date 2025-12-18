"""
Data processing for transcriptomics (RNA-seq) data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings


class TranscriptomicsDataProcessor:
    """
    Data processor for RNA-seq and other transcriptomics data.

    Handles gene expression data loading, filtering, normalization,
    and preparation for VAE training.

    Args:
        test_size: Fraction of data to hold out for validation.
        random_state: Random seed for reproducibility.

    Example:
        >>> processor = TranscriptomicsDataProcessor()
        >>> processor.load_expression_data('counts.csv')
        >>> processor.load_metadata('metadata.csv')
        >>> processor.filter_genes(top_pct=0.02)
        >>> processor.prepare_datasets()
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

        # Data containers
        self.expression_df = None
        self.metadata_df = None
        self.gene_names = None

        # Processed data
        self.expression_filtered = None
        self.scaled_data = None
        self.scaler = None

        # Labels
        self.labels = None
        self.encoded_labels = None
        self.label_encoder = None
        self.label_names = None

        # Train/val splits
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_expression_data(
        self, filepath: str, sep: str = "\t", index_col: int = 0, transpose: bool = False
    ) -> None:
        """
        Load gene expression data from file.

        Args:
            filepath: Path to expression data file (CSV/TSV).
            sep: Delimiter for the file.
            index_col: Column to use as index.
            transpose: Whether to transpose (if genes are rows).
        """
        print(f"Loading expression data from {filepath}...")

        self.expression_df = pd.read_csv(filepath, sep=sep, index_col=index_col)

        if transpose:
            self.expression_df = self.expression_df.T

        print(f"Expression data shape: {self.expression_df.shape}")
        print(f"Samples: {self.expression_df.shape[0]}, Genes: {self.expression_df.shape[1]}")

    def load_expression_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load expression data directly from DataFrame.

        Args:
            df: DataFrame with samples as rows and genes as columns.
        """
        self.expression_df = df.copy()
        print(f"Expression data shape: {self.expression_df.shape}")

    def load_metadata(self, filepath: str, sep: str = "\t", index_col: int = 0) -> None:
        """
        Load sample metadata.

        Args:
            filepath: Path to metadata file.
            sep: Delimiter for the file.
            index_col: Column to use as index (sample IDs).
        """
        print(f"Loading metadata from {filepath}...")

        # Try to infer format
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            self.metadata_df = pd.read_excel(filepath, index_col=index_col)
        else:
            self.metadata_df = pd.read_csv(filepath, sep=sep, index_col=index_col)

        print(f"Metadata shape: {self.metadata_df.shape}")

    def load_metadata_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load metadata directly from DataFrame."""
        self.metadata_df = df.copy()
        print(f"Metadata shape: {self.metadata_df.shape}")

    def load_from_dataframe(
        self,
        expression_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
        gene_column: Optional[str] = None,
    ) -> None:
        """
        Load expression and metadata from DataFrames.

        Args:
            expression_df: Expression DataFrame (samples x genes).
            metadata_df: Metadata DataFrame (samples x annotations).
            gene_column: If not None, use this column as gene names and transpose.
        """
        if gene_column is not None:
            # Genes are in a column, samples are rows
            self.expression_df = expression_df.set_index(gene_column).T
        else:
            self.expression_df = expression_df.copy()

        print(f"Expression data shape: {self.expression_df.shape}")

        if metadata_df is not None:
            self.metadata_df = metadata_df.copy()
            print(f"Metadata shape: {self.metadata_df.shape}")

    @property
    def n_genes(self) -> int:
        """Number of genes/features after filtering."""
        if self.expression_filtered is not None:
            return self.expression_filtered.shape[1]
        elif self.expression_df is not None:
            return self.expression_df.shape[1]
        return 0

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        if self.expression_filtered is not None:
            return self.expression_filtered.shape[0]
        elif self.expression_df is not None:
            return self.expression_df.shape[0]
        return 0

    def filter_genes(
        self,
        min_counts: int = 0,
        min_samples_pct: float = 0.0,
        top_pct: float = 0.02,
        by: str = "variance",
    ) -> None:
        """
        Filter genes based on expression levels.

        Args:
            min_counts: Minimum total counts across all samples.
            min_samples_pct: Minimum fraction of samples with non-zero expression.
            top_pct: Keep top X% of genes by expression metric.
            by: Metric for ranking ('mean', 'variance', 'cv').
        """
        if self.expression_df is None:
            raise ValueError("Expression data not loaded.")

        data = self.expression_df.copy()
        original_genes = data.shape[1]

        # Filter by minimum counts
        if min_counts > 0:
            gene_sums = data.sum(axis=0)
            data = data.loc[:, gene_sums >= min_counts]
            print(f"After min counts filter: {data.shape[1]} genes")

        # Filter by minimum samples
        if min_samples_pct > 0:
            expressed_frac = (data > 0).mean(axis=0)
            data = data.loc[:, expressed_frac >= min_samples_pct]
            print(f"After min samples filter: {data.shape[1]} genes")

        # Calculate metric for each gene
        if by == "mean":
            metric = data.mean(axis=0)
        elif by == "variance":
            metric = data.var(axis=0)
        elif by == "cv":
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            metric = stds / (means + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {by}")

        # Get top genes
        top_n = max(1, int(len(metric) * top_pct))
        top_genes = metric.nlargest(top_n).index

        self.expression_filtered = data[top_genes]
        self.gene_names = list(top_genes)

        print(f"Filtered from {original_genes} to {len(self.gene_names)} genes")

    def normalize(self, method: str = "log1p", pseudocount: float = 1.0) -> None:
        """
        Normalize expression data.

        Args:
            method: Normalization method ('log1p', 'log2', 'tpm', 'cpm', 'none').
            pseudocount: Value to add before log (for log methods).
        """
        data = (
            self.expression_filtered if self.expression_filtered is not None else self.expression_df
        )

        if data is None:
            raise ValueError("No expression data loaded.")

        if method == "log1p":
            data = np.log1p(data)
            print("Applied log1p transformation")
        elif method == "log2":
            data = np.log2(data + pseudocount)
            print(f"Applied log2 transformation (pseudocount={pseudocount})")
        elif method == "cpm":
            # Counts per million
            library_sizes = data.sum(axis=1)
            data = (data.T / library_sizes * 1e6).T
            print("Applied CPM normalization")
        elif method == "tpm":
            # TPM-like (simplified, assumes equal gene lengths)
            data = (data.T / data.sum(axis=1) * 1e6).T
            print("Applied TPM-like normalization")
        elif method == "none":
            print("No normalization applied")
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if self.expression_filtered is not None:
            self.expression_filtered = pd.DataFrame(
                data, index=self.expression_filtered.index, columns=self.expression_filtered.columns
            )
        else:
            self.expression_df = pd.DataFrame(
                data, index=self.expression_df.index, columns=self.expression_df.columns
            )

    def scale_features(self, method: str = "standard") -> None:
        """
        Scale features for model training.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'none').
        """
        data = (
            self.expression_filtered if self.expression_filtered is not None else self.expression_df
        )

        if data is None:
            raise ValueError("No expression data loaded.")

        if method == "standard":
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            self.scaler = RobustScaler()
        elif method == "none":
            self.scaled_data = data.values
            print("No scaling applied")
            return
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scaled_data = self.scaler.fit_transform(data)
        print(f"Applied {method} scaling")

    def prepare_datasets(
        self,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify_column: Optional[str] = None,
    ):
        """
        Prepare DataLoaders for VAE training.

        Args:
            batch_size: Batch size for DataLoaders.
            test_size: Fraction for test set.
            val_size: Fraction for validation set.
            random_state: Random seed.
            stratify_column: Column in metadata to stratify by.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Use scaled data or scale if not done yet
        if self.scaled_data is None:
            self.scale_features()

        data = self.scaled_data

        # Get stratification labels if specified
        stratify = None
        if stratify_column and self.metadata_df is not None:
            if stratify_column in self.metadata_df.columns:
                # Align metadata with data
                data_index = (
                    self.expression_filtered.index
                    if self.expression_filtered is not None
                    else self.expression_df.index
                )
                aligned_meta = self.metadata_df.loc[data_index]
                stratify = aligned_meta[stratify_column].values

                # Encode labels
                self.label_encoder = LabelEncoder()
                self.encoded_labels = self.label_encoder.fit_transform(stratify)
                self.label_names = list(self.label_encoder.classes_)
                stratify = self.encoded_labels

        # Split data
        if test_size > 0:
            if stratify is not None:
                X_trainval, X_test, stratify_trainval, _ = train_test_split(
                    data,
                    stratify,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
            else:
                X_trainval, X_test = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
                stratify_trainval = None
        else:
            X_trainval = data
            X_test = None
            stratify_trainval = stratify

        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size) if test_size > 0 else val_size
            if stratify_trainval is not None:
                X_train, X_val, _, _ = train_test_split(
                    X_trainval,
                    stratify_trainval,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    stratify=stratify_trainval,
                )
            else:
                X_train, X_val = train_test_split(
                    X_trainval, test_size=val_size_adjusted, random_state=random_state
                )
        else:
            X_train = X_trainval
            X_val = None

        self.X_train = X_train
        self.X_val = X_val

        # Create DataLoaders
        train_tensor = torch.FloatTensor(X_train)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_tensor = torch.FloatTensor(X_val)
            val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        test_loader = None
        if X_test is not None:
            test_tensor = torch.FloatTensor(X_test)
            test_loader = DataLoader(
                TensorDataset(test_tensor), batch_size=batch_size, shuffle=False
            )

        print(f"Train samples: {len(X_train)}")
        if X_val is not None:
            print(f"Val samples: {len(X_val)}")
        if X_test is not None:
            print(f"Test samples: {len(X_test)}")

        return train_loader, val_loader, test_loader

    def filter_protein_coding(self, gene_type_col: str = "gene_type") -> None:
        """
        Filter to protein coding genes only.

        Expects gene names/columns to contain gene type annotation,
        or a separate gene annotation file.

        Args:
            gene_type_col: Column name containing gene type (if in columns).
        """
        if self.expression_df is None:
            raise ValueError("Expression data not loaded.")

        # Try to filter by column name pattern (TCGA-style)
        protein_coding = [
            col for col in self.expression_df.columns if "protein_coding" in str(col).lower()
        ]

        if len(protein_coding) > 0:
            self.expression_df = self.expression_df[protein_coding]
            print(f"Filtered to {len(protein_coding)} protein coding genes")
        else:
            print("Warning: Could not identify protein coding genes by column names")

    def log_transform(self, pseudocount: float = 1.0) -> None:
        """
        Apply log transformation to expression data.

        Args:
            pseudocount: Value to add before log (prevents log(0)).
        """
        if self.expression_filtered is not None:
            self.expression_filtered = np.log2(self.expression_filtered + pseudocount)
        elif self.expression_df is not None:
            self.expression_df = np.log2(self.expression_df + pseudocount)

        print("Applied log2 transformation")

    def get_data_for_training(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get prepared data for VAE training.

        Returns:
            Tuple of (X_train, X_val, y_train, y_val).
        """
        return self.X_train, self.X_val, self.y_train, self.y_val

    def get_feature_names(self) -> List[str]:
        """Get gene/feature names."""
        if self.gene_names is None:
            if self.expression_filtered is not None:
                return list(self.expression_filtered.columns)
            elif self.expression_df is not None:
                return list(self.expression_df.columns)
        return self.gene_names

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.

        Args:
            scaled_data: Standardized data.

        Returns:
            Data in original scale.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call prepare_datasets first.")
        return self.scaler.inverse_transform(scaled_data)
