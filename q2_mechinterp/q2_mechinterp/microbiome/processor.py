"""
Data processing for microbiome (metagenomics/metatranscriptomics) data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings


class MicrobiomeDataProcessor:
    """
    Data processor for microbiome data supporting metagenomics and metatranscriptomics.
    
    Handles BIOM table loading, RCLR transformation, feature filtering,
    and dataset preparation for VAE training.
    
    Args:
        data_paths: Dictionary with paths to data files.
            Required keys depend on use case but may include:
            - 'metag_biom': Path to metagenomics BIOM file
            - 'metat_biom': Path to metatranscriptomics BIOM file
            - 'metadata': Path to metadata file
            - 'lineages': Path to taxonomy lineages file
        device: PyTorch device for tensor operations.
    
    Example:
        >>> data_paths = {
        ...     'metag_biom': 'path/to/metagenomics.biom',
        ...     'metat_biom': 'path/to/metatranscriptomics.biom',
        ...     'metadata': 'path/to/metadata.txt',
        ...     'lineages': 'path/to/lineages.txt'
        ... }
        >>> processor = MicrobiomeDataProcessor(data_paths)
        >>> processor.load_data()
        >>> processor.apply_rclr_transformation()
        >>> processor.filter_features()
        >>> processor.prepare_datasets()
    """
    
    def __init__(self, data_paths: Dict[str, str], device=None):
        self.data_paths = data_paths
        self.device = device
        
        # Data containers
        self.table_metag = None
        self.table_metaT = None
        self.table_metag_df = None
        self.table_metaT_df = None
        self.table_metag_rclr = None
        self.table_metaT_rclr = None
        self.table_metag_filtered = None
        self.table_metaT_filtered = None
        
        self.metadata = None
        self.metadata_df = None
        self.lineages = None
        self.metag_taxonomy = None
        self.metaT_taxonomy = None
        
        # Processed data
        self.common_samples = None
        self.metag_data = None
        self.metaT_data = None
        self.metag_scaled = None
        self.metaT_scaled = None
        self.combined_data = None
        
        self.labels = None
        self.encoded_labels = None
        self.label_encoder = None
        self.label_names = None
        
        self.scaler_metag = None
        self.scaler_metaT = None
    
    def load_data(self, label_column: str = 'diagnosis') -> None:
        """
        Load microbiome data from BIOM files and metadata.
        
        Args:
            label_column: Column name in metadata for labels.
        """
        try:
            import biom
            from qiime2 import Metadata
        except ImportError:
            raise ImportError(
                "biom-format and qiime2 packages required. "
                "Install with: pip install biom-format qiime2"
            )
        
        print("Loading microbiome data...")
        
        # Load biom tables
        if 'metag_biom' in self.data_paths:
            self.table_metag = biom.load_table(self.data_paths['metag_biom'])
            self.table_metag_df = self.table_metag.to_dataframe().T
            print(f"MetaG shape: {self.table_metag_df.shape}")
        
        if 'metat_biom' in self.data_paths:
            self.table_metaT = biom.load_table(self.data_paths['metat_biom'])
            self.table_metaT_df = self.table_metaT.to_dataframe().T
            print(f"MetaT shape: {self.table_metaT_df.shape}")
        
        # Load metadata
        if 'metadata' in self.data_paths:
            self.metadata = Metadata.load(self.data_paths['metadata'])
            self.metadata_df = self.metadata.to_dataframe()
            print(f"Metadata shape: {self.metadata_df.shape}")
            
            if label_column in self.metadata_df.columns:
                print(f"Label distribution:\n{self.metadata_df[label_column].value_counts()}")
        
        # Load lineages/taxonomy
        if 'lineages' in self.data_paths:
            self._load_taxonomy()
    
    def load_data_from_dataframes(
        self,
        metag_df: Optional[pd.DataFrame] = None,
        metat_df: Optional[pd.DataFrame] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        taxonomy_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Load data directly from DataFrames (alternative to file loading).
        
        Args:
            metag_df: Metagenomics abundance DataFrame (samples x features).
            metat_df: Metatranscriptomics abundance DataFrame (samples x features).
            metadata_df: Metadata DataFrame with sample annotations.
            taxonomy_df: Taxonomy DataFrame with 'genome_id' and 'taxonomy' columns.
        """
        print("Loading microbiome data from DataFrames...")
        
        if metag_df is not None:
            self.table_metag_df = metag_df.copy()
            print(f"MetaG shape: {self.table_metag_df.shape}")
        
        if metat_df is not None:
            self.table_metaT_df = metat_df.copy()
            print(f"MetaT shape: {self.table_metaT_df.shape}")
        
        if metadata_df is not None:
            self.metadata_df = metadata_df.copy()
            print(f"Metadata shape: {self.metadata_df.shape}")
        
        if taxonomy_df is not None:
            self.lineages = taxonomy_df.copy()
            self._process_taxonomy()
    
    def _load_taxonomy(self) -> None:
        """Load and process taxonomy information."""
        self.lineages = pd.read_csv(self.data_paths['lineages'], sep='\t', header=None)
        self.lineages.columns = ['genome_id', 'taxonomy']
        self._process_taxonomy()
    
    def _process_taxonomy(self) -> None:
        """Process taxonomy for features in the data."""
        if self.table_metag_df is not None:
            metag_features = list(self.table_metag_df.columns)
            self.metag_taxonomy = self.lineages[
                self.lineages['genome_id'].isin(metag_features)
            ].copy()
            print(f"MetaG features with taxonomy: {len(self.metag_taxonomy)}")
        
        if self.table_metaT_df is not None:
            metaT_features = list(self.table_metaT_df.columns)
            self.metaT_taxonomy = self.lineages[
                self.lineages['genome_id'].isin(metaT_features)
            ].copy()
            print(f"MetaT features with taxonomy: {len(self.metaT_taxonomy)}")
    
    def apply_rclr_transformation(self, nan_replacement: float = 0.0) -> None:
        """
        Apply robust CLR (centered log-ratio) transformation using gemelli.
        
        Args:
            nan_replacement: Value to replace NaN results with.
        """
        try:
            from gemelli.preprocessing import matrix_rclr
        except ImportError:
            raise ImportError(
                "gemelli package required for RCLR transformation. "
                "Install with: pip install gemelli"
            )
        
        print("Applying RCLR transformation...")
        
        if self.table_metag is not None:
            table_metag_array = self.table_metag.matrix_data.toarray().T
            table_metag_rclr = matrix_rclr(table_metag_array)
            table_metag_rclr = np.nan_to_num(table_metag_rclr, nan=nan_replacement)
            
            self.table_metag_rclr = pd.DataFrame(
                table_metag_rclr,
                index=self.table_metag.ids('sample'),
                columns=self.table_metag.ids('observation')
            )
            print(f"MetaG RCLR shape: {self.table_metag_rclr.shape}")
        
        if self.table_metaT is not None:
            table_metaT_array = self.table_metaT.matrix_data.toarray().T
            table_metaT_rclr = matrix_rclr(table_metaT_array)
            table_metaT_rclr = np.nan_to_num(table_metaT_rclr, nan=nan_replacement)
            
            self.table_metaT_rclr = pd.DataFrame(
                table_metaT_rclr,
                index=self.table_metaT.ids('sample'),
                columns=self.table_metaT.ids('observation')
            )
            print(f"MetaT RCLR shape: {self.table_metaT_rclr.shape}")
    
    def apply_clr_transformation(
        self,
        pseudocount: float = 1.0,
        use_dataframes: bool = False
    ) -> None:
        """
        Apply standard CLR transformation (alternative to RCLR).
        
        Args:
            pseudocount: Pseudocount to add before log transformation.
            use_dataframes: If True, transform self.table_*_df instead of BIOM tables.
        """
        print(f"Applying CLR transformation (pseudocount={pseudocount})...")
        
        def clr_transform(data: np.ndarray) -> np.ndarray:
            data_pseudo = data + pseudocount
            log_data = np.log(data_pseudo)
            geometric_mean = np.exp(np.mean(log_data, axis=1, keepdims=True))
            return log_data - np.log(geometric_mean)
        
        if use_dataframes:
            if self.table_metag_df is not None:
                self.table_metag_rclr = pd.DataFrame(
                    clr_transform(self.table_metag_df.values),
                    index=self.table_metag_df.index,
                    columns=self.table_metag_df.columns
                )
                print(f"MetaG CLR shape: {self.table_metag_rclr.shape}")
            
            if self.table_metaT_df is not None:
                self.table_metaT_rclr = pd.DataFrame(
                    clr_transform(self.table_metaT_df.values),
                    index=self.table_metaT_df.index,
                    columns=self.table_metaT_df.columns
                )
                print(f"MetaT CLR shape: {self.table_metaT_rclr.shape}")
        else:
            # Use BIOM tables
            if self.table_metag is not None:
                data = self.table_metag.matrix_data.toarray().T
                self.table_metag_rclr = pd.DataFrame(
                    clr_transform(data),
                    index=self.table_metag.ids('sample'),
                    columns=self.table_metag.ids('observation')
                )
                print(f"MetaG CLR shape: {self.table_metag_rclr.shape}")
            
            if self.table_metaT is not None:
                data = self.table_metaT.matrix_data.toarray().T
                self.table_metaT_rclr = pd.DataFrame(
                    clr_transform(data),
                    index=self.table_metaT.ids('sample'),
                    columns=self.table_metaT.ids('observation')
                )
                print(f"MetaT CLR shape: {self.table_metaT_rclr.shape}")
    
    def filter_features(
        self,
        min_prevalence: float = 0.1,
        top_variance_pct: float = 0.20
    ) -> None:
        """
        Filter features based on prevalence and variance.
        
        Args:
            min_prevalence: Minimum fraction of samples where feature must be non-zero.
            top_variance_pct: Keep only top X% features by variance.
        """
        print(f"Filtering features (min_prevalence={min_prevalence}, "
              f"top_variance={top_variance_pct})...")
        
        if self.table_metag_rclr is not None:
            self.table_metag_filtered = self._filter_single_table(
                self.table_metag_rclr, min_prevalence, top_variance_pct, "MetaG"
            )
        
        if self.table_metaT_rclr is not None:
            self.table_metaT_filtered = self._filter_single_table(
                self.table_metaT_rclr, min_prevalence, top_variance_pct, "MetaT"
            )
    
    def _filter_single_table(
        self,
        table: pd.DataFrame,
        min_prevalence: float,
        top_variance_pct: float,
        name: str
    ) -> pd.DataFrame:
        """Filter a single feature table."""
        # Filter by prevalence
        prevalence = (table != 0).mean(axis=0)
        filtered = table.loc[:, prevalence >= min_prevalence]
        
        # Keep top variance features
        variance = filtered.var(axis=0)
        top_n = max(1, int(len(variance) * top_variance_pct))
        top_features = variance.nlargest(top_n).index
        filtered = filtered[top_features]
        
        print(f"{name} features after filtering: {filtered.shape[1]}")
        return filtered
    
    def prepare_datasets(
        self,
        label_column: str = 'diagnosis',
        test_size: float = 0.0,
        stratify: bool = True,
        random_state: int = 42
    ) -> None:
        """
        Prepare datasets for training.
        
        Args:
            label_column: Column in metadata containing labels.
            test_size: Fraction for test split (0 = no split).
            stratify: Whether to stratify split by labels.
            random_state: Random seed for reproducibility.
        """
        print("Preparing datasets...")
        
        # Find common samples
        sample_sets = []
        if self.table_metag_filtered is not None:
            sample_sets.append(set(self.table_metag_filtered.index))
        if self.table_metaT_filtered is not None:
            sample_sets.append(set(self.table_metaT_filtered.index))
        if self.metadata_df is not None:
            sample_sets.append(set(self.metadata_df.index))
        
        if len(sample_sets) == 0:
            raise ValueError("No data loaded. Call load_data and filter_features first.")
        
        self.common_samples = set.intersection(*sample_sets)
        print(f"Common samples: {len(self.common_samples)}")
        
        # Get sample list in consistent order
        sample_list = sorted(list(self.common_samples))
        
        # Filter to common samples
        if self.table_metag_filtered is not None:
            self.metag_data = self.table_metag_filtered.loc[sample_list]
        
        if self.table_metaT_filtered is not None:
            self.metaT_data = self.table_metaT_filtered.loc[sample_list]
        
        # Handle labels
        if self.metadata_df is not None and label_column in self.metadata_df.columns:
            self.labels = self.metadata_df.loc[sample_list, label_column]
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)
            self.label_names = self.label_encoder.classes_
            
            print(f"Label distribution after encoding:")
            for i, label in enumerate(self.label_names):
                count = np.sum(self.encoded_labels == i)
                print(f"  {label}: {count}")
        
        # Scale features
        if self.metag_data is not None:
            self.scaler_metag = StandardScaler()
            self.metag_scaled = self.scaler_metag.fit_transform(self.metag_data)
            print(f"MetaG scaled shape: {self.metag_scaled.shape}")
        
        if self.metaT_data is not None:
            self.scaler_metaT = StandardScaler()
            self.metaT_scaled = self.scaler_metaT.fit_transform(self.metaT_data)
            print(f"MetaT scaled shape: {self.metaT_scaled.shape}")
        
        # Create combined dataset
        if self.metag_scaled is not None and self.metaT_scaled is not None:
            self.combined_data = np.concatenate(
                [self.metag_scaled, self.metaT_scaled], axis=1
            )
            print(f"Combined shape: {self.combined_data.shape}")
    
    def get_data_for_training(
        self,
        data_type: str = 'metag'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get prepared data for VAE training.
        
        Args:
            data_type: One of 'metag', 'metat', or 'combined'.
        
        Returns:
            Tuple of (features, labels) where labels may be None.
        """
        if data_type == 'metag':
            if self.metag_scaled is None:
                raise ValueError("MetaG data not available. Run prepare_datasets first.")
            features = self.metag_scaled
        elif data_type == 'metat':
            if self.metaT_scaled is None:
                raise ValueError("MetaT data not available. Run prepare_datasets first.")
            features = self.metaT_scaled
        elif data_type == 'combined':
            if self.combined_data is None:
                raise ValueError("Combined data not available. Run prepare_datasets first.")
            features = self.combined_data
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        return features, self.encoded_labels
    
    def get_feature_names(self, data_type: str = 'metag') -> List[str]:
        """Get feature names for the specified data type."""
        if data_type == 'metag' and self.metag_data is not None:
            return list(self.metag_data.columns)
        elif data_type == 'metat' and self.metaT_data is not None:
            return list(self.metaT_data.columns)
        elif data_type == 'combined':
            names = []
            if self.metag_data is not None:
                names.extend([f"metag_{c}" for c in self.metag_data.columns])
            if self.metaT_data is not None:
                names.extend([f"metat_{c}" for c in self.metaT_data.columns])
            return names
        else:
            raise ValueError(f"No feature names available for data_type: {data_type}")
