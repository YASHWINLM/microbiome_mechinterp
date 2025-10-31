import pandas as pd
import numpy as np
import biom
from qiime2 import Metadata
from gemelli.preprocessing import matrix_rclr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..utils.helpers import get_device


class MicrobiomeDataProcessor:
    """Class to handle microbiome data loading and preprocessing"""

    def __init__(self, data_paths, device=None):
        self.data_paths = data_paths
        self.device = device or get_device()

    def load_data(self):
        """Load microbiome data from biom files and metadata"""
        print("Loading microbiome data...")

        # Load biom tables
        self.table_metag = biom.load_table(self.data_paths["metag_biom"])
        self.table_metaT = biom.load_table(self.data_paths["metat_biom"])

        # Convert to dataframes
        self.table_metag_df = self.table_metag.to_dataframe().T
        self.table_metaT_df = self.table_metaT.to_dataframe().T

        # Load metadata
        self.metadata = Metadata.load(self.data_paths["metadata"])
        self.metadata_df = self.metadata.to_dataframe()

        # Load lineages if available
        if "lineages" in self.data_paths:
            self.lineages = pd.read_csv(
                self.data_paths["lineages"], sep="\t", header=None
            )
            self.lineages.columns = ["genome_id", "taxonomy"]
            self.process_taxonomy()

        print(f"MetaG shape: {self.table_metag_df.shape}")
        print(f"MetaT shape: {self.table_metaT_df.shape}")
        print(f"Metadata shape: {self.metadata_df.shape}")
        print(
            f"Diagnosis distribution:\n{self.metadata_df['diagnosis'].value_counts()}"
        )

    def process_taxonomy(self):
        """Process taxonomy information for features"""
        metag_features = list(self.table_metag_df.columns)
        metaT_features = list(self.table_metaT_df.columns)

        # Filter for features in the data
        self.metag_taxonomy = self.lineages[
            self.lineages["genome_id"].isin(metag_features)
        ]
        self.metaT_taxonomy = self.lineages[
            self.lineages["genome_id"].isin(metaT_features)
        ]

        print(
            f"Taxonomy loaded: MetaG={len(self.metag_taxonomy)}, MetaT={len(self.metaT_taxonomy)}"
        )

    def apply_rclr_transformation(self):
        """Apply robust centered log-ratio transformation"""
        print("Applying RCLR transformation...")

        # MetaG RCLR
        table_metag_array = self.table_metag.matrix_data.toarray().T
        table_metag_rclr = matrix_rclr(table_metag_array)
        self.table_metag_rclr = pd.DataFrame(
            np.nan_to_num(table_metag_rclr, nan=0.0),
            index=self.table_metag.ids("sample"),
            columns=self.table_metag.ids("observation"),
        )

        # MetaT RCLR
        table_metaT_array = self.table_metaT.matrix_data.toarray().T
        table_metaT_rclr = matrix_rclr(table_metaT_array)
        self.table_metaT_rclr = pd.DataFrame(
            np.nan_to_num(table_metaT_rclr, nan=0.0),
            index=self.table_metaT.ids("sample"),
            columns=self.table_metaT.ids("observation"),
        )

        print("RCLR transformation completed")

    def filter_features(self, min_prevalence=0.1, top_variance_pct=0.20):
        """Filter features by prevalence and variance"""
        print("Filtering features...")

        # MetaG filtering
        metag_prevalence = (self.table_metag_rclr != 0).mean(axis=0)
        metag_filtered = self.table_metag_rclr.loc[
            :, metag_prevalence >= min_prevalence
        ]
        metag_variance = metag_filtered.var(axis=0)
        top_n = int(len(metag_variance) * top_variance_pct)
        self.table_metag_filtered = metag_filtered[metag_variance.nlargest(top_n).index]

        # MetaT filtering
        metaT_prevalence = (self.table_metaT_rclr != 0).mean(axis=0)
        metaT_filtered = self.table_metaT_rclr.loc[
            :, metaT_prevalence >= min_prevalence
        ]
        metaT_variance = metaT_filtered.var(axis=0)
        top_n = int(len(metaT_variance) * top_variance_pct)
        self.table_metaT_filtered = metaT_filtered[metaT_variance.nlargest(top_n).index]

        print(f"Filtered MetaG: {self.table_metag_filtered.shape}")
        print(f"Filtered MetaT: {self.table_metaT_filtered.shape}")

    def prepare_datasets(self):
        """Prepare final datasets for training"""
        print("Preparing datasets...")

        # Find common samples
        self.common_samples = (
            set(self.table_metag_filtered.index)
            & set(self.table_metaT_filtered.index)
            & set(self.metadata_df.index)
        )
        self.common_samples = sorted(list(self.common_samples))

        # Subset data
        self.metag_data = self.table_metag_filtered.loc[self.common_samples]
        self.metaT_data = self.table_metaT_filtered.loc[self.common_samples]
        self.labels = self.metadata_df.loc[self.common_samples, "diagnosis"]

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.label_names = self.label_encoder.classes_

        # Scale data
        self.scaler_metag = StandardScaler()
        self.scaler_metaT = StandardScaler()
        self.metag_scaled = self.scaler_metag.fit_transform(self.metag_data)
        self.metaT_scaled = self.scaler_metaT.fit_transform(self.metaT_data)

        # trying to fix training instability issues
        # self.metag_scaled = np.clip(self.metag_scaled, -10, 10)
        # self.metaT_scaled = np.clip(self.metaT_scaled, -10, 10)

        print(f"MetaG range: [{self.metag_scaled.min()}, {self.metag_scaled.max()}]")
        print(f"MetaT range: [{self.metaT_scaled.min()}, {self.metaT_scaled.max()}]")

        self.combined_data = np.concatenate(
            [self.metag_scaled, self.metaT_scaled], axis=1
        )

        print(f"Final dataset shapes:")
        print(f"  MetaG scaled: {self.metag_scaled.shape}")
        print(f"  MetaT scaled: {self.metaT_scaled.shape}")
        print(f"  Combined: {self.combined_data.shape}")
        print(f"  Labels: {self.encoded_labels.shape}")
        print(f"  Label names: {self.label_names}")
