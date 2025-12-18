"""
Taxonomy helper functions for microbiome data analysis.
"""

import pandas as pd
from typing import Optional, List, Dict


TAXONOMY_LEVELS = {
    'domain': 'd__',
    'phylum': 'p__',
    'class': 'c__',
    'order': 'o__',
    'family': 'f__',
    'genus': 'g__',
    'species': 's__'
}


def find_ogu_by_species(
    taxonomy_df: pd.DataFrame,
    species_name: str,
    feature_data: pd.DataFrame
) -> List[str]:
    """
    Find OGU IDs corresponding to a given species name.
    
    Args:
        taxonomy_df: DataFrame with 'genome_id' index and 'taxonomy' column.
        species_name: Species name to search for (case-insensitive).
        feature_data: DataFrame with feature columns.
    
    Returns:
        List of matching OGU IDs.
    """
    matching_ogus = []
    species_lower = species_name.lower()
    
    for ogu_id in taxonomy_df.index:
        if ogu_id in feature_data.columns:
            taxon = taxonomy_df.loc[ogu_id, 'taxonomy']
            if species_lower in taxon.lower():
                matching_ogus.append(ogu_id)
    
    return matching_ogus


def get_taxonomy_info(ogu_id: str, taxonomy_df: pd.DataFrame) -> str:
    """
    Get full taxonomy information for an OGU ID.
    
    Args:
        ogu_id: OGU identifier.
        taxonomy_df: DataFrame with taxonomy information.
    
    Returns:
        Taxonomy string or "Unknown taxonomy" if not found.
    """
    if ogu_id in taxonomy_df.index:
        return taxonomy_df.loc[ogu_id, 'taxonomy']
    return "Unknown taxonomy"


def extract_species_name(taxonomy_string: str) -> Optional[str]:
    """
    Extract species name from taxonomy string, falling back to genus or family.
    
    Args:
        taxonomy_string: Full taxonomy string (e.g., "d__Bacteria;...;s__Escherichia coli").
    
    Returns:
        Species name, or genus/family if species unavailable, or None if no valid name.
    """
    # Try species first
    if 's__' in taxonomy_string:
        species_part = taxonomy_string.split('s__')[-1]
        species = species_part.split(';')[0].strip()
        if species and species not in ('', 'unclassified', 'unidentified'):
            return species
    
    # Fall back to genus
    if 'g__' in taxonomy_string:
        genus_part = taxonomy_string.split('g__')[-1]
        genus = genus_part.split(';')[0].strip()
        if genus and genus not in ('', 'unclassified', 'unidentified'):
            return f"genus_{genus}"
    
    # Fall back to family
    if 'f__' in taxonomy_string:
        family_part = taxonomy_string.split('f__')[-1]
        family = family_part.split(';')[0].strip()
        if family and family not in ('', 'unclassified', 'unidentified'):
            return f"family_{family}"
    
    return None


def format_feature_name(
    feature_id: str,
    taxonomy_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Format feature name to show both OGU ID and species/genus/family name.
    
    Args:
        feature_id: OGU/feature identifier.
        taxonomy_df: DataFrame with taxonomy information.
    
    Returns:
        Formatted feature name with taxonomy annotation.
    """
    if taxonomy_df is not None and feature_id in taxonomy_df.index:
        taxonomy = taxonomy_df.loc[feature_id, 'taxonomy']
        species_name = extract_species_name(taxonomy)
        if species_name:
            return f"{feature_id} ({species_name})"
    return feature_id


def get_taxonomic_level(taxonomy_string: str, level: str = 'species') -> str:
    """
    Extract specific taxonomic level from taxonomy string.
    
    Args:
        taxonomy_string: Full taxonomy string.
        level: Taxonomic level to extract ('domain', 'phylum', 'class', 
               'order', 'family', 'genus', 'species').
    
    Returns:
        Name at specified taxonomic level or "unclassified" if not found.
    """
    if level in TAXONOMY_LEVELS:
        prefix = TAXONOMY_LEVELS[level]
        if prefix in taxonomy_string:
            level_part = taxonomy_string.split(prefix)[-1]
            return level_part.split(';')[0].strip()
    
    return "unclassified"


def parse_taxonomy(taxonomy_string: str) -> Dict[str, str]:
    """
    Parse full taxonomy string into a dictionary.
    
    Args:
        taxonomy_string: Full taxonomy string.
    
    Returns:
        Dictionary mapping level names to taxon names.
    """
    result = {}
    for level_name, prefix in TAXONOMY_LEVELS.items():
        result[level_name] = get_taxonomic_level(taxonomy_string, level_name)
    return result


def aggregate_by_taxonomy(
    feature_data: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    level: str = 'genus'
) -> pd.DataFrame:
    """
    Aggregate feature abundances by taxonomic level.
    
    Args:
        feature_data: DataFrame with samples as rows and OGUs as columns.
        taxonomy_df: DataFrame with 'genome_id' index and 'taxonomy' column.
        level: Taxonomic level for aggregation.
    
    Returns:
        DataFrame with samples as rows and taxonomic groups as columns.
    """
    # Get taxonomic assignments for each feature
    tax_assignments = {}
    for feature in feature_data.columns:
        if feature in taxonomy_df.index:
            tax_string = taxonomy_df.loc[feature, 'taxonomy']
            tax_assignments[feature] = get_taxonomic_level(tax_string, level)
        else:
            tax_assignments[feature] = 'unknown'
    
    # Create a copy and add taxonomy level column
    data_transposed = feature_data.T.copy()
    data_transposed['tax_level'] = data_transposed.index.map(tax_assignments)
    
    # Group by taxonomy level and sum
    aggregated = data_transposed.groupby('tax_level').sum().T
    
    return aggregated


def filter_by_taxonomy(
    feature_data: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    level: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter features by taxonomic criteria.
    
    Args:
        feature_data: DataFrame with samples as rows and features as columns.
        taxonomy_df: DataFrame with taxonomy information.
        level: Taxonomic level for filtering.
        include: List of taxa to include (if provided, only these are kept).
        exclude: List of taxa to exclude.
    
    Returns:
        Filtered DataFrame.
    """
    keep_features = []
    
    for feature in feature_data.columns:
        if feature in taxonomy_df.index:
            tax_string = taxonomy_df.loc[feature, 'taxonomy']
            taxon = get_taxonomic_level(tax_string, level).lower()
            
            if include is not None:
                if any(inc.lower() in taxon for inc in include):
                    keep_features.append(feature)
            elif exclude is not None:
                if not any(exc.lower() in taxon for exc in exclude):
                    keep_features.append(feature)
            else:
                keep_features.append(feature)
    
    return feature_data[keep_features]
