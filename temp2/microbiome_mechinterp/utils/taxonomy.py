"""Taxonomy helper functions for microbiome data"""

def find_ogu_by_species(taxonomy_df, species_name, feature_data):
    """Find OGU IDs corresponding to a given species name"""
    matching_ogus = []
    for ogu_id in taxonomy_df.index:
        if ogu_id in feature_data.columns:
            taxon = taxonomy_df.loc[ogu_id, 'taxonomy']
            if species_name.lower() in taxon.lower():
                matching_ogus.append(ogu_id)
    return matching_ogus

def get_taxonomy_info(ogu_id, taxonomy_df):
    """Get full taxonomy information for an OGU ID"""
    if ogu_id in taxonomy_df.index:
        return taxonomy_df.loc[ogu_id, 'taxonomy']
    return "Unknown taxonomy"

def extract_species_name(taxonomy_string):
    """Extract species name from taxonomy string"""
    if 's__' in taxonomy_string:
        species_part = taxonomy_string.split('s__')[-1]
        species = species_part.split(';')[0].strip()
        if species and species != '' and species != 'unclassified' and species != 'unidentified':
            return species
    
    if 'g__' in taxonomy_string:
        genus_part = taxonomy_string.split('g__')[-1]
        genus = genus_part.split(';')[0].strip()
        if genus and genus != '' and genus != 'unclassified' and genus != 'unidentified':
            return f"genus_{genus}"
    
    if 'f__' in taxonomy_string:
        family_part = taxonomy_string.split('f__')[-1]
        family = family_part.split(';')[0].strip()
        if family and family != '' and family != 'unclassified' and family != 'unidentified':
            return f"family_{family}"
    
    return None

def format_feature_name(feature_id, taxonomy_df=None):
    """Format feature name to show both OGU ID and species/genus/family name"""
    if taxonomy_df is not None and feature_id in taxonomy_df.index:
        taxonomy = taxonomy_df.loc[feature_id, 'taxonomy']
        species_name = extract_species_name(taxonomy)
        if species_name:
            return f"{feature_id} ({species_name})"
    return feature_id

def get_taxonomic_level(taxonomy_string, level='species'):
    """Extract specific taxonomic level from taxonomy string"""
    level_prefixes = {
        'domain': 'd__',
        'phylum': 'p__',
        'class': 'c__', 
        'order': 'o__',
        'family': 'f__',
        'genus': 'g__',
        'species': 's__'
    }
    
    if level in level_prefixes:
        prefix = level_prefixes[level]
        if prefix in taxonomy_string:
            level_part = taxonomy_string.split(prefix)[-1]
            return level_part.split(';')[0].strip()
    
    return "unclassified"
