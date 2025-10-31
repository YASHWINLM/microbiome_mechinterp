import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..utils.taxonomy import extract_species_name, get_taxonomic_level
import os

def map_sparse_features_to_otus(extractor, sparse_features, gene_names, approach_name="features"):
    """Map sparse features back to original OTUs/genes using the method from your example"""
    
    print(f"\n{'='*60}")
    print(f"MAPPING SPARSE FEATURES TO OTUs - {approach_name.upper()}")
    print(f"{'='*60}")
    
    # Calculate sparsity metrics
    sparsity = (sparse_features == 0).mean() * 100
    print(f"Sparsity of {approach_name} encoded features: {sparsity:.2f}%")
    
    # Get model weights
    sparse_ae_weights = extractor.sparse_ae.encoder[0].weight.data.cpu().numpy()
    vae_encoder_weights = extractor.vae_model.fc_mu.weight.data.cpu().numpy()
    sparse_dim = sparse_features.shape[1]
    
    print(f"Model architecture:")
    print(f"  Sparse AE encoder weights shape: {sparse_ae_weights.shape}")
    print(f"  VAE encoder (mu) weights shape: {vae_encoder_weights.shape}")
    print(f"  Sparse dimensions: {sparse_dim}")
    print(f"  Original gene/OTU count: {len(gene_names)}")
    
    print(f"Analyzing sparse feature importance...")
    
    # Calculate feature importance for each sparse feature
    feature_importance = {}
    for i in range(sparse_dim):
        # Get weights for this sparse feature (from sparse AE decoder)
        sparse_to_latent = extractor.sparse_ae.decoder.weight[:, i].cpu().detach().numpy()
        
        # Multiply with VAE encoder weights to get effect on original genes
        # This maps: sparse feature -> latent space -> original genes/OTUs
        gene_importance = np.abs(np.dot(vae_encoder_weights.T, sparse_to_latent))
        
        # Map to gene names
        feature_importance[i] = dict(zip(gene_names, gene_importance))
    
    # Find top genes/OTUs for each sparse feature
    top_genes_per_feature = {}
    for feature_idx, gene_scores in feature_importance.items():
        # Sort genes by importance score
        sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
        # Keep top 10 genes/OTUs
        top_genes_per_feature[feature_idx] = sorted_genes[:10]
    
    # Calculate feature activations across samples
    feature_activations = sparse_features.mean(axis=0)
    top_activated_features = np.argsort(feature_activations)[-5:]
    
    print(f"\nTop OTUs/genes for most activated sparse features in {approach_name}:")
    for i, feature_idx in enumerate(top_activated_features):
        print(f"\nSparse feature {feature_idx} (activation: {feature_activations[feature_idx]:.4f}):")
        for gene, score in top_genes_per_feature[feature_idx]:
            print(f"  {gene}: {score:.4f}")
    
    # Calculate overall gene importance (weighted by activation)
    gene_overall_importance = {gene: 0 for gene in gene_names}
    for feature_idx, gene_scores in feature_importance.items():
        activation = feature_activations[feature_idx]
        for gene, score in gene_scores.items():
            gene_overall_importance[gene] += score * activation
    
    # Create importance dataframe
    sparse_importance_df = pd.DataFrame({
        'Gene_OTU': list(gene_overall_importance.keys()),
        'Sparse_AE_Importance': list(gene_overall_importance.values())
    })
    sparse_importance_df = sparse_importance_df.sort_values('Sparse_AE_Importance', ascending=False)
    
    print(f"\nTop 20 OTUs/genes by Sparse Autoencoder importance in {approach_name}:")
    print(sparse_importance_df.head(20))
    
    # Save results
    output_file = f'sparse_importance_{approach_name.lower()}.csv'
    sparse_importance_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    print(f"{'='*60}")
    print(f"SPARSE TO OTU MAPPING COMPLETE - {approach_name.upper()}")
    print(f"{'='*60}\n")
    
    return {
        'feature_importance': feature_importance,
        'top_genes_per_feature': top_genes_per_feature,
        'feature_activations': feature_activations,
        'sparse_importance_df': sparse_importance_df,
        'sparsity': sparsity
    }

def map_sparse_features_with_taxonomy(extractor, sparse_features, gene_names, taxonomy_df, approach_name="features"):
    """Enhanced mapping with taxonomy information"""
    
    print(f"\n{'='*60}")
    print(f"ENHANCED MAPPING WITH TAXONOMY - {approach_name.upper()}")
    print(f"{'='*60}")
    
    # Get base mapping
    base_results = map_sparse_features_to_otus(extractor, sparse_features, gene_names, approach_name)
    
    # Add taxonomy information
    enhanced_importance = []
    for _, row in base_results['sparse_importance_df'].iterrows():
        ogu_id = row['Gene_OTU']
        importance = row['Sparse_AE_Importance']
        
        if ogu_id in taxonomy_df.index:
            taxonomy = taxonomy_df.loc[ogu_id, 'taxonomy']
            species = extract_species_name(taxonomy)
            genus = get_taxonomic_level(taxonomy, 'genus')
            family = get_taxonomic_level(taxonomy, 'family')
        else:
            taxonomy = "Unknown"
            species = "Unknown"
            genus = "Unknown"
            family = "Unknown"
        
        enhanced_importance.append({
            'Gene_OTU': ogu_id,
            'Sparse_AE_Importance': importance,
            'Species': species if species else "Unknown",
            'Genus': genus,
            'Family': family,
            'Full_Taxonomy': taxonomy
        })
    
    enhanced_importance_df = pd.DataFrame(enhanced_importance)
    
    # Save enhanced results
    output_file = f'sparse_importance_with_taxonomy_{approach_name.lower()}.csv'
    enhanced_importance_df.to_csv(output_file, index=False)
    print(f"Enhanced results saved to {output_file}")
    
    # Print top species
    print(f"\nTop species by importance:")
    species_importance = enhanced_importance_df.groupby('Species')['Sparse_AE_Importance'].sum().sort_values(ascending=False)
    print(species_importance.head(10))
    
    return enhanced_importance_df

def visualize_sparse_mappings(mapping_results, approach_name="features", save_dir="figs"):
    """Visualize the sparse feature mappings"""
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating visualizations for {approach_name} sparse mappings...")
    
    # Plot 1: Feature activation distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sparsity histogram
    activations = mapping_results['feature_activations']
    axes[0, 0].hist(activations, bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title(f'{approach_name} Feature Activation Distribution')
    axes[0, 0].set_xlabel('Mean Activation')
    axes[0, 0].set_ylabel('Frequency')
    
    # Top 20 most important OTUs
    top_20 = mapping_results['sparse_importance_df'].head(20)
    axes[0, 1].barh(range(len(top_20)), top_20['Sparse_AE_Importance'])
    axes[0, 1].set_yticks(range(len(top_20)))
    axes[0, 1].set_yticklabels([f"{otu[:15]}..." if len(otu) > 15 else otu for otu in top_20['Gene_OTU']])
    axes[0, 1].set_title(f'Top 20 Most Important {approach_name} OTUs')
    axes[0, 1].set_xlabel('Importance Score')
    axes[0, 1].invert_yaxis()
    
    # Sparsity level
    sparsity = mapping_results['sparsity']
    axes[1, 0].pie([sparsity, 100-sparsity], labels=[f'Zero ({sparsity:.1f}%)', f'Non-zero ({100-sparsity:.1f}%)'], 
                   autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title(f'{approach_name} Feature Sparsity')
    
    # Feature importance distribution
    importance_values = mapping_results['sparse_importance_df']['Sparse_AE_Importance']
    axes[1, 1].hist(importance_values, bins=50, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title(f'{approach_name} OTU Importance Distribution')
    axes[1, 1].set_xlabel('Importance Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'sparse_mapping_analysis_{approach_name.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()
