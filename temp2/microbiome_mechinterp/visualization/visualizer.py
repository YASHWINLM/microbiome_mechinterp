import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
import os

class Visualizer:
    """Visualization tools for microbiome analysis"""
    
    def __init__(self, save_dir='figs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_latent_space(self, latent_vectors, labels, label_names, title="Latent Space"):
        """Visualize latent space with UMAP"""
        print("Computing UMAP embedding...")
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_embedding = umap_reducer.fit_transform(latent_vectors)
        
        # Create visualization DataFrame
        vis_df = pd.DataFrame({
            'UMAP_1': umap_embedding[:, 0],
            'UMAP_2': umap_embedding[:, 1],
            'Label': [label_names[l] for l in labels]
        })
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # UMAP
        for label in label_names:
            mask = vis_df['Label'] == label
            ax.scatter(vis_df.loc[mask, 'UMAP_1'], vis_df.loc[mask, 'UMAP_2'], 
                      label=label, alpha=0.7, s=20)
        ax.set_title(f'UMAP - {title}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        safe_title = title.replace(" ", "_").lower()
        save_path = os.path.join(self.save_dir, f'umap_{safe_title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")
        plt.show()
        
        return vis_df
    
    def visualize_sparse_features(self, sparse_features, labels, label_names, title="Sparse Features"):
        """Visualize sparse features with UMAP"""
        print("Computing UMAP for sparse features...")
        
        # Check sparsity
        sparsity = (sparse_features == 0).mean() * 100
        print(f"Sparsity: {sparsity:.2f}%")
        
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_embedding = umap_reducer.fit_transform(sparse_features)
        
        # Create visualization DataFrame
        vis_df = pd.DataFrame({
            'UMAP_1': umap_embedding[:, 0],
            'UMAP_2': umap_embedding[:, 1],
            'Label': [label_names[l] for l in labels]
        })
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        for label in label_names:
            mask = vis_df['Label'] == label
            ax.scatter(vis_df.loc[mask, 'UMAP_1'], vis_df.loc[mask, 'UMAP_2'], 
                      label=label, alpha=0.7, s=20)
        ax.set_title(f'Sparse Features UMAP - {title}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        safe_title = title.replace(" ", "_").lower()
        save_path = os.path.join(self.save_dir, f'sparse_umap_{safe_title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sparse features UMAP plot saved to {save_path}")
        plt.show()
        
        return vis_df
    
    def plot_feature_importance(self, importance_df, top_n=20, title="Feature Importance"):
        """Plot feature importance"""
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_features.iloc[:, 1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([str(f)[:30] + '...' if len(str(f)) > 30 else str(f) 
                           for f in top_features.iloc[:, 0]])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} {title}')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save figure
        safe_title = title.replace(" ", "_").lower()
        save_path = os.path.join(self.save_dir, f'importance_{safe_title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        plt.show()
    
    def plot_training_curves(self, train_losses, val_losses=None, title="Training Curves"):
        """Plot training curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', alpha=0.7)
        if val_losses is not None and len(val_losses) > 0:
            ax.plot(val_losses, label='Validation Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        safe_title = title.replace(" ", "_").lower()
        save_path = os.path.join(self.save_dir, f'training_{safe_title}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.show()

print("Visualizer class defined successfully!")
