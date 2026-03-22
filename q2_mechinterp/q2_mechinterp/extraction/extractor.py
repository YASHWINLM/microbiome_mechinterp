"""
Feature extraction using trained VAE and Sparse Autoencoder models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
import time

from q2_mechinterp.core.sparse import SparseAutoencoder, sparse_autoencoder_loss
from q2_mechinterp.core.utils import clean_array


class FeatureExtractor:
    """
    Extract and analyze features using trained VAE and optional Sparse Autoencoder.
    
    Provides methods for:
    - Extracting latent features from VAE
    - Training sparse autoencoder on latent space
    - Analyzing feature importance
    - Mapping sparse features back to original features
    
    Args:
        vae_model: Trained VAE model.
        device: PyTorch device.
    
    Example:
        >>> extractor = FeatureExtractor(trained_vae, device)
        >>> latent = extractor.get_latent_features(data)
        >>> extractor.train_sparse_autoencoder(latent)
        >>> sparse = extractor.get_sparse_features(latent)
        >>> importance = extractor.analyze_feature_importance(feature_names)
    """
    
    def __init__(self, vae_model: nn.Module, device: torch.device):
        self.vae_model = vae_model
        self.device = device
        self.sparse_ae: Optional[SparseAutoencoder] = None
    
    def get_latent_features(
        self,
        data: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Extract latent features from VAE encoder.
        
        Args:
            data: Input data (samples x features).
            batch_size: Batch size for processing.
        
        Returns:
            Latent feature matrix (samples x latent_dim).
        """
        self.vae_model.eval()
        
        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data).to(self.device)
        else:
            data_tensor = data.to(self.device)
        
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        latent_vectors = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                mu, _ = self.vae_model.encode(batch_data)
                latent_vectors.append(mu.cpu().numpy())
        
        return np.vstack(latent_vectors)
    
    def train_sparse_autoencoder(
        self,
        latent_features: np.ndarray,
        sparse_multiplier: int = 3,
        epochs: int = 50,
        l1_reg: float = 0.001,
        lr: float = 1e-4,
        batch_size: int = 32,
        verbose: int = 10
    ) -> List[float]:
        """
        Train sparse autoencoder on VAE latent features.
        
        Args:
            latent_features: Latent features from VAE (samples x latent_dim).
            sparse_multiplier: Multiplier for sparse dimension.
            epochs: Training epochs.
            l1_reg: L1 regularization coefficient.
            lr: Learning rate.
            batch_size: Batch size.
            verbose: Print progress every N epochs.
        
        Returns:
            List of training losses.
        """
        # Validate input
        if np.isnan(latent_features).any() or np.isinf(latent_features).any():
            print("Warning: Latent features contain NaN/inf, cleaning...")
            valid_rows = ~(np.isnan(latent_features).any(axis=1) | 
                          np.isinf(latent_features).any(axis=1))
            latent_features = latent_features[valid_rows]
            print(f"Using {valid_rows.sum()} valid samples")
            
            if latent_features.shape[0] < 10:
                print("Error: Too few valid samples")
                return []
        
        latent_dim = latent_features.shape[1]
        sparse_dim = latent_dim * sparse_multiplier
        
        print(f"Training sparse autoencoder: {latent_dim} -> {sparse_dim}")
        
        self.sparse_ae = SparseAutoencoder(
            latent_dim, sparse_dim, l1_reg
        ).to(self.device)
        
        optimizer = optim.Adam(self.sparse_ae.parameters(), lr=lr)
        
        latent_tensor = torch.FloatTensor(latent_features).to(self.device)
        dataset = TensorDataset(latent_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.sparse_ae.train()
            epoch_loss = 0
            valid_batches = 0
            
            for batch_data in loader:
                data = batch_data[0]
                
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue
                
                optimizer.zero_grad()
                reconstructed = self.sparse_ae(data)
                
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    continue
                
                loss = sparse_autoencoder_loss(data, reconstructed, self.sparse_ae, l1_reg)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                # Check gradients
                valid_grads = True
                for param in self.sparse_ae.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_grads = False
                            break
                
                if not valid_grads:
                    continue
                
                torch.nn.utils.clip_grad_norm_(self.sparse_ae.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
            
            if valid_batches > 0:
                epoch_loss /= valid_batches
                losses.append(epoch_loss)
                
                if epoch % verbose == 0 or epoch == 1:
                    print(f"Sparse AE Epoch {epoch}: Loss: {epoch_loss:.6f}")
            else:
                losses.append(float('inf'))
                print(f"Sparse AE Epoch {epoch}: No valid batches")
        
        training_time = time.time() - start_time
        print(f"Sparse autoencoder training completed in {training_time:.2f}s")
        
        return losses
    
    def get_sparse_features(
        self,
        latent_features: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Extract sparse features from trained sparse autoencoder.
        
        Args:
            latent_features: Latent features from VAE.
            batch_size: Batch size for processing.
        
        Returns:
            Sparse feature matrix.
        """
        if self.sparse_ae is None:
            raise ValueError("Sparse autoencoder not trained. Call train_sparse_autoencoder first.")
        
        # Clean input
        if np.isnan(latent_features).any() or np.isinf(latent_features).any():
            latent_features = clean_array(latent_features)
        
        self.sparse_ae.eval()
        latent_tensor = torch.FloatTensor(latent_features).to(self.device)
        
        dataset = TensorDataset(latent_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        sparse_features = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                sparse = self.sparse_ae.get_sparse_features(batch_data)
                sparse_features.append(sparse.cpu().numpy())
        
        sparse_array = np.vstack(sparse_features)
        
        # Final cleanup
        if np.isnan(sparse_array).any() or np.isinf(sparse_array).any():
            sparse_array = clean_array(sparse_array)
        
        return sparse_array
    
    def analyze_feature_importance(
        self,
        original_feature_names: List[str],
        taxonomy_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze which original features are most important.
        
        Maps sparse autoencoder activations back through the VAE
        to identify important original features.
        
        Args:
            original_feature_names: List of original feature names.
            taxonomy_df: DataFrame with taxonomy info (optional, for microbiome data).
        
        Returns:
            Dictionary with importance analysis results.
        """
        if self.sparse_ae is None:
            raise ValueError("Sparse autoencoder not trained.")
        
        try:
            # Get weight matrices
            sparse_to_latent = self.sparse_ae.decoder.weight.data.cpu().numpy()
            latent_weights = self.vae_model.fc_mu.weight.data.cpu().numpy()
            
            # Get first encoder layer weights
            # Handle different VAE architectures:
            # - TranscriptomicsVAE uses encoder_layers (nn.ModuleList)
            # - ConditionalTranscriptomicsVAE uses encoder (nn.Sequential)
            encoder_layers = []
            if hasattr(self.vae_model, 'encoder_layers'):
                # TranscriptomicsVAE style: encoder_layers is a ModuleList of Linear layers
                for module in self.vae_model.encoder_layers:
                    if isinstance(module, nn.Linear):
                        encoder_layers.append(module)
            elif hasattr(self.vae_model, 'encoder'):
                # ConditionalTranscriptomicsVAE style: encoder is nn.Sequential
                for module in self.vae_model.encoder.modules():
                    if isinstance(module, nn.Linear):
                        encoder_layers.append(module)

            if len(encoder_layers) == 0:
                print("Warning: Could not find encoder layers. Model must have 'encoder_layers' (ModuleList) or 'encoder' (Sequential) attribute.")
                return {}

            first_encoder_weights = encoder_layers[0].weight.data.cpu().numpy()

            print(f"Weight shapes:")
            print(f"  Sparse -> Latent: {sparse_to_latent.shape}")
            print(f"  Latent weights (fc_mu): {latent_weights.shape}")
            print(f"  Number of encoder layers: {len(encoder_layers)}")
            for i, layer in enumerate(encoder_layers):
                print(f"  Encoder layer {i}: {layer.weight.shape}")
            print(f"  Original features: {len(original_feature_names)}")

            # Check dimensions
            if first_encoder_weights.shape[1] != len(original_feature_names):
                print(f"Warning: Dimension mismatch!")
                print(f"  Expected: {first_encoder_weights.shape[1]}")
                print(f"  Got: {len(original_feature_names)}")

                # Use minimum to avoid index errors
                n_features = min(first_encoder_weights.shape[1], len(original_feature_names))
                original_feature_names = original_feature_names[:n_features]

            # Compute importance scores
            # Sparse activation importance
            sparse_importance = np.abs(sparse_to_latent).mean(axis=0)

            # Chain weight matrices through all encoder layers to map latent -> input
            # Start with fc_mu transposed: (hidden_last, latent_dim)
            # Then multiply through encoder layers in reverse order
            # Final result maps from input_dim to latent_dim

            # Build the chained weight matrix from latent back to input
            # fc_mu: (latent_dim, hidden_last) -> transpose to (hidden_last, latent_dim)
            chained_weights = np.abs(latent_weights.T)

            # Multiply through encoder layers in reverse order
            for layer in reversed(encoder_layers):
                layer_weights = layer.weight.data.cpu().numpy()
                # layer_weights: (output_dim, input_dim) -> transpose to (input_dim, output_dim)
                chained_weights = np.abs(layer_weights.T) @ chained_weights

            # chained_weights is now (input_dim, latent_dim)
            # Take mean across latent dimensions to get per-input importance
            input_importance = chained_weights.mean(axis=1)

            # Truncate if needed due to earlier dimension mismatch
            if len(input_importance) > len(original_feature_names):
                input_importance = input_importance[:len(original_feature_names)]
            
            # Combined importance
            combined_importance = input_importance
            
            # Create results DataFrame
            importance_df = pd.DataFrame({
                'Feature': original_feature_names,
                'Importance': combined_importance,
                'Input_Weight_Importance': input_importance
            }).sort_values('Importance', ascending=False)
            
            # Add taxonomy if available
            if taxonomy_df is not None:
                from q2_mechinterp.microbiome.taxonomy import extract_species_name, get_taxonomic_level
                
                species_names = []
                genera = []
                
                for feature in importance_df['Feature']:
                    if feature in taxonomy_df.index:
                        tax_str = taxonomy_df.loc[feature, 'taxonomy']
                        species_names.append(extract_species_name(tax_str) or 'Unknown')
                        genera.append(get_taxonomic_level(tax_str, 'genus'))
                    else:
                        species_names.append('Unknown')
                        genera.append('Unknown')
                
                importance_df['Species'] = species_names
                importance_df['Genus'] = genera
            
            results = {
                'importance_df': importance_df,
                'sparse_importance': sparse_importance,
                'input_importance': input_importance,
                'weights': {
                    'sparse_to_latent': sparse_to_latent,
                    'latent_weights': latent_weights,
                    'first_encoder': first_encoder_weights,
                    'chained_weights': chained_weights
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            return {}
    
    def map_sparse_to_original(
        self,
        sparse_features: np.ndarray,
        original_feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Map sparse feature activations back to original features.
        
        Args:
            sparse_features: Sparse feature matrix (samples x sparse_dim).
            original_feature_names: List of original feature names.
        
        Returns:
            Dictionary with mapping results.
        """
        # Calculate feature activation statistics
        feature_activations = np.mean(np.abs(sparse_features), axis=0)
        
        # Calculate sparsity
        sparsity = 100 * (sparse_features == 0).mean()
        
        # Get top activated sparse features
        top_sparse_idx = np.argsort(feature_activations)[::-1][:20]
        
        # Analyze importance through decoder weights
        importance_results = self.analyze_feature_importance(original_feature_names)
        
        if 'importance_df' in importance_results:
            sparse_importance_df = importance_results['importance_df']
        else:
            # Fallback: uniform importance
            sparse_importance_df = pd.DataFrame({
                'Gene_OTU': original_feature_names,
                'Sparse_AE_Importance': np.ones(len(original_feature_names)) / len(original_feature_names)
            })
        
        results = {
            'feature_activations': feature_activations,
            'sparsity': sparsity,
            'top_sparse_indices': top_sparse_idx,
            'sparse_importance_df': sparse_importance_df,
            'n_active_features': (feature_activations > 0).sum()
        }
        
        return results
    
    def get_reconstruction(
        self,
        data: np.ndarray,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Get VAE reconstruction of input data.
        
        Args:
            data: Input data.
            batch_size: Batch size for processing.
        
        Returns:
            Reconstructed data.
        """
        self.vae_model.eval()
        
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        reconstructions = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                recon, _, _ = self.vae_model(batch_data)
                reconstructions.append(recon.cpu().numpy())
        
        return np.vstack(reconstructions)
    
    def save(self, path: str) -> None:
        """
        Save extractor state including sparse autoencoder.
        
        Args:
            path: Path to save file.
        """
        state = {
            'vae_state_dict': self.vae_model.state_dict(),
        }
        
        if self.sparse_ae is not None:
            state['sparse_ae_state_dict'] = self.sparse_ae.state_dict()
            state['sparse_ae_config'] = {
                'latent_dim': self.sparse_ae.latent_dim,
                'sparse_dim': self.sparse_ae.sparse_dim,
                'l1_reg': self.sparse_ae.l1_reg
            }
        
        torch.save(state, path)
        print(f"Saved extractor to {path}")
    
    def load_sparse_ae(
        self,
        path: str,
        latent_dim: Optional[int] = None,
        sparse_dim: Optional[int] = None
    ) -> None:
        """
        Load sparse autoencoder from file.
        
        Args:
            path: Path to saved file.
            latent_dim: Latent dimension (if not in saved config).
            sparse_dim: Sparse dimension (if not in saved config).
        """
        state = torch.load(path, map_location=self.device)
        
        if 'sparse_ae_config' in state:
            config = state['sparse_ae_config']
            latent_dim = config['latent_dim']
            sparse_dim = config['sparse_dim']
            l1_reg = config['l1_reg']
        elif latent_dim is None or sparse_dim is None:
            raise ValueError("Sparse autoencoder config not found and dimensions not provided")
        else:
            l1_reg = 0.001
        
        self.sparse_ae = SparseAutoencoder(
            latent_dim, sparse_dim, l1_reg
        ).to(self.device)
        
        if 'sparse_ae_state_dict' in state:
            self.sparse_ae.load_state_dict(state['sparse_ae_state_dict'])
        
        print(f"Loaded sparse autoencoder from {path}")
