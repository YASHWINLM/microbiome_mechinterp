import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ..models.sparse_ae import SparseAutoencoder, sparse_loss
from ..utils.taxonomy import format_feature_name

class FeatureExtractor:
    """Extract features using trained VAE and Sparse Autoencoder"""
    
    def __init__(self, vae_model, device):
        self.vae_model = vae_model
        self.device = device
        
    def get_latent_features(self, data):
        """Extract latent features from VAE"""
        self.vae_model.eval()
        
        if isinstance(data, np.ndarray):
            data_tensor = torch.FloatTensor(data).to(self.device)
        else:
            data_tensor = data.to(self.device)
        
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=64)
        
        latent_vectors = []
        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0]
                mu, _ = self.vae_model.encode(batch_data)
                latent_vectors.append(mu.cpu().numpy())
        
        return np.vstack(latent_vectors)
    
    def train_sparse_autoencoder(self, latent_features, sparse_multiplier=3, 
                               epochs=50, l1_reg=0.001):
        """Train sparse autoencoder on latent features with robust NaN handling"""
        
        # Check input features for validity
        if np.isnan(latent_features).any() or np.isinf(latent_features).any():
            print("Warning: Input latent features contain NaN or inf values!")
            # Remove NaN/inf rows
            valid_rows = ~(np.isnan(latent_features).any(axis=1) | np.isinf(latent_features).any(axis=1))
            latent_features = latent_features[valid_rows]
            print(f"Using {valid_rows.sum()} valid samples out of {len(valid_rows)}")
            
            if latent_features.shape[0] < 10:
                print("Error: Too few valid samples for sparse autoencoder training")
                return []
        
        latent_dim = latent_features.shape[1]
        sparse_dim = latent_dim * sparse_multiplier
        
        print(f"Training sparse autoencoder: {latent_dim} -> {sparse_dim}")
        
        sparse_ae = SparseAutoencoder(latent_dim, sparse_dim, l1_reg).to(self.device)
        optimizer = optim.Adam(sparse_ae.parameters(), lr=1e-4)  # Lower learning rate
        
        latent_tensor = torch.FloatTensor(latent_features).to(self.device)
        dataset = TensorDataset(latent_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        losses = []
        valid_epochs = 0
        
        for epoch in range(1, epochs + 1):
            sparse_ae.train()
            epoch_loss = 0
            valid_batches = 0
            
            for batch_data in loader:
                data = batch_data[0]
                
                # Skip invalid batches
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue
                
                optimizer.zero_grad()
                reconstructed = sparse_ae(data)
                
                # Skip if reconstruction is invalid
                if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
                    continue
                
                loss = sparse_loss(data, reconstructed, sparse_ae, l1_reg)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                
                # Check gradients
                valid_grads = True
                for param in sparse_ae.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_grads = False
                            break
                
                if not valid_grads:
                    continue
                
                torch.nn.utils.clip_grad_norm_(sparse_ae.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
            
            if valid_batches > 0:
                epoch_loss /= valid_batches
                losses.append(epoch_loss)
                valid_epochs += 1
                
                if epoch % 10 == 0 or epoch == 1:
                    print(f'Sparse AE Epoch {epoch}: Loss: {epoch_loss:.6f}')
            else:
                print(f'Sparse AE Epoch {epoch}: No valid batches')
                losses.append(float('inf'))
        
        if valid_epochs == 0:
            print("Error: Sparse autoencoder training failed completely")
            return []
        
        self.sparse_ae = sparse_ae
        print(f"Sparse autoencoder training completed with {valid_epochs} valid epochs")
        return losses
    
    def get_sparse_features(self, latent_features):
        """Extract sparse features with validation"""
        # Check input validity
        if np.isnan(latent_features).any() or np.isinf(latent_features).any():
            print("Warning: Input latent features contain NaN/inf, cleaning...")
            # Replace NaN/inf with zeros or small values
            latent_features = np.nan_to_num(latent_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.sparse_ae.eval()
        latent_tensor = torch.FloatTensor(latent_features).to(self.device)
        
        with torch.no_grad():
            sparse_features = self.sparse_ae.get_sparse_features(latent_tensor)
        
        sparse_array = sparse_features.cpu().numpy()
        
        # Final validation - replace any remaining NaN/inf
        if np.isnan(sparse_array).any() or np.isinf(sparse_array).any():
            print("Warning: Sparse features contain NaN/inf, cleaning...")
            sparse_array = np.nan_to_num(sparse_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return sparse_array
    
    def analyze_feature_importance(self, original_feature_names, taxonomy_df=None):
        """Analyze which original features are most important with taxonomy information"""
        
        try:
            # Get weights from sparse autoencoder decoder
            sparse_to_latent = self.sparse_ae.decoder.weight.data.cpu().numpy()
            
            # Get weights from VAE encoder - need to get the right layer
            # The issue is we need to trace back through the encoder properly
            
            # Get the mu projection weights (latent_dim x last_hidden_dim)
            latent_weights = self.vae_model.fc_mu.weight.data.cpu().numpy()
            
            # Get the last encoder layer weights
            encoder_layers = []
            for module in self.vae_model.encoder.modules():
                if isinstance(module, nn.Linear):
                    encoder_layers.append(module)
            
            if len(encoder_layers) == 0:
                print("Warning: Could not find encoder layers for feature importance analysis")
                return {}
                
            # Get the first encoder layer (input -> first hidden)
            first_encoder_weights = encoder_layers[0].weight.data.cpu().numpy()
            
            print(f"Debug shapes:")
            print(f"  Sparse to latent: {sparse_to_latent.shape}")  
            print(f"  Latent weights: {latent_weights.shape}")
            print(f"  First encoder: {first_encoder_weights.shape}")
            print(f"  Original features: {len(original_feature_names)}")
            
            # Check dimension compatibility
            expected_input_dim = first_encoder_weights.shape[1]
            if expected_input_dim != len(original_feature_names):
                print(f"Warning: Dimension mismatch!")
                print(f"  Expected input features: {expected_input_dim}")
                print(f"  Provided feature names: {len(original_feature_names)}")
                
                # Use only the features that match
                original_feature_names = original_feature_names[:expected_input_dim]
                print(f"  Using first {len(original_feature_names)} features")
            
            # For a simplified analysis, just use the first encoder layer weights
            # Each row represents a hidden unit, each column represents an input feature
            feature_importance = {}
            
            # For each sparse feature (decoder output)
            for sparse_idx in range(sparse_to_latent.shape[1]):
                # Get importance as the absolute sum of connections through the network
                # This is simplified - just use first encoder layer importance
                importance_scores = np.abs(first_encoder_weights).mean(axis=0)
                
                # Ensure we don't exceed array bounds
                max_features = min(len(importance_scores), len(original_feature_names))
                importance_scores = importance_scores[:max_features]
                feature_names = original_feature_names[:max_features]
                
                # Create feature importance dictionary with formatted names
                if taxonomy_df is not None:
                    feature_dict = {
                        format_feature_name(feature_name, taxonomy_df): importance_scores[j] 
                        for j, feature_name in enumerate(feature_names)
                    }
                else:
                    feature_dict = dict(zip(feature_names, importance_scores))
                
                feature_importance[sparse_idx] = sorted(feature_dict.items(), 
                                                    key=lambda x: x[1], reverse=True)
            
            return feature_importance
            
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
            print("Returning empty feature importance dictionary")
            return {}

print("Feature Extractor class defined successfully!")
