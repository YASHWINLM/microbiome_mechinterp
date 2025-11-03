import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning interpretable features"""
    
    def __init__(self, input_dim, hidden_dim, l1_reg=0.001):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_reg = l1_reg
        
        # Single layer encoder/decoder for sparsity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True)
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
    def forward(self, x):
        """Forward pass through sparse autoencoder"""
        hidden = F.relu(self.encoder(x))
        reconstructed = self.decoder(hidden)
        return reconstructed
    
    def get_sparse_features(self, x):
        """Get sparse feature activations"""
        return F.relu(self.encoder(x))

def sparse_loss(x, reconstructed, sparse_ae, l1_reg):
    """Compute sparse autoencoder loss with L1 regularization"""
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed, x)
    
    # L1 regularization on activations
    sparse_features = sparse_ae.get_sparse_features(x)
    l1_loss = l1_reg * torch.mean(torch.abs(sparse_features))
    
    return recon_loss + l1_loss
