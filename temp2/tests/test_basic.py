import pytest
import numpy as np
import torch
from microbiome_mechinterp.models.vae import VAE, VAELoss
from microbiome_mechinterp.models.sparse_ae import SparseAutoencoder, sparse_loss

def test_vae_forward():
    """Test VAE forward pass"""
    model = VAE(input_dim=100, latent_dim=10)
    x = torch.randn(5, 100)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (5, 10)
    assert logvar.shape == (5, 10)

def test_vae_loss():
    """Test VAE loss computation"""
    x = torch.randn(5, 100)
    recon_x = torch.randn(5, 100)
    mu = torch.randn(5, 10)
    logvar = torch.randn(5, 10)
    
    total_loss, recon_loss, kl_loss, l1_loss = VAELoss(recon_x, x, mu, logvar)
    assert isinstance(total_loss.item(), float)
    assert isinstance(recon_loss.item(), float)
    assert isinstance(kl_loss.item(), float)
    assert isinstance(l1_loss.item(), float)

def test_sparse_ae():
    """Test Sparse Autoencoder"""
    model = SparseAutoencoder(input_dim=50, hidden_dim=150)
    x = torch.randn(10, 50)
    recon = model(x)
    assert recon.shape == x.shape
    
    sparse_features = model.get_sparse_features(x)
    assert sparse_features.shape == (10, 150)
