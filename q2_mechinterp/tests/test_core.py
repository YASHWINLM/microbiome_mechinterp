"""Unit tests for q2_mechinterp package."""

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Core imports
from q2_mechinterp.core.vae import BaseVAE, ConcreteVAE, vae_loss
from q2_mechinterp.core.sparse import (
    SparseAutoencoder, 
    TopKSparseAutoencoder,
    DeepSparseAutoencoder
)
from q2_mechinterp.core.utils import get_device, set_seed, EarlyStopping


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        a1 = torch.rand(10)
        set_seed(42)
        a2 = torch.rand(10)
        assert torch.allclose(a1, a2)
    
    def test_early_stopping_improvement(self):
        """Test early stopping detects improvement."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        # Improving losses
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)
        assert es.counter == 0
    
    def test_early_stopping_patience(self):
        """Test early stopping triggers after patience."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        es(1.0)  # Best
        es(1.1)  # Worse, counter=1
        es(1.2)  # Worse, counter=2
        assert not es(1.3)  # Worse, counter=3, not triggered yet
        assert es(1.4)  # Worse, counter=4 > patience, triggered


class TestVAE:
    """Test VAE implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        return torch.FloatTensor(X)
    
    @pytest.fixture
    def simple_vae(self):
        """Create a simple VAE for testing."""
        return ConcreteVAE(
            input_dim=50,
            latent_dim=10,
            hidden_dims=[32, 16]
        )
    
    def test_vae_forward(self, simple_vae, sample_data):
        """Test VAE forward pass."""
        x_recon, mu, logvar = simple_vae(sample_data)
        
        assert x_recon.shape == sample_data.shape
        assert mu.shape == (100, 10)
        assert logvar.shape == (100, 10)
    
    def test_vae_encode(self, simple_vae, sample_data):
        """Test VAE encoding."""
        mu, logvar = simple_vae.encode(sample_data)
        
        assert mu.shape == (100, 10)
        assert logvar.shape == (100, 10)
    
    def test_vae_decode(self, simple_vae):
        """Test VAE decoding."""
        z = torch.randn(10, 10)
        x_recon = simple_vae.decode(z)
        
        assert x_recon.shape == (10, 50)
    
    def test_vae_reparameterize(self, simple_vae):
        """Test reparameterization trick."""
        mu = torch.zeros(10, 10)
        logvar = torch.zeros(10, 10)
        
        z = simple_vae.reparameterize(mu, logvar)
        assert z.shape == (10, 10)
    
    def test_vae_sample(self, simple_vae):
        """Test sampling from VAE."""
        samples = simple_vae.sample(5, device='cpu')
        assert samples.shape == (5, 50)
    
    def test_vae_loss(self, simple_vae, sample_data):
        """Test VAE loss computation."""
        x_recon, mu, logvar = simple_vae(sample_data)
        loss, recon_loss, kl_loss = vae_loss(x_recon, sample_data, mu, logvar, beta=1.0)
        
        assert loss.dim() == 0  # Scalar
        assert recon_loss.dim() == 0
        assert kl_loss.dim() == 0
        assert loss >= 0


class TestSparseAutoencoder:
    """Test sparse autoencoder implementations."""
    
    @pytest.fixture
    def sample_latent(self):
        """Generate sample latent data."""
        return torch.randn(100, 20)
    
    def test_sparse_ae_forward(self, sample_latent):
        """Test sparse autoencoder forward pass."""
        sae = SparseAutoencoder(input_dim=20, hidden_dim=60)
        x_recon, hidden = sae(sample_latent)
        
        assert x_recon.shape == sample_latent.shape
        assert hidden.shape == (100, 60)
    
    def test_sparse_ae_encode(self, sample_latent):
        """Test sparse autoencoder encoding."""
        sae = SparseAutoencoder(input_dim=20, hidden_dim=60)
        hidden = sae.encode(sample_latent)
        
        assert hidden.shape == (100, 60)
    
    def test_sparse_ae_loss(self, sample_latent):
        """Test sparse autoencoder loss."""
        sae = SparseAutoencoder(input_dim=20, hidden_dim=60, l1_lambda=0.01)
        loss = sae.loss(sample_latent)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_topk_sparse_ae(self, sample_latent):
        """Test TopK sparse autoencoder."""
        sae = TopKSparseAutoencoder(input_dim=20, hidden_dim=60, k=10)
        x_recon, hidden = sae(sample_latent)
        
        # Check that exactly k values are non-zero per sample
        non_zero_counts = (hidden != 0).sum(dim=1)
        assert (non_zero_counts == 10).all()
    
    def test_deep_sparse_ae(self, sample_latent):
        """Test deep sparse autoencoder."""
        sae = DeepSparseAutoencoder(
            input_dim=20,
            hidden_dims=[60, 40]
        )
        x_recon, hidden = sae(sample_latent)
        
        assert x_recon.shape == sample_latent.shape
        assert hidden.shape == (100, 40)


class TestMicrobiomeVAE:
    """Test microbiome-specific VAE."""
    
    @pytest.fixture
    def microbiome_data(self):
        """Generate sparse microbiome-like data."""
        np.random.seed(42)
        # Simulate RCLR-transformed data
        X = np.random.randn(50, 100).astype(np.float32)
        X[np.random.random(X.shape) < 0.3] = 0  # Add sparsity
        return torch.FloatTensor(X)
    
    def test_microbiome_vae_import(self):
        """Test microbiome VAE can be imported."""
        from q2_mechinterp.microbiome import MicrobiomeVAE
        assert MicrobiomeVAE is not None
    
    def test_microbiome_vae_forward(self, microbiome_data):
        """Test microbiome VAE forward pass."""
        from q2_mechinterp.microbiome import MicrobiomeVAE
        
        vae = MicrobiomeVAE(
            input_dim=100,
            latent_dim=20,
            hidden_dims=[64, 32],
            beta=0.01
        )
        
        x_recon, mu, logvar = vae(microbiome_data)
        assert x_recon.shape == microbiome_data.shape


class TestTranscriptomicsVAE:
    """Test transcriptomics-specific VAE."""
    
    @pytest.fixture
    def expression_data(self):
        """Generate gene expression-like data."""
        np.random.seed(42)
        X = np.abs(np.random.randn(50, 200)).astype(np.float32)
        return torch.FloatTensor(X)
    
    def test_transcriptomics_vae_import(self):
        """Test transcriptomics VAE can be imported."""
        from q2_mechinterp.transcriptomics import TranscriptomicsVAE
        assert TranscriptomicsVAE is not None
    
    def test_transcriptomics_vae_forward(self, expression_data):
        """Test transcriptomics VAE forward pass."""
        from q2_mechinterp.transcriptomics import TranscriptomicsVAE
        
        vae = TranscriptomicsVAE(
            input_dim=200,
            latent_dim=30,
            hidden_dims=[128, 64]
        )
        
        x_recon, mu, logvar = vae(expression_data)
        assert x_recon.shape == expression_data.shape


class TestTrainer:
    """Test VAE trainer."""
    
    @pytest.fixture
    def training_setup(self):
        """Create VAE and data loaders for training tests."""
        from q2_mechinterp.core.vae import ConcreteVAE
        
        # Small VAE
        vae = ConcreteVAE(input_dim=20, latent_dim=5, hidden_dims=[10])
        
        # Small dataset
        X = torch.randn(100, 20)
        dataset = TensorDataset(X)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16)
        
        return vae, train_loader, val_loader
    
    def test_trainer_import(self):
        """Test trainer can be imported."""
        from q2_mechinterp.training import VAETrainer
        assert VAETrainer is not None
    
    def test_trainer_train(self, training_setup):
        """Test basic training loop."""
        from q2_mechinterp.training import VAETrainer
        
        vae, train_loader, val_loader = training_setup
        trainer = VAETrainer(vae, device='cpu', learning_rate=1e-3)
        
        train_losses, val_losses = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            verbose=False
        )
        
        assert len(train_losses) == 2
        assert len(val_losses) == 2
        assert train_losses[-1] < train_losses[0]  # Loss should decrease


class TestFeatureExtractor:
    """Test feature extraction."""
    
    @pytest.fixture
    def extractor_setup(self):
        """Create VAE and extractor for tests."""
        from q2_mechinterp.core.vae import ConcreteVAE
        from q2_mechinterp.extraction import FeatureExtractor
        
        vae = ConcreteVAE(input_dim=20, latent_dim=5, hidden_dims=[10])
        extractor = FeatureExtractor(vae, device='cpu')
        
        X = np.random.randn(50, 20).astype(np.float32)
        
        return extractor, X
    
    def test_extractor_import(self):
        """Test extractor can be imported."""
        from q2_mechinterp.extraction import FeatureExtractor
        assert FeatureExtractor is not None
    
    def test_get_latent_features(self, extractor_setup):
        """Test latent feature extraction."""
        extractor, X = extractor_setup
        
        latent = extractor.get_latent_features(X)
        assert latent.shape == (50, 5)
    
    def test_train_sparse_autoencoder(self, extractor_setup):
        """Test sparse autoencoder training."""
        extractor, X = extractor_setup
        
        latent = extractor.get_latent_features(X)
        extractor.train_sparse_autoencoder(
            latent,
            expansion_factor=2,
            epochs=5,
            verbose=False
        )
        
        assert extractor.sparse_autoencoder is not None
    
    def test_get_sparse_features(self, extractor_setup):
        """Test sparse feature extraction."""
        extractor, X = extractor_setup
        
        latent = extractor.get_latent_features(X)
        extractor.train_sparse_autoencoder(latent, expansion_factor=2, epochs=5, verbose=False)
        
        sparse = extractor.get_sparse_features(latent)
        assert sparse.shape == (50, 10)  # 5 * 2 expansion


class TestVisualization:
    """Test visualization functions."""
    
    def test_visualization_imports(self):
        """Test visualization functions can be imported."""
        from q2_mechinterp.visualization import (
            plot_training_curves,
            plot_latent_space,
            plot_feature_importance
        )
        assert plot_training_curves is not None
        assert plot_latent_space is not None
        assert plot_feature_importance is not None


class TestPackageAPI:
    """Test main package API."""
    
    def test_main_imports(self):
        """Test main package imports work."""
        from q2_mechinterp import (
            MicrobiomeVAE,
            TranscriptomicsVAE,
            VAETrainer,
            FeatureExtractor
        )
        
        assert MicrobiomeVAE is not None
        assert TranscriptomicsVAE is not None
        assert VAETrainer is not None
        assert FeatureExtractor is not None
    
    def test_version(self):
        """Test package has version."""
        import q2_mechinterp
        assert hasattr(q2_mechinterp, '__version__')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
