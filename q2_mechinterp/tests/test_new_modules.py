"""
Comprehensive tests for VAE features package.

Tests cover:
- Core validation utilities
- Data processing
- Evaluation
- Preprocessing (batch correction)
- Cross-validation
- Augmentation
- Feature selection
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Generate data with some structure
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some zeros for sparsity (like microbiome data)
    X[X < 0] = 0
    
    return X


@pytest.fixture
def labeled_data(sample_data):
    """Generate sample data with labels."""
    n_samples = sample_data.shape[0]
    n_classes = 3
    
    labels = np.random.randint(0, n_classes, n_samples)
    return sample_data, labels


@pytest.fixture
def batch_data():
    """Generate multi-batch data for batch correction tests."""
    np.random.seed(42)
    n_per_batch = 30
    n_features = 50
    
    # Batch 1: mean-shifted
    batch1 = np.random.randn(n_per_batch, n_features).astype(np.float32) + 1.0
    
    # Batch 2: variance-shifted  
    batch2 = np.random.randn(n_per_batch, n_features).astype(np.float32) * 2.0
    
    # Batch 3: standard
    batch3 = np.random.randn(n_per_batch, n_features).astype(np.float32)
    
    X = np.vstack([batch1, batch2, batch3])
    batch_labels = np.array(['batch1'] * n_per_batch + 
                            ['batch2'] * n_per_batch + 
                            ['batch3'] * n_per_batch)
    
    return X, batch_labels


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cpu')


# =============================================================================
# Test Validation Module
# =============================================================================

class TestValidation:
    """Tests for validation utilities."""
    
    def test_validate_array_basic(self):
        """Test basic array validation."""
        from q2_mechinterp.core.validation import validate_array
        
        arr = np.array([[1, 2], [3, 4]])
        result = validate_array(arr, "test", ndim=2)
        
        assert result.shape == (2, 2)
    
    def test_validate_array_nan_error(self):
        """Test that NaN raises error."""
        from q2_mechinterp.core.validation import validate_array, ValidationError
        
        arr = np.array([1.0, np.nan, 3.0])
        
        with pytest.raises(ValidationError):
            validate_array(arr, "test", allow_nan=False)
    
    def test_validate_array_wrong_dim(self):
        """Test dimension validation."""
        from q2_mechinterp.core.validation import validate_array, ValidationError
        
        arr = np.array([1, 2, 3])
        
        with pytest.raises(ValidationError):
            validate_array(arr, "test", ndim=2)
    
    def test_validate_positive(self):
        """Test positive value validation."""
        from q2_mechinterp.core.validation import validate_positive, ValidationError
        
        assert validate_positive(5, "test") == 5
        
        with pytest.raises(ValidationError):
            validate_positive(-1, "test")
    
    def test_validate_range(self):
        """Test range validation."""
        from q2_mechinterp.core.validation import validate_range, ValidationError
        
        assert validate_range(0.5, "test", min_val=0, max_val=1) == 0.5
        
        with pytest.raises(ValidationError):
            validate_range(1.5, "test", min_val=0, max_val=1)
    
    def test_validate_choice(self):
        """Test choice validation."""
        from q2_mechinterp.core.validation import validate_choice, ValidationError
        
        assert validate_choice('a', ['a', 'b', 'c'], "test") == 'a'
        
        with pytest.raises(ValidationError):
            validate_choice('d', ['a', 'b', 'c'], "test")


# =============================================================================
# Test Data Processor
# =============================================================================

class TestBaseDataProcessor:
    """Tests for BaseDataProcessor."""
    
    def test_load_from_numpy(self, sample_data):
        """Test loading data from numpy array."""
        from q2_mechinterp.core.data import BaseDataProcessor
        
        # Create concrete subclass
        class TestProcessor(BaseDataProcessor):
            def transform(self, **kwargs):
                return self
        
        processor = TestProcessor()
        processor.load_from_numpy(sample_data)
        
        assert processor.n_samples == sample_data.shape[0]
        assert processor.n_features == sample_data.shape[1]
        assert processor.is_loaded
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        from q2_mechinterp.core.data import BaseDataProcessor
        
        class TestProcessor(BaseDataProcessor):
            def transform(self, **kwargs):
                return self
        
        processor = TestProcessor()
        processor.load_from_numpy(sample_data)
        processor.split_data(test_size=0.2, val_size=0.1)
        
        assert processor.has_split
        
        # Check split sizes are reasonable
        train_x, _ = processor.get_split_data('train')
        val_x, _ = processor.get_split_data('val')
        
        assert len(train_x) > len(val_x)
    
    def test_filter_features(self, sample_data):
        """Test feature filtering."""
        from q2_mechinterp.core.data import BaseDataProcessor
        
        class TestProcessor(BaseDataProcessor):
            def transform(self, **kwargs):
                return self
        
        processor = TestProcessor()
        processor.load_from_numpy(sample_data)
        
        original_features = processor.n_features
        processor.filter_features(prevalence_threshold=0.1, max_features=20)
        
        assert processor.n_features <= 20
        assert processor.n_features <= original_features


# =============================================================================
# Test Evaluation Module
# =============================================================================

class TestEvaluation:
    """Tests for evaluation module."""
    
    def test_feature_evaluator_classification(self, labeled_data):
        """Test FeatureEvaluator for classification."""
        from q2_mechinterp.evaluation import FeatureEvaluator
        
        X, y = labeled_data
        
        evaluator = FeatureEvaluator(task='classification', cv=3, verbose=False)
        result = evaluator.evaluate(X, y, model_name='logistic')
        
        assert result.mean_score > 0  # Should have some accuracy
        assert result.metric_name == 'accuracy'
        assert len(result.cv_scores) == 3
    
    def test_feature_quality_metrics(self, sample_data):
        """Test feature quality metrics calculation."""
        from q2_mechinterp.evaluation import calculate_feature_quality_metrics
        
        # Simulate VAE outputs
        original = sample_data
        latent = np.random.randn(sample_data.shape[0], 10).astype(np.float32)
        reconstructed = original + np.random.randn(*original.shape).astype(np.float32) * 0.1
        
        metrics = calculate_feature_quality_metrics(original, latent, reconstructed)
        
        assert 'reconstruction_mse' in metrics
        assert 'latent_dim' in metrics
        assert metrics['latent_dim'] == 10
    
    def test_compare_features(self, labeled_data):
        """Test comparing multiple feature sets."""
        from q2_mechinterp.evaluation import FeatureEvaluator
        
        X, y = labeled_data
        
        # Create different feature transformations
        feature_sets = {
            'original': X,
            'standardized': (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8),
            'subset': X[:, :25]
        }
        
        evaluator = FeatureEvaluator(task='classification', cv=3, verbose=False)
        result = evaluator.compare_features(feature_sets, y, models=['logistic'])
        
        assert len(result.results_df) == 3  # 3 feature sets
        assert result.best_overall[0] in feature_sets


# =============================================================================
# Test Preprocessing (Batch Correction)
# =============================================================================

class TestBatchCorrection:
    """Tests for batch correction methods."""
    
    def test_simple_batch_corrector_center(self, batch_data):
        """Test simple centering correction."""
        from q2_mechinterp.preprocessing import SimpleBatchCorrector
        
        X, batch_labels = batch_data
        
        corrector = SimpleBatchCorrector(method='center')
        X_corrected = corrector.fit_transform(X, batch_labels)
        
        assert X_corrected.shape == X.shape
        
        # Check that batch means are more similar after correction
        batches = np.unique(batch_labels)
        means_before = [X[batch_labels == b].mean() for b in batches]
        means_after = [X_corrected[batch_labels == b].mean() for b in batches]
        
        # Standard deviation of batch means should be lower
        assert np.std(means_after) < np.std(means_before)
    
    def test_simple_batch_corrector_zscore(self, batch_data):
        """Test z-score correction."""
        from q2_mechinterp.preprocessing import SimpleBatchCorrector
        
        X, batch_labels = batch_data
        
        corrector = SimpleBatchCorrector(method='zscore')
        X_corrected = corrector.fit_transform(X, batch_labels)
        
        assert X_corrected.shape == X.shape
        assert not np.isnan(X_corrected).any()
    
    def test_combat_corrector(self, batch_data):
        """Test ComBat correction."""
        from q2_mechinterp.preprocessing import ComBatCorrector
        
        X, batch_labels = batch_data
        
        combat = ComBatCorrector(parametric=True)
        X_corrected = combat.fit_transform(X, batch_labels)
        
        assert X_corrected.shape == X.shape
        assert not np.isnan(X_corrected).any()
    
    def test_conqur_corrector(self, batch_data):
        """Test ConQuR correction."""
        from q2_mechinterp.preprocessing import ConQuRCorrector
        
        X, batch_labels = batch_data
        
        conqur = ConQuRCorrector(reference_batch='batch3')
        X_corrected = conqur.fit_transform(X, batch_labels)
        
        assert X_corrected.shape == X.shape


# =============================================================================
# Test Augmentation
# =============================================================================

class TestAugmentation:
    """Tests for data augmentation."""
    
    def test_microbiome_augmentor_noise(self, sample_data):
        """Test microbiome multiplicative noise."""
        from q2_mechinterp.augmentation import MicrobiomeAugmentor
        
        augmentor = MicrobiomeAugmentor(random_state=42)
        X_aug = augmentor.multiplicative_noise(sample_data, noise_scale=0.1)
        
        assert X_aug.shape == sample_data.shape
        # Original zeros should still be zero
        assert (X_aug[sample_data == 0] == 0).all()
    
    def test_microbiome_augmentor_dropout(self, sample_data):
        """Test microbiome dropout augmentation."""
        from q2_mechinterp.augmentation import MicrobiomeAugmentor
        
        augmentor = MicrobiomeAugmentor(random_state=42)
        X_aug = augmentor.dropout(sample_data, dropout_rate=0.1)
        
        assert X_aug.shape == sample_data.shape
        # Should have more zeros than original
        assert (X_aug == 0).sum() >= (sample_data == 0).sum()
    
    def test_transcriptomics_augmentor_noise(self, sample_data):
        """Test transcriptomics Gaussian noise."""
        from q2_mechinterp.augmentation import TranscriptomicsAugmentor
        
        augmentor = TranscriptomicsAugmentor(random_state=42)
        X_aug = augmentor.gaussian_noise(sample_data, noise_level=0.1)
        
        assert X_aug.shape == sample_data.shape
        # Values should be different (noise added)
        assert not np.allclose(X_aug, sample_data)
    
    def test_augmented_dataloader(self, sample_data):
        """Test augmented data loader."""
        from q2_mechinterp.augmentation import (
            AugmentedDataLoader, TranscriptomicsAugmentor
        )
        
        augmentor = TranscriptomicsAugmentor()
        loader = AugmentedDataLoader(
            sample_data,
            augmentor=augmentor,
            augment_prob=1.0,  # Always augment
            batch_size=16
        )
        
        # Iterate and check batches
        for batch in loader:
            assert batch.shape[1] == sample_data.shape[1]
            break


# =============================================================================
# Test Downstream Integration
# =============================================================================

class TestDownstream:
    """Tests for downstream integration."""
    
    def test_feature_selector(self, sample_data):
        """Test feature selection."""
        from q2_mechinterp.downstream import FeatureSelector
        
        selector = FeatureSelector(method='variance', n_features=10)
        selector.fit(sample_data)
        X_selected = selector.transform(sample_data)
        
        assert X_selected.shape == (sample_data.shape[0], 10)
        
        indices = selector.get_selected_indices()
        assert len(indices) == 10
    
    def test_supervised_feature_selector(self, labeled_data):
        """Test supervised feature selection."""
        from q2_mechinterp.downstream import SupervisedFeatureSelector
        
        X, y = labeled_data
        
        selector = SupervisedFeatureSelector(method='correlation', n_features=10)
        selector.fit(X, y)
        X_selected = selector.transform(X)
        
        assert X_selected.shape == (X.shape[0], 10)


# =============================================================================
# Test Cross-Validation
# =============================================================================

class TestCrossValidation:
    """Tests for cross-validation."""
    
    def test_cross_validator_basic(self, sample_data, device):
        """Test basic cross-validation."""
        from q2_mechinterp.training import CrossValidator
        from q2_mechinterp.core import ConcreteVAE
        
        cv = CrossValidator(n_splits=3, stratified=False, random_state=42)
        
        result = cv.cross_validate(
            model_class=ConcreteVAE,
            model_params={'latent_dim': 10, 'hidden_dims': [32, 16]},
            data=sample_data,
            epochs=2,  # Very few for testing
            batch_size=16,
            device=device,
            verbose=False
        )
        
        assert len(result.fold_metrics) == 3
        assert 'val_loss' in result.mean_metrics
        assert result.best_fold in [0, 1, 2]


# =============================================================================
# Test Tuning Module
# =============================================================================

class TestTuning:
    """Tests for hyperparameter tuning."""
    
    @pytest.mark.skipif(True, reason="Optuna tests are slow")
    def test_vae_tuner(self, sample_data, device):
        """Test VAE tuning with Optuna."""
        from q2_mechinterp.tuning import VAETuner
        from q2_mechinterp.core import ConcreteVAE
        
        # Simple param space
        param_space = {
            'latent_dim': {'type': 'categorical', 'choices': [10, 20]},
            'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.3}
        }
        
        tuner = VAETuner(
            model_class=ConcreteVAE,
            device=device,
            param_space=param_space
        )
        
        # Split data
        n = len(sample_data)
        train_data = sample_data[:int(0.8*n)]
        val_data = sample_data[int(0.8*n):]
        
        result = tuner.tune(
            train_data=train_data,
            val_data=val_data,
            n_trials=2,
            epochs_per_trial=2,
            show_progress_bar=False
        )
        
        assert result.best_params is not None
        assert result.n_trials == 2


# =============================================================================
# Test Protocol Compliance
# =============================================================================

class TestProtocols:
    """Tests for protocol compliance."""
    
    def test_vae_protocol(self):
        """Test VAE protocol compliance."""
        from q2_mechinterp.core import ConcreteVAE, VAEProtocol
        
        vae = ConcreteVAE(input_dim=50, latent_dim=10)
        
        # Check protocol compliance
        assert isinstance(vae, VAEProtocol)
        assert hasattr(vae, 'encode')
        assert hasattr(vae, 'decode')
        assert hasattr(vae, 'forward')
    
    def test_config_serialization(self, tmp_path):
        """Test config save/load."""
        from q2_mechinterp.core import VAEConfig
        
        config = VAEConfig(
            input_dim=100,
            latent_dim=20,
            hidden_dims=[64, 32]
        )
        
        # Save as YAML
        yaml_path = tmp_path / "config.yaml"
        config.save(yaml_path)
        
        # Load back
        loaded = VAEConfig.load(yaml_path)
        
        assert loaded.input_dim == config.input_dim
        assert loaded.latent_dim == config.latent_dim
        assert loaded.hidden_dims == config.hidden_dims


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflows."""
    
    def test_full_microbiome_pipeline(self, sample_data, device):
        """Test full microbiome workflow."""
        from q2_mechinterp.microbiome import MicrobiomeVAE
        from q2_mechinterp.training import VAETrainer
        from q2_mechinterp.extraction import FeatureExtractor
        
        # Create model
        input_dim = sample_data.shape[1]
        model = MicrobiomeVAE(
            input_dim=input_dim,
            latent_dim=10,
            hidden_dims=[32, 16]
        ).to(device)
        
        # Train (very briefly)
        trainer = VAETrainer(model, device)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(sample_data[:80])),
            batch_size=16,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(sample_data[80:])),
            batch_size=16,
            shuffle=False
        )
        
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            epochs=2,
            verbose=0
        )
        
        assert len(train_losses) == 2
        
        # Extract features
        extractor = FeatureExtractor(model, device)
        latent = extractor.get_latent_features(sample_data)
        
        assert latent.shape == (sample_data.shape[0], 10)
    
    def test_config_driven_workflow(self, sample_data, device, tmp_path):
        """Test config-driven workflow."""
        from q2_mechinterp.core import (
            VAEConfig, TrainingConfig, PipelineConfig,
            ConcreteVAE
        )
        from q2_mechinterp.training import VAETrainer
        
        # Create config
        vae_config = VAEConfig(
            input_dim=sample_data.shape[1],
            latent_dim=10,
            hidden_dims=[32, 16]
        )
        training_config = TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=1e-3
        )
        
        pipeline_config = PipelineConfig(
            vae=vae_config,
            training=training_config,
            output_dir=str(tmp_path)
        )
        
        # Save and reload
        config_path = tmp_path / "config.yaml"
        pipeline_config.save(config_path)
        loaded_config = PipelineConfig.load(config_path)
        
        assert loaded_config.vae.input_dim == vae_config.input_dim
        
        # Use config to create model
        model = ConcreteVAE(
            input_dim=loaded_config.vae.input_dim,
            latent_dim=loaded_config.vae.latent_dim,
            hidden_dims=loaded_config.vae.hidden_dims
        ).to(device)
        
        trainer = VAETrainer(
            model, device,
            lr=loaded_config.training.learning_rate
        )
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(sample_data)),
            batch_size=loaded_config.training.batch_size,
            shuffle=True
        )
        
        train_losses, _ = trainer.train(
            train_loader, None,
            epochs=loaded_config.training.epochs,
            verbose=0
        )
        
        assert len(train_losses) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
