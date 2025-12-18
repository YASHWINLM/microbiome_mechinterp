"""
Cross-validation utilities for VAE training.

This module provides:
- K-fold cross-validation for VAE models
- Stratified splitting support
- Aggregated metrics across folds
- Model selection via CV

Design Principles:
- Consistent with sklearn cross-validation API
- Support for stratified splits
- Comprehensive metric tracking
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import (
    Dict, List, Optional, Any, Union, Tuple, Type, Callable
)
from dataclasses import dataclass, field
import warnings
import time
from copy import deepcopy


@dataclass
class CVResult:
    """
    Results from cross-validation.
    
    Attributes:
        fold_metrics: Metrics for each fold
        mean_metrics: Mean of metrics across folds
        std_metrics: Standard deviation of metrics
        best_fold: Index of best performing fold
        best_model_state: State dict of best model (if saved)
        fold_histories: Training histories per fold
    """
    fold_metrics: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold: int
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    fold_histories: Optional[List[Dict[str, List[float]]]] = None
    total_time: float = 0.0
    
    def __repr__(self) -> str:
        main_metric = 'val_loss' if 'val_loss' in self.mean_metrics else list(self.mean_metrics.keys())[0]
        return (
            f"CVResult({main_metric}={self.mean_metrics[main_metric]:.4f} "
            f"± {self.std_metrics[main_metric]:.4f}, "
            f"best_fold={self.best_fold})"
        )
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "Cross-Validation Results",
            "=" * 50,
            f"Total time: {self.total_time:.1f}s",
            f"Best fold: {self.best_fold}",
            "",
            "Mean ± Std:",
        ]
        
        for metric in sorted(self.mean_metrics.keys()):
            mean = self.mean_metrics[metric]
            std = self.std_metrics[metric]
            lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
        
        lines.append("=" * 50)
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fold_metrics': self.fold_metrics,
            'mean_metrics': self.mean_metrics,
            'std_metrics': self.std_metrics,
            'best_fold': self.best_fold,
            'total_time': self.total_time,
        }


class CrossValidator:
    """
    K-fold cross-validation for VAE models.
    
    Provides robust validation of VAE performance by training
    on different data splits and aggregating results.
    
    Example:
        >>> cv = CrossValidator(n_splits=5, stratified=True)
        >>> 
        >>> result = cv.cross_validate(
        ...     model_class=MicrobiomeVAE,
        ...     model_params={'latent_dim': 50, 'hidden_dims': [256, 128]},
        ...     data=X,
        ...     labels=y,  # For stratification
        ...     epochs=50
        ... )
        >>> 
        >>> print(result.summary())
        >>> print(f"Mean loss: {result.mean_metrics['val_loss']:.4f}")
    
    Attributes:
        n_splits: Number of CV folds
        stratified: Whether to use stratified splits
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            stratified: Use stratified K-fold (requires labels)
            shuffle: Shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        
        self.n_splits = n_splits
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
    
    def cross_validate(
        self,
        model_class: Type,
        model_params: Dict[str, Any],
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        training_kwargs: Optional[Dict[str, Any]] = None,
        save_best_model: bool = True,
        verbose: bool = True
    ) -> CVResult:
        """
        Run cross-validation.
        
        Args:
            model_class: VAE model class
            model_params: Parameters for model instantiation
            data: Full dataset (n_samples, n_features)
            labels: Optional labels for stratification
            epochs: Training epochs per fold
            batch_size: Batch size
            device: PyTorch device
            training_kwargs: Additional training arguments
            save_best_model: Whether to save best fold's model
            verbose: Print progress
        
        Returns:
            CVResult with aggregated metrics
        """
        from q2_mechinterp.training import VAETrainer
        
        device = device or torch.device('cpu')
        training_kwargs = training_kwargs or {}
        
        data = np.asarray(data, dtype=np.float32)
        n_samples = len(data)
        input_dim = data.shape[1]
        
        # Create fold indices
        fold_indices = self._create_folds(n_samples, labels)
        
        fold_metrics = []
        fold_histories = []
        best_val_loss = float('inf')
        best_model_state = None
        best_fold = 0
        
        start_time = time.time()
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            if verbose:
                print(f"\n{'='*40}")
                print(f"Fold {fold_idx + 1}/{self.n_splits}")
                print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
                print(f"{'='*40}")
            
            # Create data loaders for this fold
            train_data = data[train_idx]
            val_data = data[val_idx]
            
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(train_data)),
                batch_size=batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(val_data)),
                batch_size=batch_size,
                shuffle=False
            )
            
            # Create model for this fold
            torch.manual_seed(self.random_state + fold_idx)
            model = model_class(input_dim=input_dim, **model_params).to(device)
            
            # Train
            trainer = VAETrainer(
                model, device,
                lr=training_kwargs.get('lr', 1e-3)
            )
            
            train_losses, val_losses = trainer.train(
                train_loader, val_loader,
                epochs=epochs,
                verbose=10 if verbose else 0,
                **{k: v for k, v in training_kwargs.items() if k != 'lr'}
            )
            
            # Record metrics
            best_epoch_loss = min(val_losses) if val_losses else float('inf')
            final_train_loss = train_losses[-1] if train_losses else float('inf')
            final_val_loss = val_losses[-1] if val_losses else float('inf')
            
            metrics = {
                'train_loss': final_train_loss,
                'val_loss': final_val_loss,
                'best_val_loss': best_epoch_loss,
                'n_epochs': len(train_losses),
            }
            
            # Additional metrics from trainer history
            if hasattr(trainer, 'history'):
                for key, values in trainer.history.items():
                    if values and key not in metrics:
                        metrics[f'final_{key}'] = values[-1]
            
            fold_metrics.append(metrics)
            fold_histories.append({
                'train_loss': train_losses,
                'val_loss': val_losses
            })
            
            # Track best model
            if best_epoch_loss < best_val_loss:
                best_val_loss = best_epoch_loss
                best_fold = fold_idx
                if save_best_model:
                    best_model_state = deepcopy(model.state_dict())
            
            if verbose:
                print(f"Fold {fold_idx + 1} - Best val loss: {best_epoch_loss:.4f}")
        
        total_time = time.time() - start_time
        
        # Aggregate metrics
        mean_metrics = {}
        std_metrics = {}
        
        all_metric_names = set()
        for fm in fold_metrics:
            all_metric_names.update(fm.keys())
        
        for metric_name in all_metric_names:
            values = [fm.get(metric_name, np.nan) for fm in fold_metrics]
            mean_metrics[metric_name] = float(np.nanmean(values))
            std_metrics[metric_name] = float(np.nanstd(values))
        
        return CVResult(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold=best_fold,
            best_model_state=best_model_state,
            fold_histories=fold_histories,
            total_time=total_time
        )
    
    def _create_folds(
        self,
        n_samples: int,
        labels: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create train/val indices for each fold."""
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if self.stratified and labels is not None:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            indices = list(splitter.split(np.zeros(n_samples), labels))
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            indices = list(splitter.split(np.zeros(n_samples)))
        
        return indices
    
    def get_best_model(
        self,
        cv_result: CVResult,
        model_class: Type,
        model_params: Dict[str, Any],
        input_dim: int,
        device: torch.device
    ) -> torch.nn.Module:
        """
        Get the best model from CV results.
        
        Args:
            cv_result: Cross-validation results
            model_class: Model class
            model_params: Model parameters
            input_dim: Input dimension
            device: PyTorch device
        
        Returns:
            Model loaded with best fold's weights
        """
        if cv_result.best_model_state is None:
            raise ValueError("No model state saved. Set save_best_model=True")
        
        model = model_class(input_dim=input_dim, **model_params).to(device)
        model.load_state_dict(cv_result.best_model_state)
        return model


class NestedCV:
    """
    Nested cross-validation for hyperparameter selection.
    
    Outer loop: Model evaluation
    Inner loop: Hyperparameter tuning
    """
    
    def __init__(
        self,
        outer_splits: int = 5,
        inner_splits: int = 3,
        stratified: bool = True,
        random_state: int = 42
    ):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.stratified = stratified
        self.random_state = random_state
    
    def run(
        self,
        model_class: Type,
        param_grid: Dict[str, List[Any]],
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run nested cross-validation.
        
        Args:
            model_class: VAE model class
            param_grid: Hyperparameter grid
            data: Full dataset
            labels: Optional labels for stratification
            epochs: Training epochs
            device: PyTorch device
            verbose: Print progress
        
        Returns:
            Dictionary with outer and inner CV results
        """
        from q2_mechinterp.tuning import GridSearch
        
        device = device or torch.device('cpu')
        data = np.asarray(data, dtype=np.float32)
        n_samples = len(data)
        
        # Create outer folds
        outer_cv = CrossValidator(
            n_splits=self.outer_splits,
            stratified=self.stratified,
            random_state=self.random_state
        )
        outer_folds = outer_cv._create_folds(n_samples, labels)
        
        outer_scores = []
        selected_params = []
        
        for fold_idx, (trainval_idx, test_idx) in enumerate(outer_folds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Outer Fold {fold_idx + 1}/{self.outer_splits}")
                print(f"{'='*50}")
            
            # Split trainval into train/val for inner CV
            trainval_data = data[trainval_idx]
            trainval_labels = labels[trainval_idx] if labels is not None else None
            test_data = data[test_idx]
            
            # Inner CV: Grid search
            grid_search = GridSearch(
                model_class=model_class,
                device=device,
                param_grid=param_grid,
                random_state=self.random_state + fold_idx
            )
            
            # Use first portion for training, rest for validation in inner CV
            split_idx = int(len(trainval_data) * 0.8)
            inner_train = trainval_data[:split_idx]
            inner_val = trainval_data[split_idx:]
            
            inner_result = grid_search.search(
                inner_train, inner_val,
                epochs=epochs // 2  # Fewer epochs for inner CV
            )
            
            best_params = inner_result['best_params']
            selected_params.append(best_params)
            
            if verbose:
                print(f"Best inner params: {best_params}")
            
            # Train final model on full trainval with best params
            from q2_mechinterp.training import VAETrainer
            
            input_dim = trainval_data.shape[1]
            model_params = {k: v for k, v in best_params.items()
                          if k not in ['learning_rate', 'beta', 'alpha']}
            
            model = model_class(input_dim=input_dim, **model_params).to(device)
            trainer = VAETrainer(
                model, device,
                lr=best_params.get('learning_rate', 1e-3)
            )
            
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(trainval_data)),
                batch_size=32, shuffle=True
            )
            test_loader = DataLoader(
                TensorDataset(torch.FloatTensor(test_data)),
                batch_size=32, shuffle=False
            )
            
            # Quick train on full trainval
            trainer.train(
                train_loader, None,
                epochs=epochs,
                verbose=0,
                beta=best_params.get('beta', 0.01),
                alpha=best_params.get('alpha', 0.001)
            )
            
            # Evaluate on test
            test_loss = trainer.validate(
                test_loader,
                beta=best_params.get('beta', 0.01),
                alpha=best_params.get('alpha', 0.001)
            )
            
            outer_scores.append(test_loss)
            
            if verbose:
                print(f"Outer fold test loss: {test_loss:.4f}")
        
        return {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'selected_params': selected_params,
        }


class RepeatedCV:
    """
    Repeated cross-validation for more stable estimates.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 3,
        stratified: bool = True,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratified = stratified
        self.random_state = random_state
    
    def cross_validate(
        self,
        model_class: Type,
        model_params: Dict[str, Any],
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run repeated cross-validation.
        
        Returns:
            Dictionary with aggregated results across all repeats
        """
        all_results = []
        
        for repeat in range(self.n_repeats):
            if verbose:
                print(f"\n{'#'*50}")
                print(f"Repeat {repeat + 1}/{self.n_repeats}")
                print(f"{'#'*50}")
            
            cv = CrossValidator(
                n_splits=self.n_splits,
                stratified=self.stratified,
                random_state=self.random_state + repeat * 1000
            )
            
            result = cv.cross_validate(
                model_class=model_class,
                model_params=model_params,
                data=data,
                labels=labels,
                epochs=epochs,
                device=device,
                verbose=verbose,
                **kwargs
            )
            
            all_results.append(result)
        
        # Aggregate across repeats
        all_fold_metrics = []
        for result in all_results:
            all_fold_metrics.extend(result.fold_metrics)
        
        # Compute overall statistics
        metric_names = set()
        for fm in all_fold_metrics:
            metric_names.update(fm.keys())
        
        overall_mean = {}
        overall_std = {}
        
        for metric in metric_names:
            values = [fm.get(metric, np.nan) for fm in all_fold_metrics]
            overall_mean[metric] = float(np.nanmean(values))
            overall_std[metric] = float(np.nanstd(values))
        
        return {
            'repeat_results': all_results,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'n_total_folds': len(all_fold_metrics),
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'CVResult',
    'CrossValidator',
    'NestedCV',
    'RepeatedCV',
]
