"""
Hyperparameter tuning utilities using Optuna.

This module provides:
- VAETuner for automated hyperparameter optimization
- Configurable search spaces
- Trial pruning for efficiency
- Multiple sampler support

Design Principles:
- Non-invasive: Works with any VAE model
- Efficient: Early stopping and pruning
- Reproducible: Fixed seeds for all randomization
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import (
    Dict, List, Optional, Any, Callable, Type,
    Union, Tuple
)
from dataclasses import dataclass, field
import warnings
import time

# Check for Optuna availability
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")


@dataclass
class TuningResult:
    """
    Results from hyperparameter tuning.
    
    Attributes:
        best_params: Best hyperparameters found
        best_value: Best objective value achieved
        n_trials: Number of trials completed
        study_name: Name of the Optuna study
        all_trials: List of all trial results
        total_time: Total tuning time in seconds
        best_model_path: Path to best model checkpoint (if saved)
    """
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    total_time: float = 0.0
    best_model_path: Optional[str] = None
    
    def __repr__(self) -> str:
        return (
            f"TuningResult(best_value={self.best_value:.4f}, "
            f"n_trials={self.n_trials})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'study_name': self.study_name,
            'total_time': self.total_time,
            'best_model_path': self.best_model_path,
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            f"Tuning Results: {self.study_name}",
            "=" * 50,
            f"Best value: {self.best_value:.4f}",
            f"Trials completed: {self.n_trials}",
            f"Total time: {self.total_time:.1f}s",
            "",
            "Best parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append("=" * 50)
        return '\n'.join(lines)


# Default parameter search space
DEFAULT_PARAM_SPACE = {
    'latent_dim': {
        'type': 'int',
        'low': 10,
        'high': 100
    },
    'hidden_dims_choice': {
        'type': 'categorical',
        'choices': [
            [128, 64],
            [256, 128],
            [256, 128, 64],
            [512, 256, 128],
            [512, 256, 128, 64]
        ]
    },
    'dropout_rate': {
        'type': 'float',
        'low': 0.1,
        'high': 0.5
    },
    'beta': {
        'type': 'float_log',
        'low': 0.001,
        'high': 1.0
    },
    'learning_rate': {
        'type': 'float_log',
        'low': 1e-5,
        'high': 1e-2
    }
}


class VAETuner:
    """
    Optuna-based hyperparameter tuning for VAE models.
    
    Supports:
    - Flexible parameter search spaces
    - Multiple sampling strategies (TPE, Random, Grid)
    - Trial pruning for early stopping
    - Parallel optimization
    
    Example:
        >>> tuner = VAETuner(
        ...     model_class=MicrobiomeVAE,
        ...     device=device,
        ...     param_space={
        ...         'latent_dim': {'type': 'int', 'low': 10, 'high': 100},
        ...         'beta': {'type': 'float_log', 'low': 0.001, 'high': 1.0}
        ...     }
        ... )
        >>> 
        >>> result = tuner.tune(
        ...     train_data=X_train,
        ...     val_data=X_val,
        ...     n_trials=50,
        ...     epochs_per_trial=30
        ... )
        >>> 
        >>> print(result.summary())
        >>> best_model = tuner.create_best_model(input_dim=X_train.shape[1])
    
    Attributes:
        model_class: VAE model class to tune
        param_space: Dictionary defining search space
        device: PyTorch device
    """
    
    def __init__(
        self,
        model_class: Type,
        device: torch.device,
        param_space: Optional[Dict[str, Dict[str, Any]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        direction: str = 'minimize'
    ):
        """
        Initialize tuner.
        
        Args:
            model_class: VAE model class
            device: PyTorch device
            param_space: Search space definition (None for defaults)
            fixed_params: Parameters to fix (not tune)
            random_state: Random seed
            direction: 'minimize' or 'maximize'
        """
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna required for tuning. Install with: pip install optuna"
            )
        
        self.model_class = model_class
        self.device = device
        self.param_space = param_space or DEFAULT_PARAM_SPACE.copy()
        self.fixed_params = fixed_params or {}
        self.random_state = random_state
        self.direction = direction
        
        self._study: Optional['optuna.Study'] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._input_dim: Optional[int] = None
    
    def tune(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        n_trials: int = 50,
        epochs_per_trial: int = 30,
        batch_size: int = 32,
        timeout: Optional[float] = None,
        study_name: Optional[str] = None,
        show_progress_bar: bool = True,
        n_jobs: int = 1,
        sampler: Optional[str] = 'tpe',
        pruner: Optional[str] = 'median',
        training_kwargs: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> TuningResult:
        """
        Run hyperparameter tuning.
        
        Args:
            train_data: Training data (n_samples, n_features)
            val_data: Validation data
            n_trials: Number of trials to run
            epochs_per_trial: Training epochs per trial
            batch_size: Batch size
            timeout: Optional timeout in seconds
            study_name: Name for the study
            show_progress_bar: Show Optuna progress bar
            n_jobs: Number of parallel jobs
            sampler: Sampler type ('tpe', 'random', 'cmaes')
            pruner: Pruner type ('median', 'hyperband', 'none')
            training_kwargs: Additional training arguments
            callbacks: Optional callbacks for each trial
        
        Returns:
            TuningResult with best parameters and history
        """
        self._input_dim = train_data.shape[1]
        training_kwargs = training_kwargs or {}
        
        # Create data loaders
        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Create sampler
        if sampler == 'tpe':
            optuna_sampler = TPESampler(seed=self.random_state)
        elif sampler == 'random':
            optuna_sampler = optuna.samplers.RandomSampler(seed=self.random_state)
        elif sampler == 'cmaes':
            optuna_sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        else:
            optuna_sampler = TPESampler(seed=self.random_state)
        
        # Create pruner
        if pruner == 'median':
            optuna_pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == 'hyperband':
            optuna_pruner = optuna.pruners.HyperbandPruner()
        elif pruner == 'none' or pruner is None:
            optuna_pruner = optuna.pruners.NopPruner()
        else:
            optuna_pruner = MedianPruner()
        
        # Create study
        study_name = study_name or f"vae_tuning_{int(time.time())}"
        self._study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            sampler=optuna_sampler,
            pruner=optuna_pruner
        )
        
        # Define objective
        def objective(trial: 'optuna.Trial') -> float:
            return self._objective(
                trial,
                train_loader,
                val_loader,
                epochs_per_trial,
                training_kwargs,
                callbacks
            )
        
        # Run optimization
        start_time = time.time()
        
        optuna.logging.set_verbosity(
            optuna.logging.INFO if show_progress_bar else optuna.logging.WARNING
        )
        
        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar
        )
        
        total_time = time.time() - start_time
        
        # Extract results
        self._best_params = self._study.best_params
        
        # Collect all trial info
        all_trials = []
        for trial in self._study.trials:
            trial_info = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
                'duration': (
                    trial.duration.total_seconds() 
                    if trial.duration else None
                )
            }
            all_trials.append(trial_info)
        
        return TuningResult(
            best_params=self._best_params,
            best_value=self._study.best_value,
            n_trials=len(self._study.trials),
            study_name=study_name,
            all_trials=all_trials,
            total_time=total_time
        )
    
    def _objective(
        self,
        trial: 'optuna.Trial',
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        training_kwargs: Dict[str, Any],
        callbacks: Optional[List[Callable]]
    ) -> float:
        """Objective function for a single trial."""
        from q2_mechinterp.training import VAETrainer
        
        # Suggest parameters
        params = {}
        for param_name, param_config in self.param_space.items():
            params[param_name] = self._suggest_param(trial, param_name, param_config)
        
        # Handle hidden_dims_choice -> hidden_dims conversion
        if 'hidden_dims_choice' in params:
            params['hidden_dims'] = params.pop('hidden_dims_choice')
        
        # Add fixed parameters
        params.update(self.fixed_params)
        
        # Extract training-specific params
        lr = params.pop('learning_rate', 1e-3)
        beta = params.pop('beta', 0.01)
        alpha = params.pop('alpha', 0.001)
        
        # Create model
        model_params = {'input_dim': self._input_dim}
        model_params.update({
            k: v for k, v in params.items()
            if k not in ['learning_rate', 'beta', 'alpha']
        })
        
        try:
            model = self.model_class(**model_params).to(self.device)
        except Exception as e:
            warnings.warn(f"Model creation failed: {e}")
            return float('inf')
        
        # Create trainer
        trainer = VAETrainer(model, self.device, lr=lr)
        
        # Train with pruning
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, _ = trainer.train_epoch(
                train_loader,
                beta=beta,
                alpha=alpha,
                **{k: v for k, v in training_kwargs.items() if k not in ['beta', 'alpha']}
            )
            
            # Validate
            val_loss = trainer.validate(val_loader, beta=beta, alpha=alpha)
            
            best_val_loss = min(best_val_loss, val_loss)
            
            # Report for pruning
            trial.report(val_loss, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(trial, epoch, val_loss)
        
        return best_val_loss
    
    def _suggest_param(
        self,
        trial: 'optuna.Trial',
        name: str,
        config: Dict[str, Any]
    ) -> Any:
        """Suggest a parameter value based on config."""
        param_type = config['type']
        
        if param_type == 'int':
            return trial.suggest_int(
                name,
                config['low'],
                config['high'],
                step=config.get('step', 1)
            )
        
        elif param_type == 'float':
            return trial.suggest_float(
                name,
                config['low'],
                config['high'],
                step=config.get('step'),
                log=False
            )
        
        elif param_type == 'float_log':
            return trial.suggest_float(
                name,
                config['low'],
                config['high'],
                log=True
            )
        
        elif param_type == 'categorical':
            return trial.suggest_categorical(name, config['choices'])
        
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def create_best_model(
        self,
        input_dim: Optional[int] = None
    ) -> torch.nn.Module:
        """
        Create model with best parameters.
        
        Args:
            input_dim: Input dimension (uses cached if None)
        
        Returns:
            Instantiated model with best parameters
        """
        if self._best_params is None:
            raise RuntimeError("No tuning results. Call tune() first.")
        
        input_dim = input_dim or self._input_dim
        if input_dim is None:
            raise ValueError("input_dim required")
        
        params = self._best_params.copy()
        
        # Handle hidden_dims_choice
        if 'hidden_dims_choice' in params:
            params['hidden_dims'] = params.pop('hidden_dims_choice')
        
        # Remove training params
        for key in ['learning_rate', 'beta', 'alpha']:
            params.pop(key, None)
        
        # Add fixed params
        params.update(self.fixed_params)
        
        return self.model_class(input_dim=input_dim, **params).to(self.device)
    
    def get_param_importances(self) -> Dict[str, float]:
        """
        Get hyperparameter importances.
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self._study is None:
            raise RuntimeError("No study available. Call tune() first.")
        
        try:
            importances = optuna.importance.get_param_importances(self._study)
            return dict(importances)
        except Exception as e:
            warnings.warn(f"Could not compute importances: {e}")
            return {}
    
    def plot_optimization_history(self):
        """Plot optimization history (requires matplotlib)."""
        if self._study is None:
            raise RuntimeError("No study available")
        
        try:
            import optuna.visualization as vis
            return vis.plot_optimization_history(self._study)
        except ImportError:
            warnings.warn("Visualization requires plotly")
            return None
    
    def plot_param_importances(self):
        """Plot parameter importances."""
        if self._study is None:
            raise RuntimeError("No study available")
        
        try:
            import optuna.visualization as vis
            return vis.plot_param_importances(self._study)
        except ImportError:
            warnings.warn("Visualization requires plotly")
            return None


class GridSearch:
    """
    Simple grid search for hyperparameter tuning.
    
    For small search spaces where exhaustive search is feasible.
    """
    
    def __init__(
        self,
        model_class: Type,
        device: torch.device,
        param_grid: Dict[str, List[Any]],
        random_state: int = 42
    ):
        self.model_class = model_class
        self.device = device
        self.param_grid = param_grid
        self.random_state = random_state
        self.results_: List[Dict[str, Any]] = []
    
    def search(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Run grid search.
        
        Returns:
            Dictionary with best parameters and all results
        """
        from q2_mechinterp.training import VAETrainer
        import itertools
        
        input_dim = train_data.shape[1]
        
        # Create data loaders
        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        best_params = None
        best_score = float('inf')
        self.results_ = []
        
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            
            # Extract training params
            lr = params.pop('learning_rate', 1e-3)
            beta = params.pop('beta', 0.01)
            alpha = params.pop('alpha', 0.001)
            
            # Create model
            model_params = {'input_dim': input_dim, **params}
            
            try:
                model = self.model_class(**model_params).to(self.device)
            except Exception as e:
                continue
            
            # Train
            torch.manual_seed(self.random_state)
            trainer = VAETrainer(model, self.device, lr=lr)
            
            _, val_losses = trainer.train(
                train_loader, val_loader,
                epochs=epochs,
                verbose=0,
                beta=beta,
                alpha=alpha,
                **training_kwargs
            )
            
            score = min(val_losses) if val_losses else float('inf')
            
            result = {
                'params': {**params, 'learning_rate': lr, 'beta': beta, 'alpha': alpha},
                'score': score
            }
            self.results_.append(result)
            
            if score < best_score:
                best_score = score
                best_params = result['params']
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results_
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'TuningResult',
    'VAETuner',
    'GridSearch',
    'DEFAULT_PARAM_SPACE',
]
