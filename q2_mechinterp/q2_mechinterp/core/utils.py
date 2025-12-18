"""
Core utilities for VAE feature engineering.
"""

import numpy as np
import torch
from typing import Optional


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        prefer_mps: If True, prefer MPS (Apple Silicon) over CPU when CUDA unavailable.
    
    Returns:
        torch.device: The selected computation device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_noise(data: torch.Tensor, noise_factor: float = 0.05) -> torch.Tensor:
    """
    Add Gaussian noise to input data for regularization.
    
    Args:
        data: Input tensor.
        noise_factor: Scale of noise relative to data standard deviation.
    
    Returns:
        Noisy data tensor.
    """
    noise = torch.randn_like(data) * noise_factor * data.std()
    return data + noise


def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if a tensor contains NaN or Inf values.
    
    Args:
        tensor: Tensor to check.
        name: Name for error reporting.
    
    Returns:
        True if tensor is valid (no NaN/Inf), False otherwise.
    """
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values")
        return False
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf values")
        return False
    return True


def clean_array(
    array: np.ndarray,
    nan_value: float = 0.0,
    posinf_value: float = 1.0,
    neginf_value: float = -1.0
) -> np.ndarray:
    """
    Clean numpy array by replacing NaN and Inf values.
    
    Args:
        array: Input numpy array.
        nan_value: Value to replace NaNs with.
        posinf_value: Value to replace positive infinities with.
        neginf_value: Value to replace negative infinities with.
    
    Returns:
        Cleaned numpy array.
    """
    return np.nan_to_num(array, nan=nan_value, posinf=posinf_value, neginf=neginf_value)


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        restore_best: Whether to restore best model weights on stop.
    """
    
    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0001,
        restore_best: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model being trained.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
        return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.should_stop = False
