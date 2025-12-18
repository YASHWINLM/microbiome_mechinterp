"""
Input validation utilities for consistent error handling.

This module provides reusable validation functions and decorators that ensure
consistent error messages and input checking across the entire package.

Design Principles:
- Fail fast with informative errors
- Consistent error message format
- Type coercion when safe
- Decorator support for clean function signatures
"""

import numpy as np
import torch
from typing import (
    Optional, List, Any, Union, Callable, TypeVar, Tuple,
    Sequence, Type, get_type_hints
)
from functools import wraps
from pathlib import Path
import inspect


# Type variable for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ValidationError(Exception):
    """
    Custom exception for validation errors.
    
    Provides additional context about what was being validated
    and what went wrong.
    
    Attributes:
        param_name: Name of the parameter that failed validation
        expected: What was expected
        actual: What was received
        message: Full error message
    """
    
    def __init__(
        self,
        message: str,
        param_name: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[Any] = None
    ):
        self.param_name = param_name
        self.expected = expected
        self.actual = actual
        self.message = message
        
        # Build detailed message
        details = []
        if param_name:
            details.append(f"Parameter: {param_name}")
        if expected:
            details.append(f"Expected: {expected}")
        if actual is not None:
            actual_str = str(actual)[:100] + "..." if len(str(actual)) > 100 else str(actual)
            details.append(f"Got: {actual_str}")
        
        full_message = message
        if details:
            full_message += f"\n  " + "\n  ".join(details)
        
        super().__init__(full_message)


# =============================================================================
# Array Validation
# =============================================================================

def validate_array(
    arr: Any,
    name: str = "array",
    ndim: Optional[int] = None,
    shape: Optional[Tuple[Optional[int], ...]] = None,
    min_samples: Optional[int] = None,
    min_features: Optional[int] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    dtype: Optional[type] = None,
    ensure_2d: bool = False,
    copy: bool = False
) -> np.ndarray:
    """
    Validate and optionally convert numpy array.
    
    Args:
        arr: Array to validate
        name: Name for error messages
        ndim: Required number of dimensions
        shape: Expected shape (None elements are wildcards)
        min_samples: Minimum number of samples (axis 0)
        min_features: Minimum number of features (axis 1)
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        dtype: Required dtype (will convert if different)
        ensure_2d: Reshape 1D arrays to 2D
        copy: Return a copy instead of view
    
    Returns:
        Validated (and possibly converted) array
    
    Raises:
        ValidationError: If validation fails
    
    Example:
        >>> x = validate_array(data, "input", ndim=2, min_samples=10)
    """
    # None check
    if arr is None:
        raise ValidationError(
            f"{name} cannot be None",
            param_name=name,
            expected="array-like",
            actual=None
        )
    
    # Convert to numpy array
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as e:
            raise ValidationError(
                f"{name} could not be converted to numpy array: {e}",
                param_name=name,
                expected="array-like",
                actual=type(arr).__name__
            )
    
    # Ensure 2D
    if ensure_2d and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    
    # Check dimensions
    if ndim is not None and arr.ndim != ndim:
        raise ValidationError(
            f"{name} must be {ndim}D",
            param_name=name,
            expected=f"{ndim}D array",
            actual=f"{arr.ndim}D array with shape {arr.shape}"
        )
    
    # Check shape
    if shape is not None:
        for i, (expected, actual) in enumerate(zip(shape, arr.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"{name} has wrong shape at dimension {i}",
                    param_name=name,
                    expected=f"shape {shape}",
                    actual=f"shape {arr.shape}"
                )
    
    # Check minimum samples
    if min_samples is not None and arr.shape[0] < min_samples:
        raise ValidationError(
            f"{name} must have at least {min_samples} samples",
            param_name=name,
            expected=f">= {min_samples} samples",
            actual=f"{arr.shape[0]} samples"
        )
    
    # Check minimum features
    if min_features is not None and arr.ndim >= 2 and arr.shape[1] < min_features:
        raise ValidationError(
            f"{name} must have at least {min_features} features",
            param_name=name,
            expected=f">= {min_features} features",
            actual=f"{arr.shape[1]} features"
        )
    
    # Check for NaN
    if not allow_nan and np.isnan(arr).any():
        n_nan = np.isnan(arr).sum()
        nan_ratio = n_nan / arr.size
        # Find first few locations
        nan_locs = np.argwhere(np.isnan(arr))[:3]
        raise ValidationError(
            f"{name} contains {n_nan} NaN values ({nan_ratio:.1%} of data). "
            f"First locations: {nan_locs.tolist()}",
            param_name=name,
            expected="no NaN values",
            actual=f"{n_nan} NaN values"
        )
    
    # Check for Inf
    if not allow_inf and np.isinf(arr).any():
        n_inf = np.isinf(arr).sum()
        raise ValidationError(
            f"{name} contains {n_inf} infinite values",
            param_name=name,
            expected="no infinite values",
            actual=f"{n_inf} infinite values"
        )
    
    # Convert dtype if needed
    if dtype is not None and arr.dtype != dtype:
        try:
            arr = arr.astype(dtype)
        except Exception as e:
            raise ValidationError(
                f"Cannot convert {name} to {dtype}: {e}",
                param_name=name,
                expected=f"dtype {dtype}",
                actual=f"dtype {arr.dtype}"
            )
    
    # Copy if requested
    if copy:
        arr = arr.copy()
    
    return arr


def validate_array_pair(
    X: Any,
    y: Any,
    X_name: str = "X",
    y_name: str = "y",
    check_length: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate a pair of arrays (e.g., features and labels).
    
    Args:
        X: Feature array
        y: Label array
        X_name: Name for X in error messages
        y_name: Name for y in error messages
        check_length: Whether to verify matching lengths
    
    Returns:
        Tuple of validated (X, y)
    """
    X = validate_array(X, X_name, ndim=2)
    y = validate_array(y, y_name, ndim=1)
    
    if check_length and len(X) != len(y):
        raise ValidationError(
            f"Length mismatch between {X_name} and {y_name}",
            expected=f"matching lengths",
            actual=f"{X_name} has {len(X)} samples, {y_name} has {len(y)}"
        )
    
    return X, y


# =============================================================================
# Tensor Validation
# =============================================================================

def validate_tensor(
    tensor: Any,
    name: str = "tensor",
    ndim: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: Optional[bool] = None
) -> torch.Tensor:
    """
    Validate PyTorch tensor.
    
    Args:
        tensor: Tensor to validate
        name: Name for error messages
        ndim: Required number of dimensions
        device: Required device (will move if different)
        dtype: Required dtype (will convert if different)
        requires_grad: Required requires_grad setting
    
    Returns:
        Validated tensor (possibly moved/converted)
    
    Raises:
        ValidationError: If validation fails
    """
    if tensor is None:
        raise ValidationError(
            f"{name} cannot be None",
            param_name=name,
            expected="torch.Tensor"
        )
    
    if not isinstance(tensor, torch.Tensor):
        # Try to convert
        try:
            tensor = torch.as_tensor(tensor)
        except Exception as e:
            raise ValidationError(
                f"{name} must be torch.Tensor",
                param_name=name,
                expected="torch.Tensor",
                actual=type(tensor).__name__
            )
    
    if ndim is not None and tensor.dim() != ndim:
        raise ValidationError(
            f"{name} must be {ndim}D",
            param_name=name,
            expected=f"{ndim}D tensor",
            actual=f"{tensor.dim()}D tensor with shape {tensor.shape}"
        )
    
    if torch.isnan(tensor).any():
        n_nan = torch.isnan(tensor).sum().item()
        raise ValidationError(
            f"{name} contains {n_nan} NaN values",
            param_name=name
        )
    
    if torch.isinf(tensor).any():
        n_inf = torch.isinf(tensor).sum().item()
        raise ValidationError(
            f"{name} contains {n_inf} infinite values",
            param_name=name
        )
    
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    
    if device is not None and tensor.device != device:
        tensor = tensor.to(device)
    
    if requires_grad is not None and tensor.requires_grad != requires_grad:
        tensor = tensor.requires_grad_(requires_grad)
    
    return tensor


# =============================================================================
# Value Validation
# =============================================================================

def validate_positive(
    value: Union[int, float],
    name: str,
    allow_zero: bool = False
) -> Union[int, float]:
    """
    Validate that value is positive.
    
    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether to allow zero
    
    Returns:
        Validated value
    """
    if allow_zero:
        if value < 0:
            raise ValidationError(
                f"{name} must be non-negative",
                param_name=name,
                expected=">= 0",
                actual=value
            )
    else:
        if value <= 0:
            raise ValidationError(
                f"{name} must be positive",
                param_name=name,
                expected="> 0",
                actual=value
            )
    return value


def validate_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    inclusive: bool = True
) -> Union[int, float]:
    """
    Validate that value is within range.
    
    Args:
        value: Value to validate
        name: Name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        inclusive: Whether bounds are inclusive
    
    Returns:
        Validated value
    """
    if inclusive:
        if min_val is not None and value < min_val:
            raise ValidationError(
                f"{name} must be >= {min_val}",
                param_name=name,
                expected=f">= {min_val}",
                actual=value
            )
        if max_val is not None and value > max_val:
            raise ValidationError(
                f"{name} must be <= {max_val}",
                param_name=name,
                expected=f"<= {max_val}",
                actual=value
            )
    else:
        if min_val is not None and value <= min_val:
            raise ValidationError(
                f"{name} must be > {min_val}",
                param_name=name,
                expected=f"> {min_val}",
                actual=value
            )
        if max_val is not None and value >= max_val:
            raise ValidationError(
                f"{name} must be < {max_val}",
                param_name=name,
                expected=f"< {max_val}",
                actual=value
            )
    return value


def validate_probability(
    value: float,
    name: str,
    allow_zero: bool = True,
    allow_one: bool = True
) -> float:
    """
    Validate that value is a valid probability.
    
    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether 0.0 is valid
        allow_one: Whether 1.0 is valid
    
    Returns:
        Validated value
    """
    min_val = 0.0 if allow_zero else 0.0 + 1e-10
    max_val = 1.0 if allow_one else 1.0 - 1e-10
    
    if value < 0.0 or value > 1.0:
        raise ValidationError(
            f"{name} must be between 0 and 1",
            param_name=name,
            expected="probability in [0, 1]",
            actual=value
        )
    
    if not allow_zero and value == 0.0:
        raise ValidationError(
            f"{name} cannot be zero",
            param_name=name,
            expected="probability in (0, 1]",
            actual=value
        )
    
    if not allow_one and value == 1.0:
        raise ValidationError(
            f"{name} cannot be one",
            param_name=name,
            expected="probability in [0, 1)",
            actual=value
        )
    
    return value


def validate_choice(
    value: Any,
    choices: Sequence[Any],
    name: str,
    case_sensitive: bool = True
) -> Any:
    """
    Validate that value is one of the allowed choices.
    
    Args:
        value: Value to validate
        choices: Allowed values
        name: Name for error messages
        case_sensitive: Whether string comparison is case-sensitive
    
    Returns:
        Validated value (possibly normalized)
    """
    # Handle case-insensitive string comparison
    if not case_sensitive and isinstance(value, str):
        value_lower = value.lower()
        for choice in choices:
            if isinstance(choice, str) and choice.lower() == value_lower:
                return choice
    
    if value not in choices:
        choices_str = ", ".join(repr(c) for c in choices[:10])
        if len(choices) > 10:
            choices_str += f", ... ({len(choices)} total)"
        raise ValidationError(
            f"{name} must be one of [{choices_str}]",
            param_name=name,
            expected=f"one of {list(choices)[:5]}",
            actual=repr(value)
        )
    
    return value


def validate_type(
    value: Any,
    expected_types: Union[Type, Tuple[Type, ...]],
    name: str,
    allow_none: bool = False
) -> Any:
    """
    Validate that value is of expected type(s).
    
    Args:
        value: Value to validate
        expected_types: Expected type or tuple of types
        name: Name for error messages
        allow_none: Whether None is valid
    
    Returns:
        Validated value
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(
            f"{name} cannot be None",
            param_name=name,
            expected=str(expected_types)
        )
    
    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            expected_str = " or ".join(t.__name__ for t in expected_types)
        else:
            expected_str = expected_types.__name__
        
        raise ValidationError(
            f"{name} must be {expected_str}",
            param_name=name,
            expected=expected_str,
            actual=type(value).__name__
        )
    
    return value


# =============================================================================
# Path Validation
# =============================================================================

def validate_path(
    path: Union[str, Path],
    name: str = "path",
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_parents: bool = False,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate and convert path.
    
    Args:
        path: Path to validate
        name: Name for error messages
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
        create_parents: Whether to create parent directories
        allowed_extensions: List of allowed file extensions
    
    Returns:
        Validated Path object
    """
    if path is None:
        raise ValidationError(f"{name} cannot be None", param_name=name)
    
    path = Path(path)
    
    if must_exist and not path.exists():
        raise ValidationError(
            f"{name} does not exist",
            param_name=name,
            expected="existing path",
            actual=str(path)
        )
    
    if must_be_file and path.exists() and not path.is_file():
        raise ValidationError(
            f"{name} must be a file",
            param_name=name,
            expected="file path",
            actual=f"directory: {path}"
        )
    
    if must_be_dir and path.exists() and not path.is_dir():
        raise ValidationError(
            f"{name} must be a directory",
            param_name=name,
            expected="directory path",
            actual=f"file: {path}"
        )
    
    if allowed_extensions:
        ext = path.suffix.lower()
        allowed = [e.lower() if e.startswith('.') else f'.{e.lower()}' for e in allowed_extensions]
        if ext not in allowed:
            raise ValidationError(
                f"{name} must have extension in {allowed_extensions}",
                param_name=name,
                expected=f"extension in {allowed_extensions}",
                actual=f"extension '{ext}'"
            )
    
    if create_parents and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


# =============================================================================
# Label Validation
# =============================================================================

def validate_labels(
    labels: np.ndarray,
    n_samples: int,
    name: str = "labels",
    n_classes_range: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Validate label array.
    
    Args:
        labels: Labels to validate
        n_samples: Expected number of samples
        name: Name for error messages
        n_classes_range: (min, max) number of classes
    
    Returns:
        Validated labels
    """
    labels = validate_array(labels, name, ndim=1)
    
    if len(labels) != n_samples:
        raise ValidationError(
            f"{name} length doesn't match n_samples",
            param_name=name,
            expected=f"length {n_samples}",
            actual=f"length {len(labels)}"
        )
    
    n_classes = len(np.unique(labels))
    
    if n_classes_range is not None:
        min_classes, max_classes = n_classes_range
        if n_classes < min_classes:
            raise ValidationError(
                f"{name} must have at least {min_classes} classes",
                param_name=name,
                expected=f">= {min_classes} classes",
                actual=f"{n_classes} classes"
            )
        if n_classes > max_classes:
            raise ValidationError(
                f"{name} must have at most {max_classes} classes",
                param_name=name,
                expected=f"<= {max_classes} classes",
                actual=f"{n_classes} classes"
            )
    
    return labels


# =============================================================================
# Decorators
# =============================================================================

def validated(func: F) -> F:
    """
    Decorator that adds function context to ValidationErrors.
    
    Example:
        >>> @validated
        ... def process_data(X, y):
        ...     X = validate_array(X, "X", ndim=2)
        ...     return X + 1
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            # Add function context
            raise ValidationError(
                f"In {func.__qualname__}(): {e.message}",
                param_name=e.param_name,
                expected=e.expected,
                actual=e.actual
            ) from None
    return wrapper


def validate_inputs(**param_specs):
    """
    Decorator for declarative input validation.
    
    Args:
        **param_specs: Parameter name -> validation spec dict
    
    Example:
        >>> @validate_inputs(
        ...     X={'ndim': 2, 'min_samples': 10},
        ...     learning_rate={'type': float, 'range': (0, 1)}
        ... )
        ... def train(X, learning_rate=0.01):
        ...     pass
    """
    def decorator(func: F) -> F:
        sig = inspect.signature(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to parameters
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each specified parameter
            for param_name, spec in param_specs.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    
                    # Handle array validation
                    if 'ndim' in spec or 'min_samples' in spec:
                        value = validate_array(
                            value,
                            param_name,
                            ndim=spec.get('ndim'),
                            min_samples=spec.get('min_samples'),
                            min_features=spec.get('min_features'),
                            allow_nan=spec.get('allow_nan', False)
                        )
                    
                    # Handle type validation
                    if 'type' in spec:
                        validate_type(value, spec['type'], param_name, allow_none=spec.get('allow_none', False))
                    
                    # Handle range validation
                    if 'range' in spec:
                        min_val, max_val = spec['range']
                        validate_range(value, param_name, min_val, max_val)
                    
                    # Handle choice validation
                    if 'choices' in spec:
                        validate_choice(value, spec['choices'], param_name)
                    
                    bound.arguments[param_name] = value
            
            return func(*bound.args, **bound.kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# Utility Functions
# =============================================================================

def check_is_fitted(
    estimator: Any,
    attributes: Union[str, List[str]],
    msg: Optional[str] = None
) -> None:
    """
    Check if estimator is fitted by verifying attribute existence.
    
    Args:
        estimator: Estimator to check
        attributes: Attribute(s) that should exist if fitted
        msg: Custom error message
    
    Raises:
        ValidationError: If estimator is not fitted
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    
    missing = [attr for attr in attributes if not hasattr(estimator, attr)]
    
    if missing:
        if msg is None:
            msg = (
                f"{type(estimator).__name__} is not fitted. "
                f"Missing attributes: {missing}"
            )
        raise ValidationError(msg)


def assert_all_finite(
    X: np.ndarray,
    name: str = "input"
) -> None:
    """
    Assert that array contains no NaN or Inf values.
    
    Args:
        X: Array to check
        name: Name for error messages
    
    Raises:
        ValidationError: If array contains non-finite values
    """
    if not np.isfinite(X).all():
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        raise ValidationError(
            f"{name} contains non-finite values: {n_nan} NaN, {n_inf} Inf"
        )


def coerce_to_array(
    X: Any,
    dtype: Optional[type] = None,
    copy: bool = False
) -> np.ndarray:
    """
    Safely coerce input to numpy array.
    
    Args:
        X: Input to convert
        dtype: Target dtype
        copy: Whether to force a copy
    
    Returns:
        Numpy array
    """
    import pandas as pd
    
    if isinstance(X, np.ndarray):
        arr = X.copy() if copy else X
    elif isinstance(X, pd.DataFrame):
        arr = X.values
    elif isinstance(X, pd.Series):
        arr = X.values
    elif isinstance(X, torch.Tensor):
        arr = X.detach().cpu().numpy()
    elif isinstance(X, (list, tuple)):
        arr = np.array(X)
    else:
        arr = np.asarray(X)
    
    if dtype is not None:
        arr = arr.astype(dtype)
    
    return arr
