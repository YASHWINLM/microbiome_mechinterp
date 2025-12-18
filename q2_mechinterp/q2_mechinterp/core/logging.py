"""
Structured logging utilities for experiment tracking.

This module provides:
- StructuredLogger for capturing experiment metrics
- Timer context manager for profiling
- Model summary utilities
- Progress bar abstraction

Design Principles:
- Non-intrusive: Logging should not affect experiment behavior
- Flexible outputs: Console, file, JSON, CSV
- Machine-readable: Structured format for downstream analysis
"""

import logging
import time
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Iterator
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field, asdict
import numpy as np


# =============================================================================
# Structured Logger
# =============================================================================

@dataclass
class MetricRecord:
    """Single metric recording."""
    step: Optional[int]
    epoch: Optional[int]
    timestamp: float
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuredLogger:
    """
    Logger with structured output for ML experiments.
    
    Captures parameters, metrics, and timing information in a structured
    format suitable for downstream analysis and visualization.
    
    Example:
        >>> logger = StructuredLogger("my_experiment", log_dir="./logs")
        >>> logger.log_params({'latent_dim': 50, 'beta': 0.01})
        >>> 
        >>> for epoch in range(100):
        ...     train_loss = train_epoch()
        ...     logger.log_metrics({'train_loss': train_loss}, epoch=epoch)
        >>> 
        >>> logger.save_history()
    
    Attributes:
        name: Experiment name
        log_dir: Directory for log files
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: int = logging.INFO,
        console: bool = True,
        file_logging: bool = True,
        timestamp_format: str = '%Y%m%d_%H%M%S'
    ):
        """
        Initialize logger.
        
        Args:
            name: Experiment name
            log_dir: Directory for log files (None disables file logging)
            level: Logging level
            console: Whether to log to console
            file_logging: Whether to log to file
            timestamp_format: Format for timestamps in filenames
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.timestamp = datetime.now().strftime(timestamp_format)
        
        # Setup Python logger
        self.logger = logging.getLogger(f"q2_mechinterp.{name}")
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # File handler
        if file_logging and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f'{name}_{self.timestamp}.log'
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # Structured data storage
        self._params: Dict[str, Any] = {}
        self._metrics_history: List[MetricRecord] = []
        self._artifacts: Dict[str, Path] = {}
        self._start_time = time.time()
        self._step = 0
        
        self.info(f"Initialized logger: {name}")
    
    # =========================================================================
    # Basic Logging
    # =========================================================================
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)
    
    # =========================================================================
    # Parameter Logging
    # =========================================================================
    
    def log_params(self, params: Dict[str, Any], prefix: str = "") -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameter names and values
            prefix: Optional prefix for parameter names
        """
        if prefix:
            params = {f"{prefix}.{k}": v for k, v in params.items()}
        
        self._params.update(params)
        
        # Log as formatted string
        params_str = ', '.join(f'{k}={v}' for k, v in params.items())
        self.logger.info(f"Parameters: {params_str}")
    
    def log_config(self, config: Any) -> None:
        """
        Log a configuration object.
        
        Args:
            config: Config object with to_dict() method or dict
        """
        if hasattr(config, 'to_dict'):
            params = config.to_dict()
        elif isinstance(config, dict):
            params = config
        else:
            params = {'config': str(config)}
        
        self.log_params(params)
    
    # =========================================================================
    # Metric Logging
    # =========================================================================
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step number (auto-incremented if None)
            epoch: Epoch number
            prefix: Optional prefix for metric names
        """
        if step is None:
            step = self._step
            self._step += 1
        
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Store record
        record = MetricRecord(
            step=step,
            epoch=epoch,
            timestamp=time.time() - self._start_time,
            metrics=metrics
        )
        self._metrics_history.append(record)
        
        # Format for logging
        metrics_str = ', '.join(
            f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
            for k, v in metrics.items()
        )
        
        prefix_str = ""
        if epoch is not None:
            prefix_str = f"Epoch {epoch}"
        elif step is not None:
            prefix_str = f"Step {step}"
        
        if prefix_str:
            self.logger.info(f"{prefix_str}: {metrics_str}")
        else:
            self.logger.info(metrics_str)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        **extra_metrics
    ) -> None:
        """
        Convenience method for logging epoch metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            **extra_metrics: Additional metrics to log
        """
        metrics = {'train_loss': train_loss}
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        metrics.update(extra_metrics)
        
        self.log_metrics(metrics, epoch=epoch)
    
    # =========================================================================
    # History & Export
    # =========================================================================
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of metric to retrieve
        
        Returns:
            List of metric values over time
        """
        return [
            record.metrics.get(metric_name)
            for record in self._metrics_history
            if metric_name in record.metrics
        ]
    
    def get_best_metric(self, metric_name: str, mode: str = 'min') -> Dict[str, Any]:
        """
        Get best value of a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'min' or 'max'
        
        Returns:
            Dict with best value, step, and epoch
        """
        values = []
        for record in self._metrics_history:
            if metric_name in record.metrics:
                values.append((record.metrics[metric_name], record.step, record.epoch))
        
        if not values:
            return {'value': None, 'step': None, 'epoch': None}
        
        if mode == 'min':
            best = min(values, key=lambda x: x[0])
        else:
            best = max(values, key=lambda x: x[0])
        
        return {'value': best[0], 'step': best[1], 'epoch': best[2]}
    
    def save_history(
        self,
        path: Optional[Union[str, Path]] = None,
        format: str = 'json'
    ) -> Path:
        """
        Save experiment history to file.
        
        Args:
            path: Output path (auto-generated if None)
            format: 'json' or 'csv'
        
        Returns:
            Path to saved file
        """
        if path is None:
            if self.log_dir is None:
                self.log_dir = Path('./logs')
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ext = '.json' if format == 'json' else '.csv'
            path = self.log_dir / f'{self.name}_{self.timestamp}_history{ext}'
        else:
            path = Path(path)
        
        if format == 'json':
            data = {
                'name': self.name,
                'timestamp': self.timestamp,
                'params': self._params,
                'metrics': [r.to_dict() for r in self._metrics_history],
                'total_time': time.time() - self._start_time
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            
            # Collect all metric names
            all_metrics = set()
            for record in self._metrics_history:
                all_metrics.update(record.metrics.keys())
            
            fieldnames = ['step', 'epoch', 'timestamp'] + sorted(all_metrics)
            
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in self._metrics_history:
                    row = {
                        'step': record.step,
                        'epoch': record.epoch,
                        'timestamp': record.timestamp,
                        **record.metrics
                    }
                    writer.writerow(row)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        self.info(f"History saved to {path}")
        return path
    
    def save_params(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save parameters to JSON file."""
        if path is None:
            if self.log_dir is None:
                self.log_dir = Path('./logs')
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f'{self.name}_{self.timestamp}_params.json'
        else:
            path = Path(path)
        
        with open(path, 'w') as f:
            json.dump(self._params, f, indent=2, default=str)
        
        return path
    
    # =========================================================================
    # Artifact Tracking
    # =========================================================================
    
    def log_artifact(self, name: str, path: Union[str, Path]) -> None:
        """
        Register an artifact (saved file).
        
        Args:
            name: Artifact name
            path: Path to artifact file
        """
        self._artifacts[name] = Path(path)
        self.info(f"Artifact '{name}': {path}")
    
    def get_artifacts(self) -> Dict[str, Path]:
        """Get all registered artifacts."""
        return dict(self._artifacts)
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    def summary(self) -> str:
        """
        Generate experiment summary.
        
        Returns:
            Formatted summary string
        """
        lines = [
            f"\n{'='*60}",
            f"Experiment Summary: {self.name}",
            f"{'='*60}",
            f"Duration: {time.time() - self._start_time:.2f} seconds",
            f"Total steps: {len(self._metrics_history)}",
        ]
        
        # Parameters
        if self._params:
            lines.append("\nParameters:")
            for k, v in list(self._params.items())[:10]:
                lines.append(f"  {k}: {v}")
            if len(self._params) > 10:
                lines.append(f"  ... ({len(self._params) - 10} more)")
        
        # Best metrics
        if self._metrics_history:
            lines.append("\nBest Metrics:")
            # Find common metrics
            metric_names = set()
            for record in self._metrics_history:
                metric_names.update(record.metrics.keys())
            
            for metric in sorted(metric_names):
                if 'loss' in metric.lower():
                    best = self.get_best_metric(metric, mode='min')
                else:
                    best = self.get_best_metric(metric, mode='max')
                if best['value'] is not None:
                    lines.append(f"  {metric}: {best['value']:.4f} (epoch {best['epoch']})")
        
        # Artifacts
        if self._artifacts:
            lines.append("\nArtifacts:")
            for name, path in self._artifacts.items():
                lines.append(f"  {name}: {path}")
        
        lines.append(f"{'='*60}\n")
        
        return '\n'.join(lines)
    
    def print_summary(self) -> None:
        """Print experiment summary."""
        print(self.summary())


# =============================================================================
# Timer Context Manager
# =============================================================================

@contextmanager
def timer(
    name: str = "Operation",
    logger: Optional[StructuredLogger] = None,
    log_level: str = 'info'
) -> Iterator[Dict[str, float]]:
    """
    Context manager for timing operations.
    
    Args:
        name: Name of operation being timed
        logger: Optional logger for output
        log_level: Log level ('info', 'debug', 'warning')
    
    Yields:
        Dictionary that will contain 'elapsed' after context exits
    
    Example:
        >>> with timer("Training epoch"):
        ...     train_epoch()
        Training epoch completed in 45.23 seconds
        
        >>> with timer("Forward pass") as t:
        ...     output = model(input)
        >>> print(f"Took {t['elapsed']:.2f}s")
    """
    result = {}
    start = time.time()
    
    try:
        yield result
    finally:
        elapsed = time.time() - start
        result['elapsed'] = elapsed
        
        msg = f"{name} completed in {elapsed:.2f} seconds"
        
        if logger:
            getattr(logger, log_level)(msg)
        else:
            print(msg)


class Timer:
    """
    Reusable timer for profiling multiple operations.
    
    Example:
        >>> t = Timer()
        >>> t.start('forward')
        >>> output = model(input)
        >>> t.stop('forward')
        >>> 
        >>> t.start('backward')
        >>> loss.backward()
        >>> t.stop('backward')
        >>> 
        >>> print(t.summary())
    """
    
    def __init__(self):
        self._starts: Dict[str, float] = {}
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
    
    def start(self, name: str) -> None:
        """Start timing an operation."""
        self._starts[name] = time.time()
    
    def stop(self, name: str) -> float:
        """Stop timing an operation and return elapsed time."""
        if name not in self._starts:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self._starts[name]
        
        if name not in self._totals:
            self._totals[name] = 0
            self._counts[name] = 0
        
        self._totals[name] += elapsed
        self._counts[name] += 1
        
        del self._starts[name]
        return elapsed
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset timer(s)."""
        if name:
            self._totals.pop(name, None)
            self._counts.pop(name, None)
            self._starts.pop(name, None)
        else:
            self._totals.clear()
            self._counts.clear()
            self._starts.clear()
    
    def get_average(self, name: str) -> float:
        """Get average time for an operation."""
        if name not in self._totals:
            return 0.0
        return self._totals[name] / self._counts[name]
    
    def get_total(self, name: str) -> float:
        """Get total time for an operation."""
        return self._totals.get(name, 0.0)
    
    def summary(self) -> str:
        """Get timing summary."""
        lines = ["Timing Summary:"]
        for name in sorted(self._totals.keys()):
            avg = self.get_average(name)
            total = self._totals[name]
            count = self._counts[name]
            lines.append(f"  {name}: {avg:.4f}s avg, {total:.2f}s total ({count} calls)")
        return '\n'.join(lines)


# =============================================================================
# Model Summary
# =============================================================================

def print_model_summary(
    model,
    input_shape: Optional[tuple] = None,
    show_params: bool = True,
    show_layers: bool = True
) -> str:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape (for activation sizes)
        show_params: Show parameter counts
        show_layers: Show layer breakdown
    
    Returns:
        Summary string
    """
    import torch.nn as nn
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines = [
        f"\n{'='*60}",
        f"Model: {model.__class__.__name__}",
        f"{'='*60}",
    ]
    
    if show_params:
        lines.extend([
            f"Total parameters:     {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable:        {total_params - trainable_params:,}",
        ])
    
    if show_layers:
        lines.append(f"{'='*60}")
        lines.append(f"{'Layer':<30} {'Type':<20} {'Params':>10}")
        lines.append(f"{'-'*60}")
        
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            module_type = module.__class__.__name__
            
            # Truncate long names
            if len(name) > 28:
                name = name[:25] + "..."
            if len(module_type) > 18:
                module_type = module_type[:15] + "..."
            
            lines.append(f"{name:<30} {module_type:<20} {params:>10,}")
    
    lines.append(f"{'='*60}\n")
    
    summary = '\n'.join(lines)
    print(summary)
    return summary


def count_parameters(model, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
    
    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Progress Bar Abstraction
# =============================================================================

def get_progress_bar(
    iterable,
    desc: str = "",
    total: Optional[int] = None,
    disable: bool = False,
    leave: bool = True,
    **kwargs
):
    """
    Get a progress bar (tqdm if available, else simple fallback).
    
    Args:
        iterable: Iterable to wrap
        desc: Description for progress bar
        total: Total number of items
        disable: Whether to disable progress bar
        leave: Whether to leave progress bar after completion
        **kwargs: Additional arguments for tqdm
    
    Returns:
        Wrapped iterable with progress bar
    """
    if disable:
        return iterable
    
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=leave, **kwargs)
    except ImportError:
        # Simple fallback
        return _SimpleProgressBar(iterable, desc, total)


class _SimpleProgressBar:
    """Simple fallback progress bar when tqdm is not available."""
    
    def __init__(self, iterable, desc: str = "", total: Optional[int] = None):
        self.iterable = iterable
        self.desc = desc
        self.total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
        self.n = 0
        self._last_print = 0
    
    def __iter__(self):
        for item in self.iterable:
            yield item
            self.n += 1
            self._update()
    
    def _update(self):
        # Print progress every 10% or so
        if self.total:
            progress = self.n / self.total
            if progress - self._last_print >= 0.1:
                print(f"\r{self.desc}: {self.n}/{self.total} ({progress:.0%})", end='', flush=True)
                self._last_print = progress
        
        # Print newline at end
        if self.total and self.n >= self.total:
            print()
    
    def set_postfix(self, **kwargs):
        """Compatibility method (no-op in fallback)."""
        pass
    
    def update(self, n: int = 1):
        """Compatibility method."""
        self.n += n
        self._update()


# =============================================================================
# Convenience Functions
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Setup basic logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name."""
    return logging.getLogger(f"q2_mechinterp.{name}")
