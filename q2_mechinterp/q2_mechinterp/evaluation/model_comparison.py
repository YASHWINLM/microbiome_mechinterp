"""
Model comparison utilities with statistical significance testing.

This module provides tools for:
- Comparing multiple VAE architectures/configurations
- Statistical significance testing (paired t-test, Wilcoxon, etc.)
- Effect size calculation (Cohen's d, Cliff's delta)

Design Principles:
- Rigorous: Proper statistical testing with multiple run aggregation
- Reproducible: Fixed seeds for all randomization
- Informative: Rich output with significance levels and effect sizes
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Type, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import warnings
import time


@dataclass
class ModelComparisonResult:
    """
    Results from model architecture comparison.

    Attributes:
        summary_df: DataFrame summarizing all configurations
        detailed_results: Detailed results per configuration
        best_model: Name of best performing configuration
        best_config: Configuration dict for best model
        statistical_tests: Pairwise statistical significance tests
        ranking: Ordered list of configurations by performance
    """

    summary_df: pd.DataFrame
    detailed_results: Dict[str, Any]
    best_model: str
    best_config: Dict[str, Any]
    statistical_tests: Optional[Dict[str, Dict[str, float]]] = None
    ranking: Optional[List[str]] = None

    def __repr__(self) -> str:
        return (
            f"ModelComparisonResult(best={self.best_model}, "
            f"n_configs={len(self.detailed_results)})"
        )

    def get_significant_differences(self, alpha: float = 0.05) -> List[Tuple[str, str, float]]:
        """
        Get pairs with statistically significant differences.

        Args:
            alpha: Significance level

        Returns:
            List of (config1, config2, p_value) tuples
        """
        if self.statistical_tests is None:
            return []

        significant = []
        for pair, results in self.statistical_tests.items():
            if results.get("t_pvalue", 1.0) < alpha:
                configs = pair.split("_vs_")
                significant.append((configs[0], configs[1], results["t_pvalue"]))

        return significant

    def format_summary(self) -> str:
        """Format results as a readable summary."""
        lines = [
            "=" * 60,
            "Model Comparison Summary",
            "=" * 60,
            "",
            f"Best Configuration: {self.best_model}",
            f"Number of Configurations: {len(self.detailed_results)}",
            "",
            "Rankings:",
        ]

        if self.ranking:
            for i, name in enumerate(self.ranking, 1):
                result = self.detailed_results[name]
                lines.append(f"  {i}. {name}: {result['mean_loss']:.4f} ± {result['std_loss']:.4f}")

        if self.statistical_tests:
            sig_diffs = self.get_significant_differences()
            if sig_diffs:
                lines.extend(
                    [
                        "",
                        "Significant Differences (p < 0.05):",
                    ]
                )
                for c1, c2, p in sig_diffs:
                    lines.append(f"  {c1} vs {c2}: p = {p:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class VAEComparator:
    """
    Compare multiple VAE architectures with statistical significance testing.

    Example:
        >>> comparator = VAEComparator(device=device)
        >>> configs = [
        ...     {'name': 'small', 'latent_dim': 20, 'hidden_dims': [128, 64]},
        ...     {'name': 'large', 'latent_dim': 100, 'hidden_dims': [512, 256, 128]},
        ... ]
        >>> results = comparator.compare_architectures(
        ...     model_class=MicrobiomeVAE,
        ...     configs=configs,
        ...     train_data=X_train,
        ...     val_data=X_val,
        ...     n_runs=5
        ... )
        >>> print(results.format_summary())
    """

    def __init__(self, device: torch.device, random_state: int = 42, verbose: bool = True):
        """
        Initialize comparator.

        Args:
            device: PyTorch device for training
            random_state: Base random seed (each run uses seed + run_index)
            verbose: Whether to print progress
        """
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def compare_architectures(
        self,
        model_class: Type,
        configs: List[Dict[str, Any]],
        train_data: np.ndarray,
        val_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        n_runs: int = 3,
        training_kwargs: Optional[Dict[str, Any]] = None,
        early_stopping: bool = True,
    ) -> ModelComparisonResult:
        """
        Compare different model configurations with multiple runs.

        Args:
            model_class: VAE model class to instantiate
            configs: List of configuration dicts (must include 'name' key)
            train_data: Training data array
            val_data: Validation data array
            epochs: Training epochs per run
            batch_size: Batch size
            n_runs: Number of runs per configuration
            training_kwargs: Additional training arguments
            early_stopping: Whether to use early stopping

        Returns:
            ModelComparisonResult with summary, details, and statistical tests
        """
        from q2_mechinterp.training import VAETrainer

        training_kwargs = training_kwargs or {}
        input_dim = train_data.shape[1]

        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        all_results = {}

        for config in configs:
            config_name = config.get("name", f"config_{len(all_results)}")

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Configuration: {config_name}")
                print(f"Parameters: {config}")
                print(f"{'='*50}")

            run_losses = []
            run_times = []
            run_epochs = []

            for run in range(n_runs):
                if self.verbose:
                    print(f"  Run {run + 1}/{n_runs}...", end=" ")

                seed = self.random_state + run
                torch.manual_seed(seed)
                np.random.seed(seed)

                model_config = {"input_dim": input_dim}
                model_config.update({k: v for k, v in config.items() if k != "name"})

                try:
                    model = model_class(**model_config).to(self.device)
                except Exception as e:
                    warnings.warn(f"Failed to create model: {e}")
                    run_losses.append(float("inf"))
                    continue

                trainer = VAETrainer(model, self.device, lr=training_kwargs.get("lr", 1e-3))

                start_time = time.time()
                _, val_losses = trainer.train(
                    train_loader,
                    val_loader,
                    epochs=epochs,
                    verbose=0,
                    patience=15 if early_stopping else epochs,
                    **{k: v for k, v in training_kwargs.items() if k != "lr"},
                )
                run_time = time.time() - start_time

                best_val = min(val_losses) if val_losses else float("inf")
                run_losses.append(best_val)
                run_times.append(run_time)
                run_epochs.append(len(val_losses))

                if self.verbose:
                    print(f"Best val loss: {best_val:.4f} ({len(val_losses)} epochs)")

            all_results[config_name] = {
                "config": config,
                "run_losses": np.array(run_losses),
                "run_times": np.array(run_times),
                "run_epochs": np.array(run_epochs),
                "mean_loss": np.mean(run_losses),
                "std_loss": np.std(run_losses),
                "mean_time": np.mean(run_times),
                "n_runs": len(run_losses),
            }

        # Create summary DataFrame
        summary_data = []
        for name, res in all_results.items():
            summary_data.append(
                {
                    "config_name": name,
                    "mean_val_loss": res["mean_loss"],
                    "std_val_loss": res["std_loss"],
                    "mean_time": res["mean_time"],
                    "n_runs": res["n_runs"],
                    **{k: v for k, v in res["config"].items() if k != "name"},
                }
            )

        summary_df = pd.DataFrame(summary_data).sort_values("mean_val_loss")
        best_name = summary_df.iloc[0]["config_name"]
        ranking = summary_df["config_name"].tolist()
        stat_tests = self._compute_statistical_tests(all_results)

        return ModelComparisonResult(
            summary_df=summary_df,
            detailed_results=all_results,
            best_model=best_name,
            best_config=all_results[best_name]["config"],
            statistical_tests=stat_tests,
            ranking=ranking,
        )

    def _compute_statistical_tests(self, results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Compute pairwise statistical tests between configurations."""
        from scipy import stats

        names = list(results.keys())
        tests = {}

        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                losses1 = results[name1]["run_losses"]
                losses2 = results[name2]["run_losses"]

                valid1 = ~np.isinf(losses1) & ~np.isnan(losses1)
                valid2 = ~np.isinf(losses2) & ~np.isnan(losses2)

                if valid1.sum() < 2 or valid2.sum() < 2:
                    continue

                losses1_clean = losses1[valid1]
                losses2_clean = losses2[valid2]

                min_len = min(len(losses1_clean), len(losses2_clean))
                losses1_paired = losses1_clean[:min_len]
                losses2_paired = losses2_clean[:min_len]

                test_results = {}

                # Paired t-test
                if min_len >= 2:
                    t_stat, t_pval = stats.ttest_rel(losses1_paired, losses2_paired)
                    test_results["t_statistic"] = float(t_stat)
                    test_results["t_pvalue"] = float(t_pval)

                # Wilcoxon signed-rank test
                if min_len >= 5:
                    try:
                        w_stat, w_pval = stats.wilcoxon(losses1_paired, losses2_paired)
                        test_results["wilcoxon_statistic"] = float(w_stat)
                        test_results["wilcoxon_pvalue"] = float(w_pval)
                    except Exception:
                        test_results["wilcoxon_statistic"] = np.nan
                        test_results["wilcoxon_pvalue"] = np.nan

                # Cohen's d
                pooled_std = np.sqrt((np.var(losses1_clean) + np.var(losses2_clean)) / 2)
                if pooled_std > 0:
                    cohens_d = (np.mean(losses1_clean) - np.mean(losses2_clean)) / pooled_std
                else:
                    cohens_d = 0.0
                test_results["cohens_d"] = float(cohens_d)

                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    test_results["effect_size"] = "negligible"
                elif abs_d < 0.5:
                    test_results["effect_size"] = "small"
                elif abs_d < 0.8:
                    test_results["effect_size"] = "medium"
                else:
                    test_results["effect_size"] = "large"

                test_results["cliffs_delta"] = self._cliffs_delta(losses1_clean, losses2_clean)
                test_results["significant_005"] = test_results.get("t_pvalue", 1.0) < 0.05
                test_results["significant_001"] = test_results.get("t_pvalue", 1.0) < 0.01
                test_results["mean_diff"] = float(np.mean(losses1_clean) - np.mean(losses2_clean))

                tests[f"{name1}_vs_{name2}"] = test_results

        return tests

    @staticmethod
    def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(x), len(y)
        greater = 0
        less = 0

        for xi in x:
            for yj in y:
                if xi > yj:
                    greater += 1
                elif xi < yj:
                    less += 1

        return (greater - less) / (n1 * n2)

    def compare_hyperparameters(
        self,
        model_class: Type,
        base_config: Dict[str, Any],
        param_name: str,
        param_values: List[Any],
        train_data: np.ndarray,
        val_data: np.ndarray,
        n_runs: int = 3,
        **kwargs,
    ) -> ModelComparisonResult:
        """
        Compare different values of a single hyperparameter.

        Args:
            model_class: VAE model class
            base_config: Base configuration dict
            param_name: Name of parameter to vary
            param_values: Values to test
            train_data: Training data
            val_data: Validation data
            n_runs: Number of runs per value
            **kwargs: Additional arguments for compare_architectures

        Returns:
            ModelComparisonResult
        """
        configs = []
        for value in param_values:
            config = base_config.copy()
            config[param_name] = value
            config["name"] = f"{param_name}={value}"
            configs.append(config)

        return self.compare_architectures(
            model_class=model_class,
            configs=configs,
            train_data=train_data,
            val_data=val_data,
            n_runs=n_runs,
            **kwargs,
        )


class EnsembleComparator:
    """
    Compare ensemble methods for VAE features.

    Tests different ensemble strategies:
    - Feature averaging across runs
    - Feature concatenation across runs
    - Model selection (best single run)
    """

    def __init__(self, device: torch.device, random_state: int = 42):
        self.device = device
        self.random_state = random_state

    def compare_ensemble_strategies(
        self,
        model_class: Type,
        model_config: Dict[str, Any],
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        n_models: int = 5,
        epochs: int = 50,
        **training_kwargs,
    ) -> Dict[str, Any]:
        """
        Compare different ensemble strategies for VAE features.

        Args:
            model_class: VAE model class
            model_config: Model configuration
            train_data: Training data
            val_data: Validation data
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            n_models: Number of models in ensemble
            epochs: Training epochs
            **training_kwargs: Additional training arguments

        Returns:
            Dictionary with comparison results
        """
        from q2_mechinterp.training import VAETrainer
        from q2_mechinterp.extraction import FeatureExtractor
        from q2_mechinterp.evaluation.feature_evaluation import calculate_downstream_utility

        input_dim = train_data.shape[1]
        batch_size = training_kwargs.pop("batch_size", 32)

        train_tensor = torch.FloatTensor(train_data)
        val_tensor = torch.FloatTensor(val_data)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        models = []
        val_losses = []
        latent_features_list = []

        for i in range(n_models):
            seed = self.random_state + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = {"input_dim": input_dim, **model_config}
            model = model_class(**config).to(self.device)
            trainer = VAETrainer(model, self.device)

            _, v_losses = trainer.train(
                train_loader, val_loader, epochs=epochs, verbose=0, **training_kwargs
            )

            models.append(model)
            val_losses.append(min(v_losses) if v_losses else float("inf"))

            extractor = FeatureExtractor(model, self.device)
            latent = extractor.get_latent_features(test_data)
            latent_features_list.append(latent)

        # Strategy 1: Best single model
        best_idx = np.argmin(val_losses)
        features_best = latent_features_list[best_idx]

        # Strategy 2: Average features
        features_avg = np.mean(latent_features_list, axis=0)

        # Strategy 3: Concatenate features
        features_concat = np.hstack(latent_features_list)

        results = {}

        for name, features in [
            ("best_single", features_best),
            ("average", features_avg),
            ("concatenate", features_concat),
        ]:
            task = "classification" if len(np.unique(test_labels)) < 20 else "regression"
            utility = calculate_downstream_utility(features, test_labels, task=task, cv=5)
            results[name] = {"features_shape": features.shape, **utility}

        results["model_val_losses"] = val_losses
        results["best_model_idx"] = best_idx

        return results


# =============================================================================
# Utility Functions
# =============================================================================


def bootstrap_confidence_interval(
    scores: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000, random_state: int = 42
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for scores.

    Args:
        scores: Array of scores
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        (lower_bound, upper_bound) tuple
    """
    rng = np.random.RandomState(random_state)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def permutation_test(
    scores1: np.ndarray, scores2: np.ndarray, n_permutations: int = 10000, random_state: int = 42
) -> float:
    """
    Perform permutation test for difference in means.

    Args:
        scores1: First group scores
        scores2: Second group scores
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        p-value
    """
    rng = np.random.RandomState(random_state)

    observed_diff = np.mean(scores1) - np.mean(scores2)
    combined = np.concatenate([scores1, scores2])
    n1 = len(scores1)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return count / n_permutations


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ModelComparisonResult",
    "VAEComparator",
    "EnsembleComparator",
    "bootstrap_confidence_interval",
    "permutation_test",
]
