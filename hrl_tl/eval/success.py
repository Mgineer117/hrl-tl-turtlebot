import json
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

from hrl_tl.config import SB3LowLevelTrainingConfig
from hrl_tl.eval.utils import BasicStats, EvalResult, compute_basic_stats
from hrl_tl.utils.io import get_file_with_largest_number


class SuccessRateAnalysis(BaseModel):
    """Model for success rate analysis results."""

    total_specs: int
    analyzed_specs: int
    missing_specs: int
    success_rates: list[float]
    failure_rates: list[float]
    success_rate_stats: BasicStats
    failure_rate_stats: BasicStats


class SpecPerformanceMetrics(BaseModel):
    """Model for specification performance metrics."""

    specification: str
    success_rate: float | None
    failure_rate: float | None
    mean_reward: float
    mean_episode_length: float


def generate_all_training_configs(
    spec_file_path: str, training_config_path: str
) -> list[SB3LowLevelTrainingConfig]:
    """
    Generate all training configurations based on the specifications.

    Parameters
    -----------
        spec_file_path: str
            Path to the specifications YAML file
        training_config_path: str
            Path to the training configuration YAML file

    Returns
    -------
        configs: list[SB3LowLevelTrainingConfig]
            list of SB3LowLevelTrainingConfig objects
    """
    with open(spec_file_path, "r") as f:
        spec_data: dict[str, Any] = json.load(f)

    tl_specs: list[str] = spec_data["specifications"]

    with open(training_config_path, "r") as f:
        training_config_dict: dict[str, Any] = yaml.safe_load(f)

    configs: list[SB3LowLevelTrainingConfig] = []

    for spec in tl_specs:
        config = SB3LowLevelTrainingConfig(**training_config_dict)
        config.tl_spec = spec
        configs.append(config)

    return configs


def extract_success_rates(
    training_configs: list[SB3LowLevelTrainingConfig],
) -> tuple[SuccessRateAnalysis, list[SpecPerformanceMetrics]]:
    """
    Extract success rates from convergence data and compute statistics.

    Parameters
    -----------
    convergence_data: Dictionary containing convergence analysis results

    Returns
    -------
    SuccessRateAnalysis object containing extracted data and statistics
    """

    # Extract success rates for all analyzed specifications
    success_rates = []
    failure_rates = []

    missing_spec_count: int = 0

    detailed_data: list[SpecPerformanceMetrics] = []

    for config in training_configs:
        try:
            best_model_eval_path: str = config.eval_metrics_save_path.replace(
                "final", "best"
            )
            if os.path.exists(best_model_eval_path):
                used_path: str = best_model_eval_path
            else:
                used_path: str = config.eval_metrics_save_path

            filepath: str = get_file_with_largest_number(used_path)
            with open(filepath, "r") as f:
                eval_data: dict[str, Any] = yaml.safe_load(f)

            eval_result = EvalResult(**eval_data)

            success_rates.append(eval_result.success_rate)
            failure_rates.append(eval_result.failure_rate)

            detailed_data.append(
                SpecPerformanceMetrics(
                    specification=config.tl_spec,
                    success_rate=eval_result.success_rate,
                    failure_rate=eval_result.failure_rate,
                    mean_reward=eval_result.mean_reward,
                    mean_episode_length=eval_result.mean_episode_length,
                )
            )

        except Exception as e:
            print(f"Error loading {config.eval_metrics_save_path}: {e}")
            missing_spec_count += 1
            continue

    # Compute statistics
    success_rates_array = np.array(success_rates)
    failure_rates_array = np.array(failure_rates)

    success_rate_stats = compute_basic_stats(success_rates_array)
    failure_rate_stats = compute_basic_stats(failure_rates_array)

    analysis = SuccessRateAnalysis(
        total_specs=len(training_configs),
        analyzed_specs=len(success_rates),
        missing_specs=missing_spec_count,
        success_rates=success_rates,
        failure_rates=failure_rates,
        success_rate_stats=success_rate_stats,
        failure_rate_stats=failure_rate_stats,
    )

    return analysis, detailed_data


def add_dist_to_axes(
    ax: Axes,
    series: list[float],
    stats: BasicStats,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """
    Add distribution histogram with statistics to given axes.

    Parameters
    -----------
        ax: Axes
            Matplotlib axes to plot on
        series: list[float]
            Data series to plot
        stats: BasicStats
            Statistics for the series
        title: str
            Title for the plot
        xlabel: str
            Label for x-axis
        color: str
            Color for the histogram bars
    """
    # Distribution histogram
    n, bins, patches = ax.hist(
        series, bins=20, alpha=0.7, color=color, edgecolor="black"
    )

    # Add count labels on top of each bin
    max_count = float(np.max(n))  # Convert to float for type safety
    for i, count in enumerate(n):
        if count > 0:  # Only show labels for non-empty bins
            bin_center = (bins[i] + bins[i + 1]) / 2
            ax.text(
                bin_center,
                count + max_count * 0.01,  # Slightly above the bar
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=12,
            )

    # Add mean and median lines
    ax.axvline(
        stats.mean,
        color="red",
        linestyle="--",
        label=f"Mean: {stats.mean:.3f}",
    )
    ax.axvline(
        stats.median,
        color="orange",
        linestyle="--",
        label=f"Median: {stats.median:.3f}",
    )

    # Set labels and styling
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Number of Specifications", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)


def create_success_rate_plots(
    analysis: SuccessRateAnalysis, save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive plots for success rate and failure rate distribution analysis.

    Parameters
    -----------
        analysis: SuccessRateAnalysis object containing the data
        save_path: Optional path to save the plot
    """
    fig: Figure
    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Add success rate distribution
    add_dist_to_axes(
        ax1,
        analysis.success_rates,
        analysis.success_rate_stats,
        "Success Rate Distribution",
        "Final Success Rate",
        "skyblue",
    )

    # Add failure rate distribution
    add_dist_to_axes(
        ax2,
        analysis.failure_rates,
        analysis.failure_rate_stats,
        "Failure Rate Distribution",
        "Final Failure Rate",
        "lightcoral",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(
            f"Success and failure rate distribution plots saved to: {save_path}"
        )

    plt.show()


def print_basic_stats(category: str, stats: BasicStats) -> None:
    """
    Print basic statistics for a given specification.

    Parameters
    ----------
        name: str
            Name of the specification
        stats: str
            BasicStats object containing the statistics
    """
    # Capitalize the first letter of the category and the rest are in the lower case
    capitalized_category = category.capitalize()
    print(f"Basic {capitalized_category} Statistics:")
    print(f"  Mean {capitalized_category} Rate: {stats.mean:.4f}")
    print(f"  Median {capitalized_category} Rate: {stats.median:.4f}")
    print(f"  Standard Deviation: {stats.std:.4f}")
    print(f"  Min {capitalized_category} Rate: {stats.min:.4f}")
    print(f"  Max {capitalized_category} Rate: {stats.max:.4f}")
    print()


def print_analysis_summary(analysis: SuccessRateAnalysis) -> None:
    """
    Print a detailed summary of the success rate analysis.

    Parameters
    -----------
        analysis: SuccessRateAnalysis object containing the data
    """
    print("=" * 60)
    print("SUCCESS RATE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"Total specifications: {analysis.total_specs}")
    print(f"Analyzed specifications: {analysis.analyzed_specs}")
    print(f"Missing specifications: {analysis.missing_specs}")
    print()

    print_basic_stats("success", analysis.success_rate_stats)
    print_basic_stats("failure", analysis.failure_rate_stats)

    # Percentiles
    if len(analysis.success_rates) > 0:
        percentiles = [10, 25, 50, 75, 90]
        print("Success Rate Percentiles:")
        for p in percentiles:
            value = np.percentile(analysis.success_rates, p)
            print(f"  {p}th percentile: {value:.4f}")
        print()

    # Count by success rate ranges
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print("Specifications by Success Rate Range:")
    for low, high in ranges:
        if low == 0.8:  # Last range includes 1.0
            count = sum(
                1 for rate in analysis.success_rates if low <= rate <= high
            )
        else:
            count = sum(
                1 for rate in analysis.success_rates if low <= rate < high
            )
        percentage = (
            (count / analysis.analyzed_specs * 100)
            if analysis.analyzed_specs > 0
            else 0
        )
        print(f"  {low:.1f}-{high:.1f}: {count} specs ({percentage:.1f}%)")


def save_detailed_results(
    spec_performances: list[SpecPerformanceMetrics], output_file: str
) -> None:
    """
    Save detailed results to a CSV file.

    Parameters
    -----------
    spec_performances: list[SpecPerformanceMetrics]
        List of SpecPerformanceMetrics objects containing detailed performance data
    output_file: str
        Path to the output CSV file
    """
    # Convert Pydantic models to dictionary data
    data = []
    for spec_perf in spec_performances:
        data.append(
            {
                "Specification": spec_perf.specification,
                "Success_Rate": spec_perf.success_rate,
                "Failure_Rate": spec_perf.failure_rate,
                "Mean_Reward": spec_perf.mean_reward,
                "Mean_Episode_Length": spec_perf.mean_episode_length,
            }
        )

    # Create Polars DataFrame
    df = polars.DataFrame(data)

    # Sort by success rate in descending order
    df = df.sort("Success_Rate", descending=True)

    # Save to CSV
    df.write_csv(output_file)
    print(f"Detailed results saved to: {output_file}")
