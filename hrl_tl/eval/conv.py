import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class ConvergenceMetrics(BaseModel):
    """Model for convergence metrics."""

    final_mean: float
    final_std: float
    overall_mean: float
    overall_std: float
    trend_slope: float
    trend_r_squared: float
    trend_intercept: float
    recent_cv: float
    variance_ratio: float


class ConvergenceAnalysisResult(BaseModel):
    """Model for complete convergence analysis results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics: ConvergenceMetrics
    convergence_point: int
    has_converged: bool
    total_episodes: int
    returns: NDArray[np.float64]
    final_success_rate: float


def moving_average(
    data: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
    """Calculate moving average with specified window size."""
    if len(data) < window:
        return np.array([], dtype=np.float64)
    result = np.convolve(data, np.ones(window) / window, mode="valid")
    return result.astype(np.float64)


def linear_regression(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float, float]:
    """
    Simple linear regression implementation.
    Returns slope, intercept, and R-squared.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0, float(y_mean), 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return float(slope), float(intercept), float(r_squared)


def calculate_convergence_metrics(
    returns: NDArray[np.float64], window: int = 100
) -> ConvergenceMetrics:
    """
    Calculate various convergence metrics for RL returns.

    Args:
        returns: Array of returns over time
        window: Window size for smoothing and convergence detection

    Returns:
        ConvergenceMetrics containing convergence metrics
    """
    # Basic statistics
    final_mean = (
        np.mean(returns[-window:])
        if len(returns) >= window
        else np.mean(returns)
    )
    final_std = (
        np.std(returns[-window:]) if len(returns) >= window else np.std(returns)
    )
    overall_mean = np.mean(returns)
    overall_std = np.std(returns)

    # Trend analysis using linear regression
    x = np.arange(len(returns), dtype=np.float64)
    slope, intercept, r_squared = linear_regression(x, returns)

    # Convergence detection: coefficient of variation in recent window
    data_for_cv = returns[-window:] if len(returns) >= window else returns
    recent_mean = np.mean(data_for_cv)
    recent_std = np.std(data_for_cv)
    if recent_std == 0:
        recent_cv = 0.0
    elif recent_mean != 0:
        recent_cv = float(recent_std / np.abs(recent_mean))
    else:
        recent_cv = np.inf

    # Stability: variance ratio between first and last halves
    if len(returns) >= 20:
        mid_point = len(returns) // 2
        first_half_var = np.var(returns[:mid_point])
        second_half_var = np.var(returns[mid_point:])
        variance_ratio = (
            second_half_var / first_half_var if first_half_var != 0 else np.inf
        )
    else:
        variance_ratio = 1.0

    return ConvergenceMetrics(
        final_mean=float(final_mean),
        final_std=float(final_std),
        overall_mean=float(overall_mean),
        overall_std=float(overall_std),
        trend_slope=slope,
        trend_r_squared=r_squared,
        trend_intercept=intercept,
        recent_cv=recent_cv,
        variance_ratio=float(variance_ratio),
    )


def detect_convergence_point(
    returns: NDArray[np.float64], threshold: float = 0.1, window: int = 50
) -> tuple[int, bool]:
    """
    Detect the point where the algorithm converged.

    Args:
        returns: Array of returns over time
        threshold: Coefficient of variation threshold for convergence
        window: Window size for convergence detection

    Returns:
        tuple of (convergence_point, has_converged)
    """
    if len(returns) < window:
        return len(returns), False

    overall_mean = np.mean(returns)

    for i in range(window, len(returns)):
        window_data = returns[i - window : i]
        window_mean = np.mean(window_data)
        window_std = np.std(window_data)

        # Special case: if both overall mean and window mean are 0, consider converged
        if overall_mean == 0 and window_mean == 0:
            return i - window // 2, True

        if window_std == 0:
            return i - window // 2, True

        cv = window_std / np.abs(window_mean) if window_mean != 0 else np.inf

        if cv < threshold:
            return i - window // 2, True

    return len(returns), False


def plot_convergence_analysis(
    returns: NDArray[np.float64],
    convergence_point: int,
    has_converged: bool,
    save_path: str | None = None,
) -> None:
    """
    Create comprehensive convergence plots.

    Args:
        returns: Array of returns over time
        convergence_point: Detected convergence point
        has_converged: Whether convergence was detected
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RL Reward Convergence Analysis", fontsize=16)

    # Raw returns plot
    axes[0, 0].plot(returns, alpha=0.7, color="blue", label="Raw Returns")
    if len(returns) > 20:
        smoothed = moving_average(returns, min(20, len(returns) // 10))
        axes[0, 0].plot(
            range(len(smoothed)),
            smoothed,
            color="red",
            linewidth=2,
            label="Smoothed",
        )

    if has_converged:
        axes[0, 0].axvline(
            x=convergence_point,
            color="green",
            linestyle="--",
            label=f"Convergence Point: {convergence_point}",
        )

    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].set_title("Returns Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Moving average with different window sizes
    windows = [10, 50, 100] if len(returns) > 100 else [5, 10, 20]
    for window in windows:
        if len(returns) > window:
            ma = moving_average(returns, window)
            axes[0, 1].plot(
                range(window - 1, len(returns)), ma, label=f"MA-{window}"
            )

    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Moving Average Return")
    axes[0, 1].set_title("Moving Averages")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Rolling coefficient of variation
    window = min(50, len(returns) // 5) if len(returns) > 10 else 5
    rolling_cv = []
    for i in range(window, len(returns)):
        window_data = returns[i - window : i]
        cv = (
            np.std(window_data) / np.abs(np.mean(window_data))
            if np.mean(window_data) != 0
            else 0
        )
        rolling_cv.append(cv)

    if rolling_cv:
        axes[1, 0].plot(range(window, len(returns)), rolling_cv, color="purple")
        axes[1, 0].axhline(
            y=0.1,
            color="red",
            linestyle="--",
            label="Convergence Threshold (0.1)",
        )
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Coefficient of Variation")
        axes[1, 0].set_title("Rolling Coefficient of Variation")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Distribution of returns
    axes[1, 1].hist(
        returns, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[1, 1].axvline(
        x=np.mean(returns),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(returns):.2f}",
    )
    axes[1, 1].axvline(
        x=np.median(returns),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(returns):.2f}",
    )
    axes[1, 1].set_xlabel("Return Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Returns")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def analyze_convergence(
    file_path: str,
    convergence_threshold: float = 0.1,
    convergence_window: int = 50,
    plot: bool = True,
    save_plot: bool = False,
) -> ConvergenceAnalysisResult:
    """
    Comprehensive convergence analysis of RL returns.

    Args:
        file_path: Path to npz file containing returns
        convergence_threshold: CV threshold for convergence detection
        convergence_window: Window size for convergence detection
        plot: Whether to create plots
        save_plot: Whether to save plots

    Returns:
        ConvergenceAnalysisResult containing all analysis results
    """
    # Load data
    data = np.load(file_path, allow_pickle=True)
    print("Loaded data keys:", data.files)

    if "results" in data.files:
        returns = data["results"]
    elif "returns" in data.files:
        returns = data["returns"]
    else:
        raise KeyError(
            f"Expected 'results' or 'returns' in {file_path}, but found: {data.files}"
        )
    print(f"Returns shape: {returns.shape}")

    # Handle different data structures
    if returns.ndim == 2:
        # If returns is 2D (evaluation_steps, episodes), take mean across episodes
        mean_returns = np.mean(returns, axis=0)
        print("Taking mean across episodes for analysis")
    else:
        mean_returns = returns

    # Handle successes data if available
    if "successes" in data.files:
        successes: NDArray[np.bool_] = data["successes"]
        if successes.ndim == 2:
            final_success_rate = np.mean(successes[-1, :])
        else:
            final_success_rate = np.mean(successes)
    else:
        print(
            "Warning: No 'successes' key found in data. Setting final_success_rate to 0.0"
        )
        final_success_rate = 0.0

    print(f"Analyzing {len(mean_returns)} data points")

    # Calculate metrics
    metrics = calculate_convergence_metrics(mean_returns, convergence_window)

    # Detect convergence
    convergence_point, has_converged = detect_convergence_point(
        mean_returns, convergence_threshold, convergence_window
    )

    # Compile results
    results = ConvergenceAnalysisResult(
        metrics=metrics,
        convergence_point=convergence_point,
        has_converged=has_converged,
        total_episodes=len(mean_returns),
        returns=mean_returns,
        final_success_rate=float(final_success_rate),
    )

    # Print summary
    print("\n" + "=" * 50)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total Episodes: {len(mean_returns)}")
    print(
        f"Final Mean Return: {metrics.final_mean:.3f} ± {metrics.final_std:.3f}"
    )
    print(
        f"Overall Mean Return: {metrics.overall_mean:.3f} ± {metrics.overall_std:.3f}"
    )
    print(
        f"Trend Slope: {metrics.trend_slope:.6f} (R²: {metrics.trend_r_squared:.3f})"
    )
    print(f"Recent Coefficient of Variation: {metrics.recent_cv:.3f}")
    print(f"Variance Ratio (late/early): {metrics.variance_ratio:.3f}")

    if has_converged:
        print(f"✅ CONVERGED at episode {convergence_point}")
        print(
            f"   Convergence achieved in {convergence_point / len(mean_returns) * 100:.1f}% of training"
        )
    else:
        print("❌ NOT CONVERGED")
        print("   Consider training longer or adjusting convergence criteria")

    print("=" * 50)

    # Create plots
    if plot:
        save_path = None
        if save_plot:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = f"{base_name}_convergence_analysis.png"

        plot_convergence_analysis(
            mean_returns, convergence_point, has_converged, save_path
        )

    return results
