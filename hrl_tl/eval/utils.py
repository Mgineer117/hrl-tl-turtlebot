from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel


class EvalResult(BaseModel):
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    std_episode_length: float
    success_rate: float | None = None
    failure_rate: float | None = None
    mean_discounted_return: float | None = None
    std_discounted_return: float | None = None
    episode_rewards: list[float]
    episode_lengths: list[int]
    episode_successes: list[bool] = []
    episode_failures: list[bool] = []


class SuccessBufferEval(BaseModel):
    episode_successes: list[bool] = []
    success_rate: float | None = None
    episode_failures: list[bool] = []
    failure_rate: float | None = None


class SuccessBuffer:
    """
    A buffer to store success/failure rates for evaluation to be used for SB3 `evaluate_policy` function.

    Usage
    -----
    This class is used to track the success and failure rates of episodes during evaluation.
    It can be used to compute the success and failure rates after evaluation is complete.

    Example
    -------
    ```python
    success_buffer = SuccessBuffer()

    episode_rewards, episode_lengths = evaluate_policy(
            model,
            return_episode_rewards=True,
            callback=success_buffer._log_success_callback,
    )
    success_buffer_result: SuccessBufferEval = success_buffer.post_eval()
    ```
    """

    def __init__(self):
        self._is_success_buffer: list[bool] = []
        self._is_failure_buffer: list[bool] = []

    def _log_success_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        Parameters
        ----------
        locals_: dict[str, Any]
            Local variables of the callback function.
        globals_: dict[str, Any]
            Global variables of the callback function.
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success: bool | None = info.get("is_success")
            maybe_is_failure: bool | None = info.get("is_failure")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            if maybe_is_failure is not None:
                self._is_failure_buffer.append(maybe_is_failure)

    def post_eval(self) -> SuccessBufferEval:
        """
        Return the success rate and reset the buffer.

        Returns
        -------
        result: SuccessBufferEval
            The evaluation results containing the following fields:
            -  episode_successes: list[bool]
                -- A list of success rates for each episode.
            -  success_rate: float | None
                -- The mean success rate, or None if no success rates were logged.
            -  episode_failures: list[bool]
                -- A list of failure rates for each episode.
            -  failure_rate: float | None
                -- The mean failure rate, or None if no failure rates were logged.
        """
        episode_successes: list[bool] = []
        episode_failures: list[bool] = []
        success_rate: float | None = None
        failure_rate: float | None = None

        if self._is_success_buffer:
            episode_successes: list[bool] = np.array(
                self._is_success_buffer, dtype=bool
            ).tolist()
            success_rate = float(np.mean(episode_successes))

        if self._is_failure_buffer:
            episode_failures: list[bool] = np.array(
                self._is_failure_buffer, dtype=bool
            ).tolist()
            failure_rate = float(np.mean(episode_failures))

        # Reset the buffer
        self._is_success_buffer.clear()
        self._is_failure_buffer.clear()

        result = SuccessBufferEval(
            episode_successes=episode_successes,
            success_rate=success_rate,
            episode_failures=episode_failures,
            failure_rate=failure_rate,
        )

        return result


class BasicStats(BaseModel):
    """Model for basic statistics."""

    mean: float
    median: float
    std: float
    min: float
    max: float


def compute_basic_stats(data: ArrayLike) -> BasicStats:
    """
    Compute basic statistics for a list of float values.

    Parameters
    ----------
        data: list[float]
            List of float values to compute statistics for

    Returns
    -------
        BasicStats
            Object containing the computed statistics
    """

    return BasicStats(
        mean=float(np.mean(data)),  # type: ignore
        median=float(np.median(data)),  # type: ignore
        std=float(np.std(data)),  # type: ignore
        min=float(np.min(data)),
        max=float(np.max(data)),
    )
