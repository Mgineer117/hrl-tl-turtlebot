import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from hrl_tl.utils.sb3.replay import record_replay

from .utils import SuccessBuffer, SuccessBufferEval


class SuccessEvalCallback(EvalCallback):
    """
    Extension of EvalCallback to include success/failure metrics.

    Parameters
    ----------
    eval_env : gym.Env or VecEnv
        The environment used for initialization.
    callback_on_new_best : BaseCallback, optional
        Callback to trigger when there is a new best model according to the
        mean_reward. Default is None.
    callback_after_eval : BaseCallback, optional
        Callback to trigger after every evaluation. Default is None.
    n_eval_episodes : int, default=5
        The number of episodes to test the agent.
    eval_freq : int, default=10000
        Evaluate the agent every eval_freq call of the callback.
    log_path : str, optional
        Path to a folder where the evaluations (evaluations.npz) will be saved.
        It will be updated at each evaluation. Default is None.
    best_model_save_path : str, optional
        Path to a folder where the best model according to performance on the
        eval env will be saved. Default is None.
    deterministic : bool, default=True
        Whether the evaluation should use deterministic or stochastic actions.
    render : bool, default=False
        Whether to render or not the environment during evaluation.
    verbose : int, default=1
        Verbosity level: 0 for no output, 1 for indicating information about
        evaluation results.
    warn : bool, default=True
        Passed to evaluate_policy (warns if eval_env has not been wrapped with
        a Monitor wrapper).

    Warnings
    --------
    When using multiple environments, each call to env.step() will effectively
    correspond to n_envs steps. To account for that, you can use
    eval_freq = max(eval_freq // n_envs, 1).
    """

    def __init__(
        self,
        eval_env: gym.Env | VecEnv,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.evaluations_failures: list[list[bool]] = []

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self.success_buffer: SuccessBuffer = SuccessBuffer()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self.success_buffer._log_success_callback,
            )

            success_buffer_result: SuccessBufferEval = (
                self.success_buffer.post_eval()
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(success_buffer_result.episode_successes) > 0:
                    self.evaluations_successes.append(
                        success_buffer_result.episode_successes
                    )
                    kwargs = dict(successes=self.evaluations_successes)

                # Save failures log if present
                if len(success_buffer_result.episode_failures) > 0:
                    self.evaluations_failures.append(
                        success_buffer_result.episode_failures
                    )
                    kwargs = dict(failures=self.evaluations_failures)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = (
                np.mean(episode_rewards),
                np.std(episode_rewards),
            )
            mean_ep_length, std_ep_length = (
                np.mean(episode_lengths),
                np.std(episode_lengths),
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
                )
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if success_buffer_result.success_rate:
                success_rate: float = success_buffer_result.success_rate
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if success_buffer_result.failure_rate:
                failure_rate: float = success_buffer_result.failure_rate
                if self.verbose >= 1:
                    print(f"Failure rate: {100 * failure_rate:.2f}%")
                self.logger.record("eval/failure_rate", failure_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps",
                self.num_timesteps,
                exclude="tensorboard",
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
        save_dir: str,
        name_prefix: str = "rl_model",
        file_ext: str = "gif",
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._deterministic = deterministic
        self._save_dir = save_dir
        self._name_prefix = name_prefix
        self._file_ext = file_ext

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            animation_save_path: str = os.path.join(
                self._save_dir,
                f"{self._name_prefix}_{self.num_timesteps}_steps.{self._file_ext}",
            )
            record_replay(
                self._eval_env, self.model, animation_save_path, False
            )

        return True
