import os
from typing import Any, Literal, TypeVar

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from hrl_tl.config.train import SB3BaseTrainingConfig
from hrl_tl.eval import EvalResult, SuccessBuffer, SuccessBufferEval
from hrl_tl.utils.io import add_number_to_existing_filepath
from hrl_tl.utils.sb3.replay import record_replay

from .base import BasePipeline

SB3ConfigType = TypeVar("SB3ConfigType", bound=SB3BaseTrainingConfig)


class SB3Pipeline(BasePipeline[SB3ConfigType]):
    """
    Pipeline for training with Stable Baselines3.
    This class extends the BasePipeline to include SB3-specific configurations and methods.
    """

    def __init__(self, config: SB3ConfigType, verbose: bool = True):
        self.config: SB3ConfigType = config
        self.verbose: bool = verbose

    def train(self) -> BaseAlgorithm:
        """
        Train the model using the provided training configuration.
        """
        train_env = self.config.train_vec_env()
        model = self.config.model(train_env)
        model.learn(**self.config.model_learn_config)
        train_env.close()
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        # Save the model
        # if there is already an existing model, add a number suffix e.g. "_1"
        save_path: str = self.config.model_save_path
        modified_save_path = add_number_to_existing_filepath(save_path)
        model.save(modified_save_path)

        return model

    def train_on_unsaved_model(self) -> BaseAlgorithm:
        """
        Train the model on an unsaved model.
        This method is a placeholder and should be implemented if needed.
        """
        demo_env = self.config.eval_env()
        if (
            not os.path.exists(self.config.model_save_path)
            or self.config.retrain_model
        ):
            model = self.train()
        else:
            print(
                f"SB3Pipeline: Model {self.config.model_save_path} already exists, loading..."
            )
            model = self.config.load_model(demo_env)

        return model

    def model_exists(self) -> bool:
        """
        Check if the model exists.
        """
        return os.path.exists(self.config.model_save_path)

    def load_model(
        self,
        checkpoint: int | Literal["latest", "final", "best"] = "final",
    ) -> BaseAlgorithm:
        match checkpoint:
            case "final":
                model = self.config.load_model()
            case "best":
                model = self.config.load_model(load_best=True)
            case "latest":
                model = self.config.load_checkpoint("latest")
            case _:
                model = self.config.load_checkpoint(checkpoint)

        return model

    def evaluate(
        self,
        n_eval_episodes: int = 100,
        deterministic: bool = False,
        save_to_file: bool = True,
        eval_file_name: str = "model_eval.yaml",
        checkpoint: int
        | Literal["latest", "final", "best"]
        | BaseAlgorithm = "final",
    ) -> EvalResult:
        """Evaluate the final model."""

        if self.verbose:
            print(f"SB3Pipeline: Evaluating the {checkpoint} model...")

        if isinstance(checkpoint, BaseAlgorithm):
            model: BaseAlgorithm = checkpoint
        else:
            model: BaseAlgorithm = self.load_model(checkpoint)

        success_buffer = SuccessBuffer()
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            self.config.eval_vec_env(),
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
            callback=success_buffer._log_success_callback,
        )
        assert isinstance(episode_rewards, list)
        assert isinstance(episode_lengths, list)

        # Only save four decimal places for readability
        decimal_places: int = 4
        mean_reward: float = float(
            np.mean(episode_rewards).round(decimal_places)
        )
        std_reward: float = float(np.std(episode_rewards).round(decimal_places))
        mean_episode_length: float = float(
            np.mean(episode_lengths).round(decimal_places)
        )
        std_episode_length: float = float(
            np.std(episode_lengths).round(decimal_places)
        )
        success_buffer_result: SuccessBufferEval = success_buffer.post_eval()

        eval_result = EvalResult(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            **success_buffer_result.model_dump(),
        )

        if self.verbose:
            self._print_eval_result(eval_result)

        if save_to_file:
            eval_prefix = f"{checkpoint}_"
            eval_file_path = os.path.join(
                self.config.eval_save_dir, eval_prefix + eval_file_name
            )
            modified_eval_file_path = add_number_to_existing_filepath(
                eval_file_path
            )
            self._save_eval_result(eval_result, modified_eval_file_path)

        return eval_result

    def record_replay(
        self,
        model: BaseAlgorithm,
        save_path: str | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Record a replay of the model's performance in the evaluation environment.
        """
        if save_path is None:
            save_path = self.config.animation_save_path

        record_replay(
            self.config.eval_env(), model, save_path, self.verbose or verbose
        )
