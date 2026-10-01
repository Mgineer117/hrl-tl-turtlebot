import os
from typing import Any, Self, TypeVar

import yaml

import wandb
from hrl_tl.baseline.algorithms.HIRO import HIRO
from hrl_tl.baseline.utils.utils import seed_all
from hrl_tl.config.train import HiroTrainingConfig
from hrl_tl.eval import EvalResult
from hrl_tl.utils.baseline import record_replay
from hrl_tl.utils.io import get_class

from .base import BasePipeline

HiroConfigType = TypeVar("HiroConfigType", bound=HiroTrainingConfig)


class HiroPipeline(BasePipeline[HiroConfigType]):
    """
    Pipeline for training the HIRO model.
    Inherits from BasePipeline and uses HiroTrainingConfig.
    """

    def __init__(self, config: HiroConfigType, verbose: bool = True):
        super().__init__(config, verbose)

    def train(self) -> HIRO:
        """
        Train the HIRO model using the provided configuration.
        """

        seed_all(self.config.seed)
        env = self.config.train_env()
        model: HIRO = self.config.model(env)
        model.learn(
            total_timesteps=self.config.algo_config.timesteps,
            eval_freq=self.config.algo_config.eval_freq,
        )

        wandb.finish()
        if model.writer:
            model.writer.close()

        model.save(self.config.model_save_path)

        return model

    def train_on_unsaved_model(self) -> HIRO:
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
                f"HiroPipeline: Model {self.config.model_save_path} already exists, loading..."
            )
            model = self.config.load_model(demo_env)

        return model

    def evaluate(
        self,
        n_eval_episodes: int | None = None,
        save_to_file: bool = True,
        eval_file_name: str = "final_model_eval.yaml",
    ) -> EvalResult:
        """Evaluate the final model."""

        if self.verbose:
            print(
                f"HiroPipeline: Evaluating model {self.config.model_save_path}..."
            )

        model: HIRO = self.config.load_model(self.config.eval_env())

        if n_eval_episodes is None:
            n_eval_episodes = self.config.algo_config.eval_episodes

        eval_result: EvalResult = model.evaluate(
            n_eval_episodes=n_eval_episodes
        )

        if self.verbose:
            self._print_eval_result(eval_result)

        if save_to_file:
            eval_file_path = os.path.join(
                self.config.model_save_dir, eval_file_name
            )
            self._save_eval_result(eval_result, eval_file_path)

        return eval_result

    def record_replay(
        self, model: HIRO, save_path: str | None = None, verbose: bool = True
    ) -> None:
        """
        Record a replay of the model's performance in the evaluation environment.
        """
        if save_path is None:
            save_path = self.config.animation_save_path

        record_replay(
            self.config.eval_env(), model, save_path, self.verbose or verbose
        )

    @classmethod
    def from_yaml(cls, path: str, replicate_dir: str = "") -> Self:
        """
        Create a pipeline instance from a YAML training configuration file.
        """
        with open(path, "r") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        data.update({"replicate_dir": replicate_dir})
        config_class_name: str = data["config_class"]
        config_class: type[HiroConfigType] | None = get_class(config_class_name)
        assert config_class is not None, (
            f"Config class {config_class_name} not found."
        )
        config: HiroConfigType = config_class(**data)
        return cls(config)
