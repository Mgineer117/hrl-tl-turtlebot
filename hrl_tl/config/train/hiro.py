import os
import random
from typing import Any, Self

import gymnasium as gym
import yaml
from gymnasium.core import Env, Wrapper
from pydantic import computed_field
from torch.utils.tensorboard import SummaryWriter

from hrl_tl.baseline.algorithms.HIRO import HIRO
from hrl_tl.baseline.log.wandb_logger import WandbLogger
from hrl_tl.config.algo import HiroConfig
from hrl_tl.config.env import EnvMakeConfig
from hrl_tl.utils.io import format_large_number, get_class

from .base import BaseTrainingConfig


class HiroTrainingConfig(BaseTrainingConfig):
    """
    Configuration class for training the HIRO model.
    Inherits from BaseTrainingConfig and adds specific parameters for HIRO.
    """

    gpu_id: int = 0
    experiment_id: str = "4.a"
    models_dir: str = "out/fourroom/hiro"
    replicate_dir: str = ""
    model_name: str = "hiro_fourroom"
    model_filename: str = "model.pt"
    monitor_dir: str = "log"
    tb_dir: str = "tb"
    eval_dir: str = "eval"
    config_dir: str = "configs/fourroom"
    env_config_file: str = "env/full_random.yaml"
    algo_config_file: str = "baseline/hiro.yaml"
    num_runs: int = 1
    rendering: bool = False
    retrain_model: bool = False
    verbose: bool = True

    @property
    def model_full_name(self) -> str:
        total_timesteps: int = self.algo_config.timesteps
        return self.model_name + f"_{format_large_number(total_timesteps)}"

    @property
    def model_save_dir(self) -> str:
        return os.path.join(
            self.models_dir,
            self.experiment_id,
            self.model_full_name,
            self.replicate_dir,
        )

    @property
    def _model_filename(self) -> str:
        return self.model_filename.replace(
            ".pt",
            f"_{self.experiment_id}"
            + f"_{format_large_number(self.algo_config.timesteps)}.pt",
        )

    @computed_field
    @property
    def model_save_path(self) -> str:
        return os.path.join(self.model_save_dir, self._model_filename)

    @property
    def eval_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, self.eval_dir)

    @property
    def animation_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, "animation")

    @computed_field
    @property
    def animation_save_path(self) -> str:
        return os.path.join(
            self.animation_save_dir, self._model_filename
        ).replace(".pt", ".gif")

    @property
    def monitor_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, self.monitor_dir)

    @property
    def tb_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, self.tb_dir)

    @property
    def device(self) -> str:
        """Returns the device to be used for training."""
        return f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu"

    @property
    def env_config_dict(self) -> dict[str, Any]:
        with open(
            os.path.join(self.config_dir, self.env_config_file), "r"
        ) as f:
            return yaml.safe_load(f)

    @computed_field
    @property
    def env_config(self) -> EnvMakeConfig:
        return EnvMakeConfig(**self.env_config_dict)

    @property
    def algo_config_dict(self) -> dict[str, Any]:
        with open(
            os.path.join(self.config_dir, self.algo_config_file), "r"
        ) as f:
            return yaml.safe_load(f)

    @computed_field
    @property
    def algo_config(self) -> HiroConfig:
        return HiroConfig(**self.algo_config_dict)

    @computed_field
    @property
    def seed(self) -> int:
        """Returns the seed for reproducibility."""
        return random.randint(1, 100_000)

    def train_env(self) -> Env:
        return gym.make(**self.env_config.model_dump(context={"flatten": True}))

    def eval_env(self) -> Env:
        return self.train_env()

    def wandb_config(self) -> dict[str, Any]:
        """
        Returns the configuration for WandB logging.
        This includes the project name, group, and other relevant parameters.
        """
        run_name: str = "-".join(
            (
                self.algo_config.algo_name,
                self.env_config.id,
                self.unique_id,
                "seed:" + str(self.seed),
            )
        )
        return {
            "project": self.experiment_id,
            "group": "-".join((self.exp_time, self.unique_id)),
            "name": run_name,
            "config": self.model_dump(),
            "log_dir": self.monitor_save_dir,
            "log_txt": True,
            "fps": self.algo_config.render_fps,
        }

    def model(self, train_env: Env | Wrapper) -> HIRO:
        logger = WandbLogger(**self.wandb_config())

        os.makedirs(self.tb_save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.tb_save_dir)
        return HIRO(
            env=train_env,
            logger=logger,
            writer=writer,
            seed=self.seed,
            algo_config=self.algo_config,
            device=self.device,
        )

    def load_model(self, env: Env | Wrapper | None = None) -> HIRO:
        """Load the HIRO model from the saved path.

        Parameters
        ----------
        env : Env | Wrapper, optional
            The environment to use for loading the model. Defaults to None.

        Returns
        -------
        HIRO
            The loaded HIRO model.
        """
        if env is None:
            env = self.train_env()
        model = HIRO(
            env=env,
            seed=self.seed,
            algo_config=self.algo_config,
            device=self.device,
        )
        model.load(self.model_save_path)

        if self.verbose:
            print(f"Loaded HIRO model from {self.model_save_path}")

        return model
