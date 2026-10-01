import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Self

import yaml
from gymnasium.core import Env, Wrapper
from pydantic import BaseModel, computed_field
from stable_baselines3.common.vec_env import VecEnv

from hrl_tl.utils.io import format_large_number


class BaseTrainingConfig(BaseModel):
    verbose: bool = False

    def train_env(self) -> Env | Wrapper:
        """Return the environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def eval_env(self) -> Env | Wrapper:
        """Return the evaluation environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def train_vec_env(self) -> VecEnv:
        """Return the vectorized environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def eval_vec_env(self) -> VecEnv:
        """Return the evaluation vectorized environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def model(self, train_env):
        """Return the model name."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def model_learn_config(self) -> dict[str, Any]:
        """Return the model learning configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def load_model(self, env=None):
        """Load the model from the saved path."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Load the configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @property
    def unique_id(self) -> str:
        """Generate a unique identifier for the training run."""
        return str(uuid.uuid4())[:4]

    @property
    def exp_time(self) -> str:
        """Get the current time formatted for the experiment."""
        return datetime.now().strftime("%m-%d_%H-%M-%S.%f")
