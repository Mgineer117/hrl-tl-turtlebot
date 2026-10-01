from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from hrl_tl.utils.io import get_class


class SB3AlgorithmConfig(BaseModel):
    """
    Configuration for the Stable Baselines3 algorithm.
    """

    algorithm_name: str = Field(
        default="PPO",
        description="Name of the Stable Baselines3 algorithm to use.",
    )
    algo_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters for the algorithm.",
    )

    @property
    def algo_class(self) -> type[BaseAlgorithm]:
        """
        Returns the class of the algorithm based on the algorithm name.
        """
        module_class_name = (
            self.algorithm_name
            if "." in self.algorithm_name
            else f"stable_baselines3.{self.algorithm_name}"
        )

        alg_class = get_class(module_class_name)
        if alg_class is None:
            raise ValueError(
                f"Algorithm class '{module_class_name}' not found."
            )
        return alg_class


class SB3LearnConfig(BaseModel):
    total_timesteps: int
    progress_bar: bool = True
    callback: list[BaseCallback] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HiroConfig(BaseModel):
    algo_name: str = "HIRO"
    timesteps: int = 3_000_000
    gamma: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    eval_freq: int = 100
    K_epochs: int = 5
    target_kl: float = 0.01
    actor_fc_dims: list[int] = Field(
        default_factory=lambda: [256, 256],
        description="Dimensions of the fully connected layers in the actor network.",
    )
    critic_fc_dims: list[int] = Field(
        default_factory=lambda: [512, 512],
        description="Dimensions of the fully connected layers in the critic network.",
    )
    num_mini_batches: int = 8
    mini_batch_size: int = 512
    batch_size: int = 4096
    render_fps: int = 10
    eval_episodes: int = 100
