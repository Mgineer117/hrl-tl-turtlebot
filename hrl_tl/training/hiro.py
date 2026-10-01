"""HIRO training workflow and configuration reader for multi-environment experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict
from rl_pipeline.sb3 import (
    SB3PipelineConfig,
    SB3PipelineConfigReader,
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.training.replicate import run_replicate_training
from hrl_tl.training.types import ReplayModel


class HiroGoalConfig(BaseModel):
    """Declarative goal specification stored inside training YAML configurations.

    Attributes:
        subgoal_low: Lower coordinate bounds for the subgoal Box space.
        subgoal_high: Upper coordinate bounds for the subgoal Box space.
        goal_key: Observation dictionary key extracted by the state-to-goal projection.
    """

    subgoal_low: list[float]
    subgoal_high: list[float]
    goal_key: str = "agent_pos"

    model_config = ConfigDict(extra="forbid")

    def to_subgoal_space(self) -> spaces.Box:
        """Constructs the Gymnasium Box space for manager subgoals.

        Returns:
            The configured Box space with float32 coordinates.
        """
        return spaces.Box(
            low=np.array(self.subgoal_low, dtype=np.float32),
            high=np.array(self.subgoal_high, dtype=np.float32),
            dtype=np.float32,
        )

    def to_projection_fn(
        self,
    ) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
        """Constructs the state-to-goal projection function.

        Returns:
            A callable extracting the goal array from the observation dictionary.
        """
        key = self.goal_key
        return lambda obs: obs[key]


class HiroPipelineConfigReader(SB3PipelineConfigReader):
    """Pipeline reader that auto-injects HIRO goal space and projection."""

    hiro_goal_config: HiroGoalConfig | None = None

    def to_config(self) -> SB3PipelineConfig:
        """Builds SB3PipelineConfig and populates HIRO algo_kwargs if configured.

        Returns:
            The fully resolved runtime SB3PipelineConfig.
        """
        config = super().to_config()
        if self.hiro_goal_config is not None:
            config.algo_config.algo_kwargs["subgoal_space"] = (
                self.hiro_goal_config.to_subgoal_space()
            )
            config.algo_config.algo_kwargs["state_to_goal_proj_fn"] = (
                self.hiro_goal_config.to_projection_fn()
            )
        return config


def apply_hiro_goal_config(
    reader: SB3ReplicatePipelineConfigReader,
    exp_config: SB3ReplicatePipelineConfig,
) -> None:
    """Injects HIRO goal space and projection into pipeline configs.

    Args:
        reader: The replicate pipeline config reader containing the HIRO
            goal configuration.
        exp_config: The built replicate pipeline config to modify.
    """
    hiro_goal_config = reader.single_pipeline_config.hiro_goal_config
    if hiro_goal_config is None:
        return
    subgoal_space = hiro_goal_config.to_subgoal_space()
    projection_fn = hiro_goal_config.to_projection_fn()
    for ind_config in exp_config.ind_pipeline_configs:
        ind_config.algo_config.algo_kwargs.update(
            {
                "subgoal_space": subgoal_space,
                "state_to_goal_proj_fn": projection_fn,
            }
        )


def run_hiro_training(
    config_path: str | Path,
    retrain_model: bool = False,
    record_replays: bool = True,
    replay_model: ReplayModel = "best",
    device: str | None = None,
    num_replicates: int | None = None,
    replicate_start_id: int | None = None,
    verbose: bool = True,
) -> tuple[SB3ReplicatePipeline, list[BaseAlgorithm]]:
    """Executes replicate training for a HIRO baseline policy.

    Args:
        config_path: Path to the replicate experiment configuration YAML file.
        retrain_model: Whether to retrain an existing model.
        record_replays: Whether to record replays after training.
        replay_model: Which model checkpoint to use for replays ('best',
            'final', or 'latest').
        device: Override GPU device (e.g. '0', '1', 'cuda:0', 'cpu').
        num_replicates: Optional override for number of replicates to train.
        replicate_start_id: Optional override for starting replicate ID.
        verbose: Whether to print verbose progress information.

    Returns:
        A tuple of (pipeline, models), where pipeline is the executed
        SB3ReplicatePipeline and models is the list of best loaded models.

    Raises:
        FileNotFoundError: If the configuration YAML file does not exist.
    """
    return run_replicate_training(
        config_path=config_path,
        config_reader_cls=HiroPipelineConfigReader,
        retrain_model=retrain_model,
        record_replays=record_replays,
        replay_model=replay_model,
        device=device,
        num_replicates=num_replicates,
        replicate_start_id=replicate_start_id,
        verbose=verbose,
        post_config_hook=apply_hiro_goal_config,
    )
