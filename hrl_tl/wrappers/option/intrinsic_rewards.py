from __future__ import annotations

import copy
from typing import Any, Generic, cast

import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import utils as space_utils
from rl_pipeline.sb3 import (
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)
from sb3_hrl import ALLO
from sb3_hrl.option import BaseIntrinsicReward
from typing_extensions import override


def observation_to_flat_array(
    observation: Any,
    observation_space: spaces.Space[Any] | None = None,
) -> np.ndarray:
    """Convert an observation to a 2D numpy array of shape (batch_size, flat_obs_dim).

    Args:
        observation: Environment observation (single or batched array / dict).
        observation_space: Optional Gymnasium observation space. If provided,
            Gymnasium's space_utils.flatten is used to flatten observations,
            properly encoding structured subspaces (e.g. one-hot MultiDiscrete).

    Returns:
        Flattened 2D numpy array with shape (batch_size, flat_obs_dim).

    Raises:
        ValueError: If a dict observation has inconsistent batch sizes or is empty.
    """
    if observation_space is not None:
        if isinstance(observation, dict):
            if isinstance(observation_space, spaces.Dict):
                first_key = next(iter(observation.keys()))
                first_val = np.asarray(observation[first_key])
                first_subspace = observation_space.spaces.get(first_key)
                subspace_shape = (
                    first_subspace.shape if first_subspace is not None else None
                )
                subspace_ndim = (
                    len(subspace_shape) if subspace_shape is not None else 0
                )
                is_batched = first_val.ndim > subspace_ndim
            else:
                first_key = next(iter(observation.keys()))
                first_val = np.asarray(observation[first_key])
                is_batched = False

            if is_batched:
                batch_size = first_val.shape[0]
                flat_list = [
                    space_utils.flatten(
                        observation_space,
                        {k: v[i] for k, v in observation.items()},
                    )
                    for i in range(batch_size)
                ]
                return np.asarray(flat_list, dtype=np.float32)

            flat = space_utils.flatten(observation_space, observation)
            return np.asarray(flat, dtype=np.float32).reshape(1, -1)

        obs_array = np.asarray(observation)
        space_shape = observation_space.shape
        space_ndim = len(space_shape) if space_shape is not None else 0
        if obs_array.ndim > space_ndim:
            batch_size = obs_array.shape[0]
            flat_list = [
                space_utils.flatten(observation_space, obs_array[i])
                for i in range(batch_size)
            ]
            return np.asarray(flat_list, dtype=np.float32)

        flat = space_utils.flatten(observation_space, observation)
        return np.asarray(flat, dtype=np.float32).reshape(1, -1)

    if isinstance(observation, dict):
        # Match SB3 behavior: copy dict observations before conversion.
        observation = copy.deepcopy(observation)

        flat_parts: list[np.ndarray] = []
        batch_size: int | None = None
        for key, obs in observation.items():
            obs_array = np.asarray(obs)
            obs_array = obs_array.reshape((-1, *obs_array.shape))

            if batch_size is None:
                batch_size = obs_array.shape[0]
            elif obs_array.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch size in dict observation for key '{key}': "
                    f"expected {batch_size}, got {obs_array.shape[0]}."
                )

            flat_parts.append(obs_array.reshape(obs_array.shape[0], -1))

        if not flat_parts:
            raise ValueError("Received an empty dict observation.")

        return np.concatenate(flat_parts, axis=1)

    obs_array = np.asarray(observation)
    obs_array = obs_array.reshape((-1, *obs_array.shape))
    return obs_array.reshape(obs_array.shape[0], -1)


def build_observation_batch(
    states: list[Any],
    observation_space: spaces.Space[Any] | None = None,
) -> th.Tensor:
    """Build a stacked 2D PyTorch float tensor from a list of observations.

    Args:
        states: Sequence of observations.
        observation_space: Optional Gymnasium observation space.

    Returns:
        Tensor of shape (batch_size, flat_obs_dim) on CPU with dtype float32.
    """
    flattened_observations: list[np.ndarray] = []

    for obs in states:
        flat_obs_batch = observation_to_flat_array(
            obs, observation_space=observation_space
        )
        for flat_obs in flat_obs_batch:
            flattened_observations.append(
                np.asarray(flat_obs, dtype=np.float32).reshape(-1)
            )

    obs_array = np.stack(flattened_observations, axis=0).astype(
        np.float32, copy=False
    )
    return th.as_tensor(obs_array, dtype=th.float32)


class AlloIntrinsicReward(
    BaseIntrinsicReward[ObsType, ActType], Generic[ObsType, ActType]
):
    """Intrinsic reward calculator based on ALLO feature representations."""

    def __init__(
        self,
        pipeline_config_path: str,
        eig_idx: int,
        reverse_reward: bool,
        rep_idx: int = 0,
        device: str = "cpu",
    ) -> None:
        """Initialize ALLO intrinsic reward.

        Args:
            pipeline_config_path: Path to the pipeline config YAML for ALLO.
            eig_idx: Eigenvector index used to compute intrinsic reward.
            reverse_reward: Whether to invert reward sign.
            rep_idx: Replicate index of the ALLO extractor.
            device: Computing device for model evaluation.
        """
        super().__init__()
        exp_config: SB3ReplicatePipelineConfig = (
            SB3ReplicatePipelineConfigReader.from_yaml(
                pipeline_config_path
            ).to_config()
        )

        self.pipeline = SB3ReplicatePipeline(config=exp_config, verbose=True)
        self.pipeline.ind_pipeline_configs[rep_idx].device = device
        self.allo_model: ALLO = cast(
            ALLO, self.pipeline.load_model(rep_idx, "final")
        )
        self.eig_idx: int = eig_idx
        self.reverse_reward: bool = reverse_reward

    @override
    def intrinsic_reward(
        self,
        obs: ObsType,
        action: ActType,
        next_obs: ObsType,
        external_reward: float,
        done: bool,
    ) -> float:
        """Compute option intrinsic reward as difference in eigenvector features.

        Args:
            obs: Observation before transition.
            action: Action taken.
            next_obs: Observation after transition.
            external_reward: Environment reward.
            done: Termination flag.

        Returns:
            Scalar intrinsic reward.
        """
        del action, external_reward, done
        obs_list = [obs, next_obs]
        obs_batch = build_observation_batch(
            obs_list,
            observation_space=self.allo_model.observation_space,
        )
        obs_batch = obs_batch.to(self.allo_model.device)
        with th.no_grad():
            features: th.Tensor = self.allo_model.feature_net(obs_batch)
        rewards = features[:, self.eig_idx].detach().cpu().numpy()
        intrinsic_reward = float(rewards[1] - rewards[0])
        if self.reverse_reward:
            intrinsic_reward = -intrinsic_reward
        return intrinsic_reward
