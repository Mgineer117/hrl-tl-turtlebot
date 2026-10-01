import copy
from typing import Any, Generic

import numpy as np
import torch
import yaml
from gym_tl_tools import TLObservationReward
from gymnasium.core import Env, ObsType
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from rl_pipeline.core import ConfigReader
from rl_pipeline.core.utils.io import get_class
from rl_pipeline.sb3 import SB3Pipeline, SB3PipelineConfigReader
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor
from torch.distributions import Categorical

from hrl_tl.config.wrapper import TLSB3PipelineConfigReader

from .base import LowLevelPolicy, LowLevelPolicyBuffer, PolicyType, TLObs


class CompositePolicy(Generic[ObsType, PolicyType]):
    def __init__(
        self,
        tl_spec: str,
        primitives: list[str],
        primitive_model_map: dict[str, PolicyType],
        verbose: bool = False,
    ) -> None:
        self.tl_spec: str = tl_spec
        self.primitives: list[str] = primitives
        self.primitive_model_map: dict[str, PolicyType] = primitive_model_map
        self.verbose: bool = verbose

        # Get used primitives in the tl_spec
        self.used_primitives: list[str] = [
            p for p in self.primitives if p in self.tl_spec
        ]
        assert self.used_primitives, "No used primitives found"
        self.used_models: list[PolicyType] = [
            self.primitive_model_map[p] for p in self.used_primitives
        ]
        assert self.used_models, "No used models found"

        self.used_primitives_filtered: list[str] = self.used_primitives
        self.used_models_filtered: list[PolicyType] = self.used_models

    def filter_terminal_predicates(
        self, obs: ObsType, current_env: Env, tl_wrapper_args: dict[str, Any]
    ) -> None:
        """
        Filter out the primitives whose terminal predicates are already satisfied in the current observation
        by forwarding the the automaton state of the automaton in the TL wrapper constructed from each of the
        primitive specs and the current environment.

        obs: ObsType
            The current observation from the environment.
        current_env: Env
            The current environment.
        tl_wrapper_args: dict[str, Any]
            The arguments used to construct the TL wrapper.
        """

        # Remove "tl_spec" from tl_wrapper_args if it exists
        tl_wrapper_args.pop("tl_spec", None)

        used_primitives_filtered: list[str] = []
        used_models_filtered: list[PolicyType] = []

        for p, m in zip(self.used_primitives, self.used_models):
            # Construct the TL wrapper for the primitive spec
            tl_env = TLObservationReward[ObsType, NDArray](
                copy.deepcopy(current_env), tl_spec=p, **tl_wrapper_args
            )
            tl_env.automaton.reset()
            tl_env.forward_aut(obs, {})

            if tl_env.is_aut_terminated:
                if self.verbose:
                    print(f"Primitive {p} is filtered out")
            else:
                used_primitives_filtered.append(p)
                used_models_filtered.append(m)

        self.used_primitives_filtered = used_primitives_filtered
        self.used_models_filtered = used_models_filtered

    def predict(
        self,
        observation: ObsType,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        raise NotImplementedError


class DQNSkillMachinePolicy(
    CompositePolicy[np.ndarray | dict[str, np.ndarray], DQN]
):
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        with torch.no_grad():
            q_values_lists: list[Tensor] = []
            if isinstance(observation["aut_state"], int):
                observation["aut_state"] = 1
            else:
                match observation["aut_state"].ndim:
                    case 1:
                        observation["aut_state"][:] = 1
                    case 0:
                        observation["aut_state"] = np.array(
                            1, dtype=observation["aut_state"].dtype
                        )
            obs_tensor, _ = self.used_models_filtered[0].policy.obs_to_tensor(
                observation
            )

            for model in self.used_models_filtered:
                # q_val_tmp: Tensor = model.policy.q_net(obs_tensor)
                q_values: Tensor = model.policy.q_net(
                    obs_tensor
                )  # q_val_tmp.flatten()
                q_values_lists.append(q_values)

            q_values_lists_tensor: Tensor = torch.stack(q_values_lists, dim=0)
            conjugated_q_values: Tensor = torch.min(
                q_values_lists_tensor, dim=0
            ).values
            action_tensor = torch.argmax(conjugated_q_values, dim=-1)
            action = action_tensor.detach().cpu().numpy()

        return action, None


class PPOCompositePolicy(
    CompositePolicy[np.ndarray | dict[str, np.ndarray], PPO]
):
    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        with torch.no_grad():
            if isinstance(observation["aut_state"], int):
                observation["aut_state"] = 1
            else:
                match observation["aut_state"].ndim:
                    case 1:
                        observation["aut_state"][:] = 1
                    case 0:
                        observation["aut_state"] = np.array(
                            1, dtype=observation["aut_state"].dtype
                        )

            obs_tensor, _ = self.used_models_filtered[0].policy.obs_to_tensor(
                observation
            )
            # Get action probabilities from each model
            action_prob_lists: list[Tensor] = []
            for model in self.used_models_filtered:
                action_dist: CategoricalDistribution = (
                    model.policy.get_distribution(obs_tensor)
                )
                action_probs = action_dist.distribution.probs
                action_prob_lists.append(action_probs)

            # Combine action probabilities by multiplying them
            combined_action_probs: Tensor = torch.prod(
                torch.stack(action_prob_lists), dim=0
            )
            combined_action_probs /= torch.sum(
                combined_action_probs
            )  # Normalize
            combined_distribution = Categorical(combined_action_probs)

            action = combined_distribution.sample().detach().cpu().numpy()

        return action, None


class CompositeLowLevelPolicyConfig(BaseModel):
    composite_policy_class: type[CompositePolicy]
    primitive_pipeline_map: dict[str, SB3Pipeline]
    # primitive_pipeline_map: dict[str, BaseAlgorithm]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PrimitivePipelineMapConfigReader(BaseModel):
    prim_spec_map_config_file: str
    pipeline_config_file: str


class CompositeLowLevelPolicyConfigReader(
    BaseModel, ConfigReader[CompositeLowLevelPolicyConfig]
):
    composite_policy_class: str
    primitive_pipeline_map_config: PrimitivePipelineMapConfigReader

    def to_config(self) -> CompositeLowLevelPolicyConfig:
        composite_policy_class: type[CompositePolicy] = get_class(
            self.composite_policy_class
        )
        primitive_pipeline_map: dict[str, SB3Pipeline] = {}
        # primitive_pipeline_map: dict[str, BaseAlgorithm] = {}
        with open(
            self.primitive_pipeline_map_config.prim_spec_map_config_file, "r"
        ) as f:
            prim_spec_map: dict[str, str] = yaml.safe_load(f)["mapping"]

        training_config_reader: SB3PipelineConfigReader = (
            SB3PipelineConfigReader.from_yaml(
                self.primitive_pipeline_map_config.pipeline_config_file
            )
        )
        for prim, tl_spec in prim_spec_map.items():
            tl_config_reader = TLSB3PipelineConfigReader(
                tl_spec=tl_spec, pipeline_config=training_config_reader
            )
            tl_config = tl_config_reader.to_config()
            primitive_pipeline_map[prim] = SB3Pipeline(
                tl_config, verbose=False
            )  # .load_model("best", device="cuda:2")

        config = CompositeLowLevelPolicyConfig(
            composite_policy_class=composite_policy_class,
            primitive_pipeline_map=primitive_pipeline_map,
        )
        return config


class SkillMachinesLowLevelPolicyBuffer(
    LowLevelPolicyBuffer[CompositeLowLevelPolicyConfig]
):
    def __init__(self, policy_args: CompositeLowLevelPolicyConfig) -> None:
        super().__init__(policy_args)

        # primitive_model_map: dict[str, BaseAlgorithm] = {}
        # for spec, pipeline_config in policy_args.primitive_pipeline_map.items():
        #     pipeline = SB3Pipeline(pipeline_config, verbose=False)
        #     primitive_model_map[spec] = pipeline.load_model("best")

        if isinstance(policy_args, dict):
            primitive_model_map: dict[str, SB3Pipeline] = policy_args[
                "primitive_pipeline_map"
            ]
        else:
            primitive_model_map: dict[str, SB3Pipeline] = (
                policy_args.primitive_pipeline_map
            )

        primitive_model_maps: dict[str, BaseAlgorithm] = {}
        for spec, pipeline in primitive_model_map.items():
            # pipeline = SB3Pipeline(pipeline_config, verbose=False)
            primitive_model_maps[spec] = pipeline.load_model("best")
        self.primitive_model_map: dict[str, BaseAlgorithm] = (
            primitive_model_maps
        )
        self.primitives: list[str] = list(primitive_model_maps.keys())


class SkillMachinesLowLevelPolicy(
    LowLevelPolicy[
        CompositePolicy, CompositeLowLevelPolicyConfig, ObsType, NDArray
    ]
):
    policy_args_validator = CompositeLowLevelPolicyConfig
    policy_args_reader = CompositeLowLevelPolicyConfigReader
    buffer_class = SkillMachinesLowLevelPolicyBuffer

    def define_policy(
        self, policy_args: CompositeLowLevelPolicyConfig
    ) -> CompositePolicy:
        assert self.buffer

        buffer: SkillMachinesLowLevelPolicyBuffer = self.buffer  # type: ignore

        if isinstance(policy_args, dict):
            policy_args = CompositeLowLevelPolicyConfig(**policy_args)

        policy: CompositePolicy = policy_args.composite_policy_class(
            tl_spec=self.tl_spec,
            primitives=buffer.primitives,
            primitive_model_map=buffer.primitive_model_map,
        )

        return policy

    def act(
        self,
        obs: TLObs[ObsType],
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, NDArray] | None = None,
        tl_wrapper_args: dict[str, Any] = {},
    ) -> NDArray:
        """
        Predict the action using the low-level policy.
        """
        if current_env:
            self.policy.filter_terminal_predicates(
                obs, current_env, tl_wrapper_args
            )
        action, _ = self.policy.predict(obs)
        # Ensure action is a numpy int64 scalar
        return action

    def delete_policy(self) -> None:
        del self.policy
