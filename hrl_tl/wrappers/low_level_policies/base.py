import copy
from abc import ABC, abstractmethod
from typing import Any, Generic, TypedDict, TypeVar

import torch
from gym_tl_tools import TLObservationReward
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from pydantic import BaseModel
from rl_pipeline.core import ConfigReader

PolicyType = TypeVar("PolicyType")
PolicyArgsType = TypeVar("PolicyArgsType", bound=BaseModel)


class TLObs(TypedDict, Generic[ObsType]):
    obs: ObsType
    aut_state: int


class LowLevelPolicyBuffer(Generic[PolicyArgsType]):
    def __init__(self, policy_args: PolicyArgsType) -> None:
        self.policy_args = policy_args

    def at_reset(self) -> None:
        pass


class LowLevelPolicy(
    Generic[PolicyType, PolicyArgsType, ObsType, ActType], ABC
):
    policy_args_validator: type[PolicyArgsType] = BaseModel  # type: ignore
    policy_args_reader: type[ConfigReader[PolicyArgsType]] = ConfigReader  # type: ignore
    buffer_class: type[LowLevelPolicyBuffer] = LowLevelPolicyBuffer  # type: ignore

    def __init__(
        self,
        env: Env[ObsType, ActType],
        tl_spec: str,
        max_policy_steps: int,
        policy_args: PolicyArgsType,
        tl_wrapper_args: dict[str, Any] = {},
        buffer: LowLevelPolicyBuffer[PolicyArgsType] | None = None,
    ) -> None:
        self.env: Env[ObsType, ActType] = env
        self.tl_spec: str = tl_spec
        self.max_policy_steps: int = max_policy_steps
        self.tl_wrapper_args: dict[str, Any] = tl_wrapper_args
        self.policy_step: int = 0
        self.buffer: LowLevelPolicyBuffer[PolicyArgsType] | None = buffer
        self.policy: PolicyType = self.define_policy(policy_args)

    def update_env(
        self,
        current_env: Env[ObsType, ActType],
        obs: ObsType,
        info: dict[str, Any],
        tl_wrapper_args: dict[str, Any],
    ) -> None:
        """Update the environment and observation."""

        # Remove "tl_spec" from tl_wrapper_args if it exists
        tl_wrapper_args.pop("tl_spec", None)

        self.tl_env = TLObservationReward[ObsType, ActType](
            copy.deepcopy(current_env), tl_spec=self.tl_spec, **tl_wrapper_args
        )
        self.tl_env.automaton.reset()
        self.tl_env.forward_aut(obs, info)

    @property
    def is_aut_terminated(self) -> bool:
        """Check if the automaton has terminated."""
        return self.tl_env.is_aut_terminated

    def predict(
        self,
        current_env: Env[ObsType, ActType],
        obs: ObsType,
        info: dict[str, Any],
        tl_wrapper_args: dict[str, Any],
        excluded_obs_keys: list[str],
    ) -> tuple[ActType, bool, bool]:
        """
        Predict the action using the low-level policy.

        Parameters
        ----------
        current_env: Env[ObsType, ActType]
            The current environment in which the policy is acting.
        obs: ObsType
            The observation from the high-level environment.
        info: dict[str, Any]
            Additional information from the high-level environment, such as the current state of the automaton.

        Returns
        -------
        action: ActType
            The action predicted by the low-level policy.
        terminated: bool
            Whether the low-level policy has terminated.
        truncated: bool
            Whether the low-level policy has been truncated.

        """
        self.update_env(current_env, obs, info, tl_wrapper_args)
        aut_state: int = self.tl_env.automaton.current_state
        terminated: bool = self.tl_env.is_aut_terminated

        if isinstance(obs, dict):
            obs = copy.deepcopy(obs)
            for key in excluded_obs_keys:
                obs.pop(key, None)
            obs_input: TLObs[ObsType] = {**obs, "aut_state": aut_state}
        else:
            obs_input: TLObs[ObsType] = {"obs": obs, "aut_state": aut_state}
        action = self.act(obs_input, info, current_env, tl_wrapper_args)
        self.policy_step += 1
        truncated: bool = self.policy_step >= self.max_policy_steps

        return action, terminated, truncated

    def delete_policy(self) -> None:
        del self.policy
        torch.cuda.empty_cache()

    @abstractmethod
    def define_policy(self, policy_args: PolicyArgsType) -> PolicyType: ...

    @abstractmethod
    def act(
        self,
        obs: TLObs[ObsType],
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, ActType] | None = None,
        tl_wrapper_args: dict[str, Any] = {},
    ) -> ActType: ...
