from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, Generic, Literal, SupportsFloat, TypedDict

import numpy as np
import yaml
from gym_tl_tools import Predicate, RewardConfigDict, TLObservationRewardConfig
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_serializer,
)
from rl_pipeline.core import ConfigReader

from hrl_tl.utils.io import get_class
from hrl_tl.wrappers.utils.spec_rep import SpecRep, SpecRepArgsDict

from .low_level_policies import (
    LowLevelPolicy,
    LowLevelPolicyBuffer,
    PolicyArgsType,
    PolicyType,
)


class TLWrapperArgsDict(TypedDict, Generic[ObsType, ActType]):
    atomic_predicates: list[Predicate]
    var_value_info_generator: Callable[
        [Env[ObsType, ActType], ObsType, dict[str, Any]], dict[str, Any]
    ]
    reward_config: RewardConfigDict
    early_termination: bool


class TLHighLevelWrapperConfig(BaseModel, Generic[PolicyType, PolicyArgsType]):
    spec_rep_cls_name: str
    spec_rep_args: SpecRepArgsDict = {}
    low_level_policy_cls_name: str
    low_level_policy_args_file_path: str = (
        "configs/fourroom/train/tl_hrl_pretrained.yaml"
    )
    max_low_level_policy_steps: int = 10
    all_formulae_file_path: str = "out/fourroom/all_formulae.json"
    invalid_tl_action: Literal["stay", "random"] = "random"
    tl_wrapper_config_file: str = "configs/fourroom/train/tl_wrapper.yaml"
    excluded_obs_keys: list[str] = Field(default_factory=list)
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def spec_rep_class(self) -> type[SpecRep]:
        rep_class = get_class(self.spec_rep_cls_name)
        assert rep_class is not None
        return rep_class

    @computed_field
    @property
    def low_level_policy_class(
        self,
    ) -> type[LowLevelPolicy]:
        pol_class = get_class(self.low_level_policy_cls_name)
        assert pol_class is not None
        return pol_class

    @computed_field
    @property
    def low_level_policy_args(self) -> PolicyArgsType:
        """
        Returns the low-level policy arguments as a Pydantic model.
        This allows for validation and serialization of the arguments.
        """
        policy_class: type[LowLevelPolicy] | None = self.low_level_policy_class

        if not policy_class:
            raise ValueError(
                f"Low-level policy class {self.low_level_policy_cls_name} not found."
            )

        # Create an instance to access the policy_args_validator
        reader: type[ConfigReader[PolicyArgsType]] = (
            policy_class.policy_args_reader
        )  # type: ignore
        validator: type[PolicyArgsType] = policy_class.policy_args_validator  # type: ignore
        with open(self.low_level_policy_args_file_path, "r") as f:
            data = yaml.safe_load(f)

        # return validator.model_validate(data)
        read_config = reader(**data)
        validator_config = read_config.to_config()
        return validator(**validator_config.model_dump())

    @computed_field
    @property
    def tl_wrapper_config(self) -> TLObservationRewardConfig:
        """
        Returns the TLObservationReward configuration as a Pydantic model.
        This allows for validation and serialization of the configuration.
        """
        with open(self.tl_wrapper_config_file, "r") as f:
            data = yaml.safe_load(f)

        return TLObservationRewardConfig.model_validate(data)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        return {
            "spec_rep_class": self.spec_rep_class,
            "spec_rep_args": self.spec_rep_args,
            "low_level_policy_class": self.low_level_policy_class,
            "low_level_policy_args": self.low_level_policy_args,
            "max_low_level_policy_steps": self.max_low_level_policy_steps,
            "all_formulae_file_path": self.all_formulae_file_path,
            "invalid_tl_action": self.invalid_tl_action,
            "tl_wrapper_args": self.tl_wrapper_config.model_dump(),
            "excluded_obs_keys": self.excluded_obs_keys,
            "verbose": self.verbose,
        }


class TLHighLevelWrapper(
    Wrapper[ObsType, NDArray[np.integer], ObsType, ActType],
    RecordConstructorArgs,
    Generic[ObsType, ActType, PolicyType, PolicyArgsType],
):
    def _add_step_count_to_obs(self, obs: ObsType, policy_step: int) -> ObsType:
        """Add normalized low-level policy step count to dict observations."""
        if not isinstance(obs, dict):
            return obs

        denom = max(1, self.max_low_level_policy_steps)
        step_ratio = np.float32(policy_step / denom)

        obs_with_step = dict(obs)
        obs_with_step["step_count"] = np.array([step_ratio], dtype=np.float32)
        return obs_with_step  # type: ignore[return-value]

    def _build_observation_space_with_step_count(
        self,
        observation_space: spaces.Space,
    ) -> spaces.Space:
        """Extend Dict observation spaces with a normalized step_count field."""
        step_count_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        if not isinstance(observation_space, spaces.Dict):
            return spaces.Dict(
                observation=observation_space,
                step_count=step_count_space,
            )
        else:
            spaces_dict = dict(observation_space.spaces)
            spaces_dict["step_count"] = step_count_space
            return spaces.Dict(spaces_dict)

    def __init__(
        self,
        env: Env[ObsType, ActType],
        spec_rep_class: type[SpecRep],
        spec_rep_args: SpecRepArgsDict,
        low_level_policy_class: type[
            LowLevelPolicy[PolicyType, PolicyArgsType, ObsType, ActType]
        ],
        low_level_policy_args: PolicyArgsType,
        max_low_level_policy_steps: int = 10,
        all_formulae_file_path: str = "out/maze/all_formulae_2_cla_2_max_pred.json",
        invalid_tl_action: Literal["stay", "random"] = "stay",
        stay_action: ActType = np.int64(0),
        excluded_obs_keys: list[str] | None = None,
        tl_wrapper_args: dict[str, Any] = {},
        verbose: bool = False,
    ) -> None:
        """
        Initializes the TLHighLevelWrapper.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment to wrap.
        spec_rep_class : type[SpecRep]
            The specification representation class to use.
        spec_rep_args : SpecRepArgsDict
            Arguments for the specification representation class.
        low_level_policy : Callable[[ObsType, int, TLObservationReward[ObsType, ActType], PolicyArgsType], ActType]
            The low-level policy function that takes the observation, automaton state,
            low-level environment, and policy arguments, and returns an action.
        low_level_policy_args : PolicyArgsType, optional
            Arguments for the low-level policy function (default is an empty dictionary).
        max_low_level_policy_steps : int = 10
            The maximum number of steps the low-level policy can take before resetting.
        all_formulae_file_path : str = "out/maze/all_formulae_2_cla_2_max_pred.json"
            Path to the JSON file containing all formulae specifications.
        stay_action : ActType = np.int64(0)
            The action to take when no valid temporal logic specification is available.
        tl_wrapper_args : TLWrapperArgsDict[ObsType, ActType] = {}
            Arguments for the TLObservationReward wrapper.
        verbose : bool = False
            If True, prints verbose output during execution.
        """
        RecordConstructorArgs.__init__(
            self,
            spec_rep=spec_rep_class,
            spec_rep_args=spec_rep_args,
            low_level_policy_class=low_level_policy_class,
            # low_level_policy_args=low_level_policy_args,
            max_low_level_policy_steps=max_low_level_policy_steps,
            all_formulae_file_path=all_formulae_file_path,
            invalid_tl_action=invalid_tl_action,
            stay_action=stay_action,
            tl_wrapper_args=tl_wrapper_args,
            excluded_obs_keys=excluded_obs_keys,
            verbose=verbose,
        )
        Wrapper.__init__(self, env)

        self.excluded_obs_keys: list[str] = (
            excluded_obs_keys if excluded_obs_keys is not None else []
        )
        self.tl_wrapper_args: dict[str, Any] = tl_wrapper_args
        if isinstance(self.tl_wrapper_args["atomic_predicates"][0], dict):
            self.tl_wrapper_args["atomic_predicates"] = [
                Predicate(**predicate)
                for predicate in self.tl_wrapper_args["atomic_predicates"]
            ]
        self.predicate_names: list[str] = [
            predicate.name
            for predicate in self.tl_wrapper_args["atomic_predicates"]
        ]
        spec_rep_args.update({"predicate_names": self.predicate_names})
        self.spec_rep: SpecRep = spec_rep_class(**spec_rep_args)
        self.invalid_tl_action: Literal["stay", "random"] = invalid_tl_action
        self.stay_action: ActType = stay_action
        self.max_low_level_policy_steps: int = max_low_level_policy_steps
        self.verbose: bool = verbose

        if ".json" in all_formulae_file_path:
            with open(all_formulae_file_path, "r") as f:
                all_formulae = json.load(f)
        elif (
            ".yaml" in all_formulae_file_path
            or ".yml" in all_formulae_file_path
        ):
            with open(all_formulae_file_path, "r") as f:
                all_formulae = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file format for all_formulae_file_path: {all_formulae_file_path}"
            )

        self.specs: list[str] = all_formulae["specifications"]

        self.action_space = self.spec_rep.action_space
        self.observation_space = self._build_observation_space_with_step_count(
            self.env.observation_space
        )

        self.low_level_policy_class = low_level_policy_class
        self.low_level_policy_args = low_level_policy_args

        self.low_level_policy_buffer: LowLevelPolicyBuffer = (
            low_level_policy_class.buffer_class(low_level_policy_args)
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Saves the last observation and returns the original environment's reset observation.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs: ObsType = obs
        self.last_info: dict[str, Any] = info
        self.last_policy_args: dict[str, Any] | None = None
        self._selected_info: dict[str, str | None] = {}
        obs = self._add_step_count_to_obs(obs, policy_step=0)

        # self.low_level_policy_step: int = 0
        # self.current_tl_env: TLObservationReward[ObsType, ActType] | None = None
        self.low_level_policy: (
            LowLevelPolicy[PolicyType, PolicyArgsType, ObsType, ActType] | None
        ) = None
        self.low_level_policy_buffer.at_reset()

        return obs, info

    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute the same selection path using simulated dynamics."""
        low_level_action = self.select_action(action)
        obs, reward, terminated, truncated, info = self.env.step(
            low_level_action
        )
        obs, info = self.observe(obs, info)
        return obs, reward, terminated, truncated, info

    @property
    def current_observation(self) -> ObsType:
        """Latest observation with the existing high-level step encoding."""
        step = self.low_level_policy.policy_step if self.low_level_policy else 0
        return self._add_step_count_to_obs(self.last_obs, step)

    def select_action(
        self,
        action: NDArray[np.integer],
        *,
        allow_fallback: bool = True,
    ) -> ActType | None:
        """
        Select one low-level action without advancing environment dynamics.

        External executors call observe() with the measured result afterwards.
        With allow_fallback=False, invalid or already terminated skills return
        None instead of a random/stay action. The default preserves step().
        """

        if self.verbose:
            print(
                f"High-level action: {action},\n"
                f"Low-level policy step: {self.low_level_policy.policy_step if self.low_level_policy else 0},\n"
                "Current TL spec: " + "none"
                if not self.low_level_policy
                else f"{self.low_level_policy.tl_spec}"
            )

        if not self.low_level_policy:
            # Convert the high-level action to a temporal logic specification

            current_tl_spec: str = self.spec_rep.weights2ltl(action)

            if (
                current_tl_spec == "0"
                or current_tl_spec == "1"
                or not current_tl_spec
                or current_tl_spec not in self.specs
            ):
                if self.verbose:
                    print(f"- Invalid TL spec: {current_tl_spec}, ")
                self.low_level_policy = None
            else:
                try:
                    policy_args_update: dict[str, Any] = (
                        self.spec_rep.action2policy_args(action)
                    )
                    self.last_policy_args = policy_args_update
                    policy_args = copy.deepcopy(self.low_level_policy_args)
                    if isinstance(policy_args, dict):
                        policy_args.update(policy_args_update)
                    elif isinstance(policy_args, BaseModel):
                        policy_args = policy_args.model_copy(
                            update=policy_args_update
                        )
                    else:
                        raise ValueError(
                            f"Unsupported policy_args type: {type(policy_args)}"
                        )
                    self.low_level_policy = self.low_level_policy_class(
                        env=self.env,
                        tl_spec=current_tl_spec,
                        max_policy_steps=self.max_low_level_policy_steps,
                        policy_args=policy_args,
                        tl_wrapper_args=self.tl_wrapper_args,
                        buffer=self.low_level_policy_buffer,
                    )

                except IndexError as e:
                    raise ValueError(
                        f"Invalid TL spec: {current_tl_spec}. Error: {e}"
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Invalid TL spec: {current_tl_spec}. Error: {e}"
                    )

                self.low_level_policy.update_env(
                    self.env,
                    self.last_obs,
                    self.last_info,
                    tl_wrapper_args=self.tl_wrapper_args,
                )
                if self.low_level_policy.is_aut_terminated:
                    # If the automaton is terminated, we reset the low-level policy
                    self.low_level_policy = None
                    if self.verbose:
                        print(
                            f"- Automaton terminated for TL spec: {current_tl_spec},\n"
                            f"-- Low-level policy reset to None"
                        )
                else:
                    if self.verbose:
                        print(
                            f"- New TL spec: {self.low_level_policy.tl_spec},\n"
                            f"-- Low-level policy step reset to 0"
                        )
        else:
            pass

        self._selected_info = {
            "current_tl_spec": (
                self.low_level_policy.tl_spec if self.low_level_policy else None
            )
        }

        low_level_action: ActType

        if self.low_level_policy:
            if not allow_fallback:
                self.low_level_policy.update_env(
                    self.env,
                    self.last_obs,
                    self.last_info,
                    self.tl_wrapper_args,
                )
                if self.low_level_policy.is_aut_terminated:
                    self.low_level_policy.delete_policy()
                    self.low_level_policy = None
                    return None
            low_level_action, ll_terminated, ll_truncated = (
                self.low_level_policy.predict(
                    self.env,
                    self.last_obs,
                    self.last_info,
                    tl_wrapper_args=self.tl_wrapper_args,
                    excluded_obs_keys=self.excluded_obs_keys,
                )
            )
            if self.verbose:
                print(
                    f"- Low-level action: {low_level_action},\n"
                    f"-- Low-level policy step: {self.low_level_policy.policy_step},\n"
                    f"-- Is automaton terminated: {ll_terminated}, "
                )

            if ll_terminated or ll_truncated:
                self.low_level_policy.delete_policy()
                self.low_level_policy = None
            else:
                pass
        else:
            if not allow_fallback:
                return None
            match self.invalid_tl_action:
                case "random":
                    # If the specification is not in the list, we return a random action
                    low_level_action = self.env.action_space.sample()
                    if self.verbose:
                        print(f"-- using random action: {low_level_action}")
                case "stay":
                    # If the specification is not in the list, we return the stay action
                    low_level_action = self.stay_action
                    if self.verbose:
                        print(f"-- using stay action: {low_level_action}")
                case _:
                    raise ValueError(
                        f"Invalid invalid_tl_action: {self.invalid_tl_action}"
                    )

        return low_level_action

    def observe(
        self, obs: ObsType, info: dict[str, Any]
    ) -> tuple[ObsType, dict[str, Any]]:
        """Accept a simulated or measured state without a policy step."""
        current_policy_step = (
            self.low_level_policy.policy_step if self.low_level_policy else 0
        )

        info = dict(info)
        info.update(self._selected_info)
        info.update(self.last_policy_args if self.last_policy_args else {})
        self.last_obs = obs
        self.last_info = info

        obs = self._add_step_count_to_obs(obs, policy_step=current_policy_step)

        return obs, info
