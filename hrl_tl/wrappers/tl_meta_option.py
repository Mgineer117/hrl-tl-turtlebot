from __future__ import annotations

import copy
import json
import logging
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, SupportsFloat, override

import gymnasium as gym
import numpy as np
import yaml
from gym_tl_tools import Predicate
from gymnasium import Wrapper
from gymnasium.core import ActType, ObsType
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from pydantic import BaseModel
from sb3_hrl.option.wrappers import OptionEnvWrapper, PrimitiveStepTimeLimit

from hrl_tl.wrappers.low_level_policies import (
    LowLevelPolicy,
    LowLevelPolicyBuffer,
)
from hrl_tl.wrappers.utils.spec_rep import SpecRep

_logger = logging.getLogger(__name__)


type StepResult[Observation] = tuple[
    Observation, SupportsFloat, bool, bool, dict[str, Any]
]


class TLMetaOptionWrapper[ObsType, ActType](
    OptionEnvWrapper[ObsType, NDArray[np.integer], ObsType, ActType],
    RecordConstructorArgs,
):
    """A gymnasium wrapper executing temporal logic options over primitive actions.

    Attributes:
        spec_rep: Representation mapping discrete meta-actions to temporal logic specifications.
        specs: Set of all valid specification strings.
        low_level_policy_class: Class used to instantiate low-level option controllers.
        low_level_policy_args: Base arguments passed to low-level policy instances.
        max_low_level_policy_steps: Cutoff limit on primitive steps per option execution.
        tl_wrapper_args: Environment arguments forwarded to low-level policies.
        predicate_names: List of atomic predicate names available in the environment.
        excluded_obs_keys: Keys stripped before low-level policy evaluation.
        reward_type: Reward accumulation scheme ('smdp' or 'intra_option').
        gamma: Discount factor used when reward_type is 'smdp'.
        verbose: Whether to log option transitions to stdout.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        spec_rep_class: type[SpecRep[Any]] | None = None,
        spec_rep_args: Mapping[str, Any] | None = None,
        spec_rep: SpecRep[Any] | None = None,
        low_level_policy_class: type[LowLevelPolicy[Any, Any, ObsType, ActType]]
        | None = None,
        low_level_policy_args: Mapping[str, Any] | None = None,
        max_low_level_policy_steps: int = 10,
        all_formulae_file_path: str = "out/maze/all_formulae_2_cla_2_max_pred.json",
        tl_wrapper_args: Mapping[str, Any] | None = None,
        excluded_obs_keys: Sequence[str] | None = None,
        reward_type: Literal["smdp", "intra_option"] = "smdp",
        gamma: float = 0.99,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the meta-option wrapper.

        Args:
            env: The underlying base environment to wrap.
            spec_rep_class: Specification representation class to instantiate.
            spec_rep_args: Arguments for the specification representation class.
            spec_rep: Already instantiated specification representation.
            low_level_policy_class: Low-level policy constructor.
            low_level_policy_args: Arguments for the low-level policy.
            max_low_level_policy_steps: Maximum primitive execution steps per option.
            all_formulae_file_path: Path to the JSON/YAML file containing valid formulae.
            tl_wrapper_args: Environment wrapper arguments forwarded to subpolicies.
            excluded_obs_keys: Keys excluded from low-level policy observations.
            reward_type: Mode of reward accumulation ('smdp' or 'intra_option').
            gamma: Discount factor used when reward_type is 'smdp'.
            verbose: Whether to print verbose output during execution.
            **kwargs: Ignored extra arguments for backward compatibility.
        """
        RecordConstructorArgs.__init__(
            self,
            spec_rep_class=spec_rep_class,
            spec_rep_args=spec_rep_args,
            spec_rep=spec_rep,
            low_level_policy_class=low_level_policy_class,
            max_low_level_policy_steps=max_low_level_policy_steps,
            all_formulae_file_path=all_formulae_file_path,
            tl_wrapper_args=tl_wrapper_args,
            excluded_obs_keys=excluded_obs_keys,
            reward_type=reward_type,
            gamma=gamma,
            verbose=verbose,
        )
        OptionEnvWrapper.__init__(self, env)  # type: ignore[arg-type]
        self.env: gym.Env[ObsType, ActType] = env

        self.tl_wrapper_args: dict[str, Any] = dict(tl_wrapper_args or {})
        if "atomic_predicates" in self.tl_wrapper_args and isinstance(
            self.tl_wrapper_args["atomic_predicates"], list
        ):
            if self.tl_wrapper_args["atomic_predicates"] and isinstance(
                self.tl_wrapper_args["atomic_predicates"][0], dict
            ):
                self.tl_wrapper_args["atomic_predicates"] = [
                    Predicate(**pred)
                    for pred in self.tl_wrapper_args["atomic_predicates"]
                ]
            self.predicate_names: list[str] = [
                pred.name for pred in self.tl_wrapper_args["atomic_predicates"]
            ]
        else:
            self.predicate_names = []

        if spec_rep is not None:
            self.spec_rep = spec_rep
        elif spec_rep_class is not None:
            rep_args: dict[str, Any] = dict(spec_rep_args or {})
            if self.predicate_names:
                rep_args["predicate_names"] = self.predicate_names
            self.spec_rep = spec_rep_class(**rep_args)
        else:
            raise ValueError(
                "Either spec_rep or spec_rep_class must be provided."
            )

        self.action_space = self.spec_rep.action_space
        self.observation_space = self.env.observation_space

        self.specs: set[str] = self._load_specs(all_formulae_file_path)
        self.low_level_policy_class = low_level_policy_class
        self.low_level_policy_args: dict[str, Any] = dict(
            low_level_policy_args or {}
        )
        self.max_low_level_policy_steps = max_low_level_policy_steps
        self.excluded_obs_keys: list[str] = list(excluded_obs_keys or [])
        self.reward_type = reward_type
        self.gamma = gamma
        self.verbose = verbose

        self._low_level_policy_buffer: LowLevelPolicyBuffer[Any] | None = None
        if self.low_level_policy_class is not None:
            self._low_level_policy_buffer = (
                self.low_level_policy_class.buffer_class(
                    self.low_level_policy_args
                )
            )

        self._last_obs: ObsType | None = None
        self._last_info: dict[str, Any] | None = None

    @classmethod
    def _load_specs(cls, file_path: str | Path) -> set[str]:
        """Loads valid specifications from a JSON or YAML file.

        Args:
            file_path: Path to the specification dictionary file.

        Returns:
            Set of valid temporal logic specification strings.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        try:
            with path.open("r", encoding="utf-8") as f:
                if suffix == ".json":
                    data = json.load(f)
                elif suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported file format for formulae file: {path}"
                    )
            return set(data.get("specifications", []))
        except FileNotFoundError:
            _logger.exception("Formulae file not found: %s", path)
            raise

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment and option state.

        Args:
            seed: Seed for the environment random number generator.
            options: Additional options for environment reset.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        self.reset_render_frames()
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._last_info = info
        if self._low_level_policy_buffer is not None:
            self._low_level_policy_buffer.at_reset()
        return obs, info

    def observe(self, obs: ObsType, info: dict[str, Any]) -> None:
        """Set the initial measured state without executing a primitive."""
        self._last_obs = obs
        self._last_info = info

    def _step_random_primitive(
        self, current_tl_spec: str | None
    ) -> Generator[ActType, StepResult[ObsType], StepResult[ObsType]]:
        """Executes a single random action in the base environment as fallback.

        Args:
            current_tl_spec: Decoded specification string or None if invalid.

        Yields:
            Primitive action to execute before sending back its transition.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        self.reset_render_frames()
        action_prim = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = yield action_prim
        self.record_primitive_frame()
        self._last_obs = obs
        self._last_info = info
        info.update(
            {
                "invalid_option": True,
                "current_tl_spec": current_tl_spec,
                "meta_option_steps": 1,
            }
        )
        if self.record_render_frames:
            info["render_frames"] = self.pop_render_frames()
        return obs, float(reward), terminated, truncated, info

    @override
    def step(
        self, action: NDArray[np.integer]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute an option with the native environment's primitive steps."""
        execution = self.execute_option(action)
        try:
            primitive = next(execution)
            while True:
                primitive = execution.send(self.env.step(primitive))
        except StopIteration as completed:
            return completed.value
        finally:
            execution.close()

    def execute_option(
        self, action: NDArray[np.integer]
    ) -> Generator[ActType, StepResult[ObsType], StepResult[ObsType]]:
        """Execute an option by yielding actions and receiving completed steps.

        The robot and step() share the same option execution rules.

        Args:
            action: Discrete meta-action representing formula weights.

        Yields:
            Primitive action to execute before sending back its transition.

        Returns:
            Tuple of (obs, accumulated_reward, terminated, truncated, info).

        Raises:
            TypeError: If low_level_policy_class is missing or policy args type is invalid.
        """
        if self.low_level_policy_class is None:
            raise TypeError(
                "low_level_policy_class must be configured to step."
            )

        if self._last_obs is None:
            self._last_obs, self._last_info = self.reset()

        current_tl_spec: str = self.spec_rep.weights2ltl(action)

        # 1. Guard against invalid specifications.
        if (
            current_tl_spec in ("0", "1", "")
            or current_tl_spec not in self.specs
        ):
            if self.verbose:
                print(f"- Invalid TL spec: {current_tl_spec}")
            return (
                yield from self._step_random_primitive(current_tl_spec=None)
            )

        # 2. Build low-level policy instance.
        policy_args_update: dict[str, Any] = self.spec_rep.action2policy_args(
            action
        )
        policy_args = copy.deepcopy(self.low_level_policy_args)
        if isinstance(policy_args, dict):
            policy_args.update(policy_args_update)
        elif isinstance(policy_args, BaseModel):
            policy_args = policy_args.model_copy(update=policy_args_update)
        else:
            raise TypeError(
                f"Unsupported policy_args type: {type(policy_args)}"
            )

        low_level_policy = self.low_level_policy_class(
            env=self.env,
            tl_spec=current_tl_spec,
            max_policy_steps=self.max_low_level_policy_steps,
            policy_args=policy_args,
            tl_wrapper_args=self.tl_wrapper_args,
            buffer=self._low_level_policy_buffer,
        )
        low_level_policy.update_env(
            self.env,
            self._last_obs,
            self._last_info or {},
            tl_wrapper_args=self.tl_wrapper_args,
        )

        # 3. Guard against already terminated automaton.
        if low_level_policy.is_aut_terminated:
            if self.verbose:
                print(
                    f"- Automaton already terminated for spec: {current_tl_spec}"
                )
            low_level_policy.delete_policy()
            return (
                yield from self._step_random_primitive(
                    current_tl_spec=current_tl_spec
                )
            )

        # 4. Macro option execution loop (nesting depth <= 2).
        self.reset_render_frames()
        total_reward: float = 0.0
        effective_gamma: float = 1.0
        steps: int = 0

        try:
            while True:
                ll_action, ll_terminated, ll_truncated = (
                    low_level_policy.predict(
                        self.env,
                        self._last_obs,
                        self._last_info or {},
                        tl_wrapper_args=self.tl_wrapper_args,
                        excluded_obs_keys=self.excluded_obs_keys,
                    )
                )
                obs, reward, terminated, truncated, info = yield ll_action
                self.record_primitive_frame()

                if self.reward_type == "smdp":
                    total_reward += effective_gamma * float(reward)
                else:
                    total_reward += float(reward)
                effective_gamma *= self.gamma

                steps += 1
                self._last_obs = obs
                self._last_info = info

                if terminated or truncated or ll_terminated or ll_truncated:
                    break

        finally:
            low_level_policy.delete_policy()

        info.update(
            {
                "invalid_option": False,
                "current_tl_spec": current_tl_spec,
                "meta_option_steps": steps,
            }
        )
        info.update(policy_args_update)
        if self.record_render_frames:
            info["render_frames"] = self.pop_render_frames()
        return obs, total_reward, terminated, truncated, info


class TLMetaOptionPrimitiveStepTimeLimitWrapper[ObsType, ActType](
    Wrapper[ObsType, NDArray[np.integer], ObsType, NDArray[np.integer]],
    RecordConstructorArgs,
):
    """A composite wrapper combining TLMetaOptionWrapper with PrimitiveStepTimeLimit.

    Attributes:
        meta_env: The underlying TLMetaOptionWrapper instance.
        max_episode_steps: Maximum primitive steps before episode truncation.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        max_episode_steps: int,
        **wrapper_kwargs: Any,
    ) -> None:
        """Initializes the composite wrapper.

        Args:
            env: The underlying base environment.
            max_episode_steps: Maximum primitive steps allowed before episode truncation.
            **wrapper_kwargs: Arguments forwarded to TLMetaOptionWrapper.
        """
        RecordConstructorArgs.__init__(
            self,
            env=env,
            max_episode_steps=max_episode_steps,
            **wrapper_kwargs,
        )
        self.meta_env: TLMetaOptionWrapper[ObsType, ActType] = (
            TLMetaOptionWrapper(env, **wrapper_kwargs)
        )
        self.max_episode_steps = max_episode_steps
        wrapped_env: PrimitiveStepTimeLimit = PrimitiveStepTimeLimit(
            self.meta_env, max_episode_steps=max_episode_steps
        )
        super().__init__(wrapped_env)  # type: ignore
