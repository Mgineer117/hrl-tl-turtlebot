"""Training-time meta options connected to completed robot primitives."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, TextIO

import numpy as np
from stable_baselines3.common import base_class

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action
from hrl_tl.robot_demo.world import arena, primitive
from hrl_tl.wrappers import gc_ltl, tl_meta_option
from hrl_tl.wrappers.low_level_policies import utils


class RobotHierarchy:
    """The trained meta controller and native primitive state for one episode.

    Attributes:
        reason: Episode termination reason, or an empty string.
        display: Latest task and option information for the stage display.
        decision: Metadata for the last issued robot movement.
    """

    def __init__(
        self,
        layout: arena.Arena,
        upper: base_class.BaseAlgorithm,
        wrapper_kwargs: dict[str, Any],
        log: TextIO,
        max_actions: int | None,
    ) -> None:
        if max_actions is not None and max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self._layout: arena.Arena = layout
        self._upper: base_class.BaseAlgorithm = upper
        self._zone: primitive.PrimitiveZone = primitive.PrimitiveZone(layout)
        self._meta: tl_meta_option.TLMetaOptionWrapper = (
            tl_meta_option.TLMetaOptionWrapper(self._zone.env, **wrapper_kwargs)
        )
        self._meta.reset(seed=layout.seed)
        self._zone.env.action_space.seed(layout.seed)
        try:
            self._validate_spaces()
        except ValueError:
            self._zone.close()
            raise
        self._log: TextIO = log
        self._manual_limit: int | None = max_actions
        self._execution: (
            Generator[
                np.ndarray, tl_meta_option.StepResult, tl_meta_option.StepResult
            ]
            | None
        ) = None
        self._started: bool = False
        self._issued: int = 0
        self._option: dict[str, Any] = {"id": 0, "spec": None}
        self._upper_action: np.ndarray | None = None
        self._policy_args: dict[str, Any] = {}
        self.reason: str = ""
        self.decision: dict[str, Any] | None = None

    @property
    def display(self) -> dict[str, Any]:
        """Task state at the last completed primitive, plus the active option."""
        return {
            "arena_id": self._layout.identity,
            "subtask": self._zone.info["current_subtask"],
            "visits": self._zone.info["visit_counts"],
            "reason": self.reason,
            "option": self._option,
        }

    def next_action(
        self, measured: pose.Pose2D
    ) -> action.NativeMovementAction | None:
        """Complete the previous step, then resume the native option program."""
        if self.reason:
            return None
        transition = None
        if not self._started:
            self._zone.start(measured)
            self._started = True
        else:
            transition = self._zone.complete(measured)
        if self._zone.reason:
            self.reason = self._zone.reason
            if self._execution is not None and transition is not None:
                try:
                    self._execution.send(transition)
                except StopIteration:
                    self._execution = None
        if (
            self._manual_limit is not None
            and self._issued >= self._manual_limit
            and not self.reason
        ):
            self.reason = "episode_limit"
        if self.reason:
            self.close_option()
            return None
        selected = self._resume(transition)
        if selected is None:
            return None
        command = self._zone.plan(np.asarray(selected))
        self._issued += 1
        self.decision = {
            "decision": self._issued,
            "upper_action": self._upper_action.tolist(),
            "spec": self._option["spec"],
            "option": dict(self._option),
            "policy_args": self._policy_args,
        }
        self._log.write(
            json.dumps(
                {
                    "event": "policy",
                    **self.decision,
                    "arena_id": self._layout.identity,
                    "pose": measured.model_dump(),
                    "observation": {
                        key: np.asarray(value).tolist()
                        for key, value in self._zone.observation.items()
                    },
                    "action": command.model_dump(
                        exclude={"target", "native_target"}
                    ),
                    "target": command.target.model_dump(),
                    "native_target": command.native_target.model_dump(),
                    "target_clipped": command.target != command.native_target,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self._log.flush()
        return command

    def close_option(self) -> None:
        """Release an active low-level policy without another inference."""
        if self._execution is not None:
            self._execution.close()
            self._execution = None

    def close(self) -> None:
        """Release the option and native environment."""
        self.close_option()
        self._zone.close()

    def _resume(
        self, transition: tl_meta_option.StepResult | None
    ) -> np.ndarray | None:
        if self._execution is not None:
            try:
                return self._execution.send(transition)
            except StopIteration as completed:
                self._execution = None
                _, _, terminated, truncated, _ = completed.value
                if terminated or truncated:
                    self.reason = self._zone.reason or "episode_limit"
                    return None
                # Native PrimitiveStepTimeLimit checks at option boundaries.
                if self._zone.steps >= self._layout.max_steps:
                    self.reason = "episode_limit"
                    return None
        self._upper_action, _ = self._upper.predict(
            self._zone.observation, deterministic=True
        )
        spec = self._meta.spec_rep.weights2ltl(self._upper_action)
        self._option = {
            "id": self._option["id"] + 1,
            "spec": spec,
            "black_in_spec": "psi_b" in spec,
        }
        self._policy_args = self._meta.spec_rep.action2policy_args(
            self._upper_action
        )
        if self._issued == 0:
            self._meta.observe(self._zone.observation, self._zone.info)
        self._execution = self._meta.execute_option(self._upper_action)
        return next(self._execution)

    def _validate_spaces(self) -> None:
        if self._upper.observation_space != self._zone.env.observation_space:
            raise ValueError("PPO observation space does not match this arena")
        if self._upper.action_space != self._meta.action_space:
            raise ValueError("PPO action space does not match the TL wrapper")
        if not np.array_equal(self._zone.env.action_space.nvec, [8, 5]):
            raise ValueError("Robot movement requires MultiDiscrete([8, 5])")
        lower_args = self._meta.low_level_policy_args
        lower = lower_args["model"]
        predicates = self._meta.tl_wrapper_args["atomic_predicates"]
        representation = (
            gc_ltl.OneHotGoalRep(predicates)
            if lower_args["goal_rep"] == "one_hot"
            else gc_ltl.IndexGoalRep(predicates)
        )
        lower_obs = {
            key: value
            for key, value in self._zone.observation.items()
            if key not in self._meta.excluded_obs_keys
        }
        lower_obs["aut_state"] = np.array([0])
        goal_obs = utils.create_goal_conditioning_obs(
            lower_obs, 0, representation
        )
        if not lower.observation_space.contains(goal_obs):
            raise ValueError("CPC observation space does not match this arena")
        if lower.action_space != self._zone.env.action_space:
            raise ValueError("CPC action space does not match this arena")
