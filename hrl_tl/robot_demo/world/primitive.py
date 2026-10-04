"""Native primitive dynamics completed by measured robot positions."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
from contgrid.core import world as native_world
from contgrid.envs.zone import env as zone_env

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action
from hrl_tl.robot_demo.world import arena


class PrimitiveZone:
    """A native Zone state advanced once per completed robot movement.

    Attributes:
        env: Native environment consumed by the unmodified low-level policies.
        observation: Policy observation, including native velocity state.
        info: Native task information at the last completed primitive step.
        reason: Terminal task reason, or an empty string.
        steps: Number of completed primitive transitions.
    """

    def __init__(self, layout: arena.Arena) -> None:
        self.env: zone_env.ZoneEnv = zone_env.ZoneEnv(**layout.environment)
        self.observation: dict[str, np.ndarray]
        self.info: dict[str, Any]
        self.observation, self.info = self.env.reset(seed=layout.seed)
        # ContGrid agents otherwise share mutable constructor defaults.
        for agent in self.env.env.world.agents:
            agent.state = copy.deepcopy(agent.state)
            agent.action = copy.deepcopy(agent.action)
        self.reason: str = ""
        self.steps: int = 0
        self._layout: arena.Arena = layout
        self._planned: native_world.World | None = None
        self._action: np.ndarray | None = None
        self._stamp: float = float("-inf")

    def start(self, measured: pose.Pose2D, *, evaluate: bool = True) -> None:
        """Initialize the position from mocap, with the native reset velocity."""
        self._set_position(measured)
        agent = self.env.env.world.agents[0]
        agent.state.vel = np.zeros(2, dtype=np.float64)
        if evaluate:
            self._evaluate()
        else:
            self.observation = self.env.scenario.observation(
                agent, self.env.env.world
            )
            self.info = self.env.scenario.info(agent, self.env.env.world)
            if not self.env.observation_space.contains(self.observation):
                raise ValueError("Measured state is outside the native Zone space")

    def plan(self, selected: np.ndarray) -> action.NativeMovementAction:
        """Translate a discrete direction and magnitude to a world waypoint.

        Native physics still updates the checkpoint's velocity state on a
        copy. Position and task state advance only after the measured move.
        """
        if self._planned is not None:
            raise RuntimeError("The previous primitive is still moving")
        if not self.env.action_space.contains(selected):
            raise ValueError("Invalid native primitive action")
        planned = copy.deepcopy(self.env.env.world)
        self.env.env.action_mode.update_agent_action(
            planned.agents[0], selected, planned
        )
        planned.step()
        position = self.env.env.world.agents[0].state.pos
        distance = self._layout.robot.step_lengths_sim[int(selected[1])]
        angle = int(selected[0]) * math.tau / 8
        native_target = pose.zone_to_world(
            pose.Point2D(
                x=float(position[0] + distance * math.cos(angle)),
                y=float(position[1] + distance * math.sin(angle)),
            ),
            self._layout.robot.frame,
        )
        target = native_target
        if self._layout.physical_walls:
            # Physical rooms retain the calibrated footprint and tracking inset.
            bounds = self._layout.robot.bounds
            inset = (
                bounds.margin
                + self._layout.robot.motion.wall_stop_margin
                + self._layout.robot.motion.position_tolerance
            )
            target = pose.Point2D(
                x=float(
                    np.clip(
                        native_target.x,
                        bounds.min_x + inset,
                        bounds.max_x - inset,
                    )
                ),
                y=float(
                    np.clip(
                        native_target.y,
                        bounds.min_y + inset,
                        bounds.max_y - inset,
                    )
                ),
            )
        self._planned = planned
        self._action = selected.copy()
        return action.NativeMovementAction(
            direction_index=int(selected[0]),
            magnitude_index=int(selected[1]),
            target=target,
            native_target=native_target,
        )

    def complete(
        self, measured: pose.Pose2D
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Commit native velocity and the measured endpoint exactly once."""
        if self._planned is None:
            raise RuntimeError("No primitive is waiting for completion")
        if measured.stamp <= self._stamp:
            raise ValueError("Primitive completion requires a new pose")
        base = self.env.env
        agent = base.world.agents[0]
        planned_agent = self._planned.agents[0]
        agent.state = copy.deepcopy(planned_agent.state)
        agent.action = copy.deepcopy(planned_agent.action)
        self._set_position(measured)
        self.steps += 1
        base.steps += 1
        base.current_actions[0] = self._action
        reward = self._evaluate()
        base.rewards[agent.name] = reward
        base._cumulative_rewards[agent.name] = reward
        base.terminations[agent.name] = agent.terminated
        base.infos[agent.name] = self.info
        self._planned = None
        self._action = None
        return self.observation, reward, agent.terminated, False, self.info

    def close(self) -> None:
        """Release the native environment without committing pending motion."""
        self.env.close()

    def _set_position(self, measured: pose.Pose2D) -> None:
        point = pose.world_to_zone(measured, self._layout.robot.frame)
        self.env.env.world.agents[0].state.pos = np.array(
            [point.x, point.y], dtype=np.float64
        )
        self._stamp = measured.stamp

    def _evaluate(self) -> float:
        world = self.env.env.world
        agent = world.agents[0]
        reward = float(self.env.scenario.reward(agent, world))
        self.observation = self.env.scenario.observation(agent, world)
        self.info = self.env.scenario.info(agent, world)
        if not self.env.observation_space.contains(self.observation):
            raise ValueError("Measured state is outside the native Zone space")
        if agent.terminated:
            self.reason = (
                "task_success" if self.info["is_success"] else "task_failure"
            )
        return reward
