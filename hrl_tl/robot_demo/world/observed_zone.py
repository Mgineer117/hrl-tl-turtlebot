"""Native Zone observations and task progression from measured robot poses."""

from __future__ import annotations

from typing import Any

import numpy as np
from contgrid.envs.zone import env as zone_env

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world import arena


class ObservedZone:
    """A data-only native environment driven by measurements, not dynamics.

    Attributes:
        env: Native environment safe to copy for existing TL wrappers.
        observation: Most recent native observation.
        info: Most recent native task information.
        reason: Latched task termination reason, or an empty string.
    """

    def __init__(self, layout: arena.Arena) -> None:
        self.env: zone_env.ZoneEnv = zone_env.ZoneEnv(**layout.environment)
        self.observation: dict[str, np.ndarray]
        self.info: dict[str, Any]
        self.observation, self.info = self.env.reset(seed=layout.seed)
        self.reason: str = ""
        self._layout: arena.Arena = layout
        self._previous: pose.Pose2D | None = None

    def update(self, measured: pose.Pose2D) -> bool:
        """Process a source-ordered pose once, including visits during motion."""
        previous = self._previous
        if previous is not None and measured.stamp <= previous.stamp:
            return False
        point = pose.world_to_zone(measured, self._layout.robot.frame)
        world = self.env.env.world
        agent = world.agents[0]
        position = np.array([point.x, point.y], dtype=np.float64)
        velocity = np.zeros(2, dtype=np.float64)
        if previous is not None:
            old = pose.world_to_zone(previous, self._layout.robot.frame)
            velocity = (position - [old.x, old.y]) / (
                measured.stamp - previous.stamp
            )
        agent.state.pos = position
        agent.state.vel = velocity
        agent.state.rot = measured.yaw + self._layout.robot.frame.rotation_rad
        self._previous = measured
        if not self.reason:
            # Native reward also updates entry counts and ordered subtasks.
            self.env.scenario.reward(agent, world)
            if agent.terminated:
                self.reason = (
                    "task_success"
                    if self.env.scenario.is_success
                    else "task_failure"
                )
        self.observation = self.env.scenario.observation(agent, world)
        self.info = self.env.scenario.info(agent, world)
        if not self.env.observation_space.contains(self.observation):
            raise ValueError("Measured state is outside the native Zone space")
        return True

    def close(self) -> None:
        """Release native environment resources."""
        self.env.close()
