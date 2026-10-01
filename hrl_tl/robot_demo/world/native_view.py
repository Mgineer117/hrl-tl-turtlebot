"""Display measured robot state with the original ContGrid Zone renderer."""

from __future__ import annotations

import copy
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from contgrid.envs.zone import env as zone_env

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world import arena


class NativeZoneView:
    """A display copy of ZoneEnv; it never steps dynamics or runs a policy."""

    def __init__(
        self,
        layout: arena.Arena,
        *,
        render_mode: Literal["human", "rgb_array"] = "human",
    ) -> None:
        environment = copy.deepcopy(layout.environment)
        environment["render_mode"] = render_mode
        self.env = zone_env.ZoneEnv(**environment)
        self.env.reset(seed=layout.seed)
        self._layout = layout
        self._previous: pose.Pose2D | None = None
        self._state: dict[str, Any] = {}

    def update_pose(self, measured: pose.Pose2D) -> bool:
        """Move only the displayed agent, accepting advancing source stamps."""
        previous = self._previous
        if previous is not None and measured.stamp <= previous.stamp:
            return False
        point = pose.world_to_zone(measured, self._layout.robot.frame)
        agent = self.env.env.world.agents[0]
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
        return True

    def update_state(self, state: dict[str, Any]) -> bool:
        """Use task progress from the running policy, including late joins."""
        if state.get("arena_id") != self._layout.identity:
            return False
        self._state = copy.deepcopy(state)
        scenario = self.env.scenario
        scenario.current_subtask_idx = int(state["subtask"]["idx"])
        for color, count in state["visits"].items():
            if color in ("yellow", "red", "white", "black"):
                setattr(scenario, f"{color}_visit_count", int(count))
        scenario.is_success = state.get("reason") == "task_success"
        self.env.env.world.agents[0].terminated = bool(state.get("reason"))
        return True

    @property
    def window_open(self) -> bool:
        """Do not reopen a figure the user has closed."""
        figure = self.env.env.fig
        return figure is None or plt.fignum_exists(figure.number)

    def render(self) -> np.ndarray | None:
        """Call the original ZoneEnv.render() without duplicating its drawing."""
        native = self.env.env
        native.enable_render(native.render_mode)
        title = "Zone 2D | waiting for policy"
        if self._state:
            goal = self._state["subtask"].get("goal")
            option = self._state.get("option", {})
            title = (
                f"Zone 2D | goal: {goal} | option {option.get('id', 0)}: "
                f"{option.get('spec') or '-'}"
            )
            if self._state.get("reason"):
                title += f" | {self._state['reason']}"
        manager = native.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(title)
        return self.env.render()

    def close(self) -> None:
        """Close only this display's native environment and figure."""
        self.env.close()
