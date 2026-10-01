"""The movement-provider boundary, independent of ROS and model libraries."""

from __future__ import annotations

from typing import Any

from hrl_tl.robot_demo import config, pose
from hrl_tl.robot_demo.control import action


class Feedback(config.Settings):
    """Provider progress and the decision behind the last returned action."""

    ready: bool = True
    reason: str = ""
    fault: bool = False
    observed_at: float | None = None
    display: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None


class ActionProvider:
    """A provider called once after each measured stop, outside ROS callbacks.

    A trained provider retains its active high-level skill internally and
    returns the final two-index low-level action. None ends the episode.
    """

    def next_action(
        self, measured: pose.Pose2D
    ) -> action.MovementAction | pose.Point2D | None:
        """Return a movement, an explicit waypoint, or end-of-sequence."""
        raise NotImplementedError

    def receive_pose(self, measured: pose.Pose2D) -> None:
        """Accept live measurements; stateless providers need no history."""

    @property
    def arena_id(self) -> str | None:
        """Required simulator layout identity, if this is a Gazebo policy."""
        return None

    def feedback(self) -> Feedback:
        """Return progress without blocking ROS control callbacks."""
        return Feedback()

    def close(self) -> None:
        """Cancel outstanding work and release provider resources."""
