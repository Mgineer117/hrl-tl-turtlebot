"""Open Gazebo space around the unchanged native training world."""

from __future__ import annotations

from hrl_tl.robot_demo import config, pose
from hrl_tl.robot_demo.world import arena


def with_virtual_walls(layout: arena.Arena) -> arena.Arena:
    """Remove physical walls while retaining native physics and observations.

    The outer extent of the native wall cells supplies room for turning around
    the original waypoints. The resulting bounds are for the generated Gazebo
    ground plane; they do not recalibrate the working area of a real robot.

    Args:
        layout: Resolved training world with its original robot configuration.

    Returns:
        A new snapshot with unchanged zones, spawn, policy inputs and dynamics.
    """
    corners = [
        pose.zone_to_world(pose.Point2D(x=x, y=y), layout.robot.frame)
        for min_x, max_x, min_y, max_y in layout.walls
        for x, y in (
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        )
    ]
    bounds = config.Bounds(
        min_x=min(point.x for point in corners),
        max_x=max(point.x for point in corners),
        min_y=min(point.y for point in corners),
        max_y=max(point.y for point in corners),
        margin=layout.robot.bounds.margin,
    )
    values = layout.model_dump()
    values["physical_walls"] = False
    values["robot"]["bounds"] = bounds.model_dump()
    values["robot"]["motion"]["turn_clearance"] = 0.05
    return arena.Arena.model_validate(values)
