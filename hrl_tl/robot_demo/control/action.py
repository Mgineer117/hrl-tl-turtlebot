"""Conversion of the final Zone movement action to a fixed short waypoint."""

from __future__ import annotations

import math

import pydantic

from hrl_tl.robot_demo import config, pose


class MovementAction(config.Settings):
    """Final low-level action, not the five-element high-level PPO output."""

    direction_index: int = pydantic.Field(ge=0, lt=8, strict=True)
    magnitude_index: int = pydantic.Field(ge=0, lt=5, strict=True)


class NativeMovementAction(MovementAction):
    """A primitive action with its native and physically reachable endpoints."""

    target: pose.Point2D
    native_target: pose.Point2D


def target_from_action(
    movement: MovementAction, start: pose.Pose2D, settings: config.DemoConfig
) -> pose.Point2D:
    """Freeze a world waypoint relative to the measured action-start pose.

    Learned commands carry an endpoint computed from the discrete action.
    Sequence-test commands use the configured distance mapping instead.
    Neither index is a physical velocity command.
    """
    if isinstance(movement, NativeMovementAction):
        return movement.target
    heading = (
        movement.direction_index * math.tau / 8 - settings.frame.rotation_rad
    )
    distance = (
        settings.step_lengths_sim[movement.magnitude_index]
        / settings.frame.sim_units_per_meter
    )
    return pose.Point2D(
        x=start.x + distance * math.cos(heading),
        y=start.y + distance * math.sin(heading),
    )
