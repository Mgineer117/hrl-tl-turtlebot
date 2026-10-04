"""Validated settings for the ROS and ROS-independent demo components."""

from __future__ import annotations

import itertools
from typing import Literal, Self

import pydantic


class Settings(pydantic.BaseModel):
    """Immutable settings rejecting unknown fields and nonfinite values."""

    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False
    )


class FrameConfig(Settings):
    """World-to-Zone transform: scale * R(rotation) * (world - origin)."""

    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    rotation_rad: float = 0.0
    sim_units_per_meter: float = pydantic.Field(default=1.0, gt=0)


class Bounds(Settings):
    """World-frame operating rectangle with a robot-footprint inset, in m."""

    min_x: float = -2.0
    max_x: float = 2.0
    min_y: float = -2.0
    max_y: float = 2.0
    margin: float = pydantic.Field(default=0.12, ge=0)

    @pydantic.model_validator(mode="after")
    def validate_size(self) -> Self:
        """Require usable space after applying the footprint inset."""
        if min(self.max_x - self.min_x, self.max_y - self.min_y) <= (
            2 * self.margin
        ):
            raise ValueError("Bounds must contain space beyond the margin")
        return self

    def contains(self, x: float, y: float, extra_margin: float = 0.0) -> bool:
        """Return whether a world position lies within the inset rectangle."""
        return (
            self.min_x + self.margin + extra_margin
            <= x
            <= self.max_x - self.margin - extra_margin
            and self.min_y + self.margin + extra_margin
            <= y
            <= self.max_y - self.margin - extra_margin
        )


class MotionConfig(Settings):
    """Motion limits in m/s, rad/s, m, rad, and seconds."""

    max_linear_speed: float = pydantic.Field(default=0.06, gt=0, le=0.21)
    min_linear_speed: float = pydantic.Field(default=0.0, ge=0)
    max_angular_speed: float = pydantic.Field(default=0.6, gt=0, le=0.85)
    linear_gain: float = pydantic.Field(default=1.0, gt=0)
    angular_gain: float = pydantic.Field(default=2.0, gt=0)
    position_tolerance: float = pydantic.Field(default=0.01, gt=0)
    heading_tolerance: float = pydantic.Field(default=0.06, gt=0)
    reorient_threshold: float = pydantic.Field(default=0.25, gt=0)
    turn_clearance: float = pydantic.Field(default=0.0, ge=0)
    stop_linear_speed: float = pydantic.Field(default=0.012, gt=0)
    stop_angular_speed: float = pydantic.Field(default=0.04, gt=0)
    settle_duration: float = pydantic.Field(default=0.3, gt=0)
    motion_timeout: float = pydantic.Field(default=5.0, gt=0)
    wall_stop_margin: float = pydantic.Field(default=0.05, ge=0)
    pose_timeout: float = pydantic.Field(default=0.5, gt=0)
    startup_timeout: float = pydantic.Field(default=15.0, gt=0)
    action_timeout: float = pydantic.Field(default=5.0, gt=0)

    @pydantic.model_validator(mode="after")
    def validate_hysteresis(self) -> Self:
        """Require distinct turn/drive thresholds."""
        if self.min_linear_speed > self.max_linear_speed:
            raise ValueError("min_linear_speed must not exceed max_linear_speed")
        if self.heading_tolerance >= self.reorient_threshold:
            raise ValueError("heading_tolerance must be < reorient_threshold")
        if self.wall_stop_margin < self.max_linear_speed * self.pose_timeout:
            raise ValueError(
                "wall_stop_margin must cover motion during pose_timeout"
            )
        return self


class RosConfig(Settings):
    """Topic names and the explicitly selected pose/velocity wire formats."""

    pose_topic: str = "/qualysis/tb3_1"
    pose_encoding: Literal["legacy_euler", "quaternion"] = "legacy_euler"
    pose_frame: str = "mocap"
    heading_offset_rad: float = 0.0
    mocap_offset_x_m: float = 0.0
    mocap_offset_y_m: float = 0.0
    marker_offset_x_m: float = 0.0
    marker_offset_y_m: float = 0.0
    mocap_rotation_rad: float = 0.0
    align_first_pose_to_start: bool = False
    cmd_vel_topic: str = "/robot_demo/cmd_vel"
    cmd_vel_type: Literal["twist", "twist_stamped"] = "twist"
    stop_topic: str = "/robot_demo/stop"
    control_hz: float = pydantic.Field(default=30.0, gt=0, le=200)


class DemoConfig(Settings):
    """Single-robot runtime configuration and retained arena schema."""

    motion: MotionConfig = pydantic.Field(default_factory=MotionConfig)
    frame: FrameConfig = pydantic.Field(default_factory=FrameConfig)
    bounds: Bounds = pydantic.Field(default_factory=Bounds)
    ros: RosConfig = pydantic.Field(default_factory=RosConfig)
    step_lengths_sim: tuple[float, float, float, float, float] = (
        0.05,
        0.09,
        0.13,
        0.17,
        0.21,
    )

    # Existing arena.json files include these values in their identity hash.
    actions: tuple[tuple[pydantic.StrictInt, pydantic.StrictInt], ...] = (
        (0, 3),
        (2, 3),
        (4, 3),
        (6, 3),
    )

    @pydantic.model_validator(mode="after")
    def validate_actions(self) -> Self:
        """Validate retained actions and resolvable movement distances."""
        if min(
            self.bounds.max_x - self.bounds.min_x,
            self.bounds.max_y - self.bounds.min_y,
        ) <= 2 * (
            self.bounds.margin
            + self.motion.wall_stop_margin
            + self.motion.position_tolerance
        ):
            raise ValueError("Wall stop margin leaves no reachable area")
        distances = self.step_lengths_sim
        if any(a >= b for a, b in itertools.pairwise(distances)):
            raise ValueError("step_lengths_sim must be strictly increasing")
        if distances[0] / self.frame.sim_units_per_meter <= (
            self.motion.position_tolerance
        ):
            raise ValueError("Shortest step must exceed the tolerance")
        if any(not (0 <= d < 8 and 0 <= m < 5) for d, m in self.actions):
            raise ValueError(
                "Actions must contain direction 0..7, strength 0..4"
            )
        return self
