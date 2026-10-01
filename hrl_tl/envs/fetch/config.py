"""Configuration schemas for the Fetch Reach-Avoid environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def compute_triangle_zones(
    center: Sequence[float],
    radius: float = 0.18,
    sphere_z: float = 0.50,
) -> tuple[list[list[float]], list[list[float]]]:
    """Compute 3D coordinates for triangle vertices and edge midpoints.

    Args:
        center: 2D or 3D center position [x, y, z].
        radius: Distance from center to each vertex in meters.
        sphere_z: Height (z-coordinate) for all spheres in meters.

    Returns:
        A tuple of (yellow_zones, red_zones) lists with 3 points each.
    """
    cx, cy = center[0], center[1]
    dx = round(radius * (3.0**0.5) / 2.0, 3)
    dy = round(radius / 2.0, 3)
    r_val = round(radius, 3)

    yellow_zones = [
        [round(cx, 3), round(cy + r_val, 3), sphere_z],
        [round(cx - dx, 3), round(cy - dy, 3), sphere_z],
        [round(cx + dx, 3), round(cy - dy, 3), sphere_z],
    ]
    red_zones = [
        [
            round((yellow_zones[0][0] + yellow_zones[1][0]) / 2.0, 3),
            round((yellow_zones[0][1] + yellow_zones[1][1]) / 2.0, 3),
            sphere_z,
        ],
        [
            round((yellow_zones[0][0] + yellow_zones[2][0]) / 2.0, 3),
            round((yellow_zones[0][1] + yellow_zones[2][1]) / 2.0, 3),
            sphere_z,
        ],
        [
            round((yellow_zones[1][0] + yellow_zones[2][0]) / 2.0, 3),
            round((yellow_zones[1][1] + yellow_zones[2][1]) / 2.0, 3),
            sphere_z,
        ],
    ]
    return yellow_zones, red_zones


class FetchActionConfig(BaseModel):
    """Configuration for the Fetch agent action space."""

    action_mode: Literal["multidiscrete_3d", "continuous"] = "multidiscrete_3d"
    num_bins: int = 5
    max_displacement: float = 0.05

    model_config = ConfigDict(extra="forbid")


class FetchRewardConfig(BaseModel):
    """Configuration for environment reward parameters."""

    step_penalty: float = 0.002

    model_config = ConfigDict(extra="forbid")


class FetchSubtaskConfig(BaseModel):
    """Specification of an ordered subtask in the reach-avoid sequence."""

    goal: Literal["yellow", "white"]
    obstacle: Literal["red"] | None = "red"
    reward: float = 0.0
    penalty: float = -1.0
    goal_absorbing: bool = False
    obstacle_absorbing: bool = True

    model_config = ConfigDict(extra="forbid")


class FetchZoneConfig(BaseModel):
    """Configuration for zone placement and spatial parameters."""

    triangle_radius: float = 0.18
    yellow_zone: list[list[float]] = Field(
        default_factory=lambda: [
            [1.300, 0.930, 0.500],
            [1.144, 0.660, 0.500],
            [1.456, 0.660, 0.500],
        ]
    )
    red_zone: list[list[float]] = Field(
        default_factory=lambda: [
            [1.222, 0.795, 0.500],
            [1.378, 0.795, 0.500],
            [1.300, 0.660, 0.500],
        ]
    )
    white_zone: list[list[float]] = Field(default_factory=list)
    zone_size: float = 0.035
    center: list[float] = Field(default_factory=lambda: [1.300, 0.750, 0.550])
    agent_perturbation: float = 0.02
    render_view: Literal["default", "top_down", "combined"] = "combined"
    top_down_azimuth: float = 90.0

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def populate_zones_from_radius(self) -> FetchZoneConfig:
        """Populate yellow and red zones if triangle_radius is customized."""
        default_yellow = [
            [1.300, 0.930, 0.500],
            [1.144, 0.660, 0.500],
            [1.456, 0.660, 0.500],
        ]
        if self.triangle_radius != 0.18 and self.yellow_zone == default_yellow:
            y_zones, r_zones = compute_triangle_zones(
                self.center, self.triangle_radius
            )
            self.yellow_zone = y_zones
            self.red_zone = r_zones
        return self


class FetchReachAvoidConfig(BaseModel):
    """Top-level configuration schema for FetchReachAvoidEnv."""

    action_config: FetchActionConfig = Field(default_factory=FetchActionConfig)
    reward_config: FetchRewardConfig = Field(default_factory=FetchRewardConfig)
    zone_config: FetchZoneConfig = Field(default_factory=FetchZoneConfig)
    subtask_seq: list[FetchSubtaskConfig] = Field(
        default_factory=lambda: [
            FetchSubtaskConfig(
                goal="yellow",
                obstacle="red",
                reward=0.0,
                penalty=-1.0,
                goal_absorbing=False,
                obstacle_absorbing=True,
            ),
            FetchSubtaskConfig(
                goal="white",
                obstacle="red",
                reward=100.0,
                penalty=-1.0,
                goal_absorbing=True,
                obstacle_absorbing=True,
            ),
        ]
    )

    model_config = ConfigDict(extra="forbid")
