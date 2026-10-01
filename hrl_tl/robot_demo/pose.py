"""Planar pose and coordinate conversion independent of ROS."""

from __future__ import annotations

import math

from hrl_tl.robot_demo import config


class Point2D(config.Settings):
    """A finite planar position; callers define world metres or Zone units."""

    x: float
    y: float


class Pose2D(Point2D):
    """Measured pose with yaw in radians and two separate time bases.

    Attributes:
        yaw: Counterclockwise heading from the world's positive x axis.
        stamp: Source timestamp in seconds, used for ordering and velocity.
        received_at: Local monotonic reception time, used for the watchdog.
    """

    yaw: float
    stamp: float
    received_at: float


def angle_difference(target: float, current: float) -> float:
    """Return the signed shortest angular displacement in radians."""
    return math.atan2(math.sin(target - current), math.cos(target - current))


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    """Normalize a nonzero quaternion and extract its planar yaw.

    Raises:
        ValueError: The quaternion is zero or contains nonfinite values.
    """
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid pose quaternion")
    qx, qy, qz, qw = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def mocap_to_world(measured: Pose2D, settings: config.RosConfig) -> Pose2D:
    """Map the QTM marker to the robot reference point in arena world metres."""
    x, y = measured.x, measured.y
    c, s = math.cos(settings.mocap_rotation_rad), math.sin(settings.mocap_rotation_rad)
    yaw = measured.yaw + settings.mocap_rotation_rad + settings.heading_offset_rad
    cy, sy = math.cos(yaw), math.sin(yaw)
    return Pose2D(
        x=c * x - s * y + settings.mocap_offset_x_m
        - cy * settings.marker_offset_x_m + sy * settings.marker_offset_y_m,
        y=s * x + c * y + settings.mocap_offset_y_m
        - sy * settings.marker_offset_x_m - cy * settings.marker_offset_y_m,
        yaw=yaw,
        stamp=measured.stamp,
        received_at=measured.received_at,
    )


def world_to_zone(point: Point2D, frame: config.FrameConfig) -> Point2D:
    """Apply the configured world-to-Zone rotation, origin, and scale."""
    x, y = point.x - frame.origin_x_m, point.y - frame.origin_y_m
    c, s = math.cos(frame.rotation_rad), math.sin(frame.rotation_rad)
    return Point2D(
        x=frame.sim_units_per_meter * (c * x - s * y),
        y=frame.sim_units_per_meter * (s * x + c * y),
    )


def zone_to_world(point: Point2D, frame: config.FrameConfig) -> Point2D:
    """Invert the same transform used by observations and movement actions."""
    c, s = math.cos(frame.rotation_rad), math.sin(frame.rotation_rad)
    scale = frame.sim_units_per_meter
    return Point2D(
        x=frame.origin_x_m + (c * point.x + s * point.y) / scale,
        y=frame.origin_y_m + (-s * point.x + c * point.y) / scale,
    )
