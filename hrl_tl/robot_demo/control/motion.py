"""Feedback-driven turn, drive, and settle state machine without ROS."""

from __future__ import annotations

import enum
import math
from typing import Literal

from hrl_tl.robot_demo import config, pose


class State(enum.StrEnum):
    """Motion executor states."""

    WAIT_POSE = "wait_pose"
    IDLE = "idle"
    ROTATING = "rotating"
    REPOSITIONING = "repositioning"
    MOVING = "moving"
    SETTLING = "settling"
    HOLDING = "holding"
    STOPPED = "stopped"
    FAULT = "fault"


class Velocity(config.Settings):
    """Body-forward speed in m/s and counterclockwise yaw rate in rad/s."""

    linear: float = 0.0
    angular: float = 0.0


class MotionResult(config.Settings):
    """Exactly one terminal result for an accepted command."""

    command_id: int
    outcome: Literal["target_reached", "stopped", "fault"]
    reason: str
    start_pose: pose.Pose2D
    final_pose: pose.Pose2D
    target: pose.Point2D
    started_at: float
    finished_at: float


class MotionExecutor:
    """A single-command controller driven by fresh measured poses."""

    def __init__(
        self, settings: config.MotionConfig, bounds: config.Bounds
    ) -> None:
        self._config: config.MotionConfig = settings
        self._bounds: config.Bounds = bounds
        self._state: State = State.WAIT_POSE
        self._pose: pose.Pose2D | None = None
        self._target: pose.Point2D | None = None
        self._start: pose.Pose2D | None = None
        self._command_id: int = 0
        self._started_at: float = 0.0
        self._stable_since: float | None = None
        self._speed: float = math.inf
        self._yaw_rate: float = math.inf
        self._result: MotionResult | None = None
        self._reason: str = ""
        self._retreat_sign: int = 1
        self._retreat_heading: float = 0.0
        self._clearance_prepared: bool = False

    @property
    def state(self) -> State:
        """Current state; terminal states require a new executor to restart."""
        return self._state

    @property
    def current_pose(self) -> pose.Pose2D | None:
        """Latest accepted source-ordered pose."""
        return self._pose

    @property
    def reason(self) -> str:
        """Most recent completion or stop reason."""
        return self._reason

    def update_pose(self, measured: pose.Pose2D) -> bool:
        """Accept only advancing source times; estimate actual motion.

        Returns:
            Whether the pose advanced. Replays cannot refresh the watchdog.
        """
        previous = self._pose
        if previous is not None and measured.stamp <= previous.stamp:
            return False
        if previous is not None:
            dt = measured.stamp - previous.stamp
            self._speed = (
                math.hypot(measured.x - previous.x, measured.y - previous.y)
                / dt
            )
            self._yaw_rate = abs(
                pose.angle_difference(measured.yaw, previous.yaw) / dt
            )
        self._pose = measured
        if not self._bounds.contains(
            measured.x, measured.y, self._config.wall_stop_margin
        ):
            self.stop(measured.received_at, "wall_clearance", fault=True)
        elif self._state == State.WAIT_POSE:
            self._state = State.IDLE
        return True

    def submit(self, command_id: int, target: pose.Point2D, now: float) -> bool:
        """Accept one fixed target; ignore previously issued command IDs.

        Raises:
            ValueError: A new command arrives while not ready or is unsafe.
        """
        if command_id <= self._command_id:
            return False
        if self._state != State.IDLE or self._pose is None:
            raise ValueError("Controller is not ready for a new command")
        if not self._bounds.contains(
            target.x, target.y, self._config.wall_stop_margin
        ):
            raise ValueError("Target is inside the wall stop buffer")
        self._command_id = command_id
        self._target, self._start = target, self._pose
        self._started_at = now
        self._clearance_prepared = False
        self._stable_since = None
        self._state = State.ROTATING
        self._reason = ""
        return True

    def stop(self, now: float, reason: str, *, fault: bool = False) -> None:
        """Latch a stop/fault, producing one result if a command is active."""
        if self._state in (State.STOPPED, State.FAULT):
            return
        self._finish(now, "fault" if fault else "stopped", reason)
        self._state = State.FAULT if fault else State.STOPPED

    def take_result(self) -> MotionResult | None:
        """Consume a completion once, preventing duplicate step accounting."""
        result, self._result = self._result, None
        return result

    def tick(self, now: float) -> Velocity:
        """Compute a bounded velocity; stop on missing poses or timeout."""
        if self._state in (State.STOPPED, State.FAULT) or self._pose is None:
            return Velocity()
        if now - self._pose.received_at > self._config.pose_timeout:
            self._finish(now, "stopped", "pose_timeout")
            self._state = State.IDLE
            return Velocity()
        if self._target is None:
            return Velocity()
        distance = math.hypot(
            self._target.x - self._pose.x, self._target.y - self._pose.y
        )
        if now - self._started_at >= self._config.motion_timeout:
            if self._state == State.HOLDING and distance <= self._config.position_tolerance:
                self._finish(now, "target_reached", "target_reached")
                self._state = State.IDLE
            else:
                self._finish(now, "stopped", "motion_timeout")
                self._state = State.IDLE
            return Velocity()
        if self._state == State.HOLDING:
            if distance > self._config.position_tolerance:
                self._state = State.ROTATING
            return Velocity()
        if self._state == State.REPOSITIONING:
            return self._reposition(distance)
        if self._state == State.SETTLING:
            return self._settle(now, distance)
        if distance <= self._config.position_tolerance:
            self._state = State.SETTLING
            self._stable_since = None
            return Velocity()
        angle = pose.angle_difference(
            math.atan2(
                self._target.y - self._pose.y, self._target.x - self._pose.x
            ),
            self._pose.yaw,
        )
        if self._state == State.ROTATING:
            if (
                not self._clearance_prepared
                and distance < self._config.turn_clearance
                and abs(angle) > self._config.reorient_threshold
            ):
                self._retreat_sign = -1 if math.cos(angle) >= 0 else 1
                self._retreat_heading = self._pose.yaw
                clearance = self._config.turn_clearance
                if not self._bounds.contains(
                    self._pose.x
                    + self._retreat_sign
                    * clearance
                    * math.cos(self._retreat_heading),
                    self._pose.y
                    + self._retreat_sign
                    * clearance
                    * math.sin(self._retreat_heading),
                    self._config.wall_stop_margin,
                ):
                    self.stop(now, "insufficient_turning_space", fault=True)
                    return Velocity()
                self._clearance_prepared = True
                self._state = State.REPOSITIONING
                return self._reposition(distance)
            if abs(angle) <= self._config.heading_tolerance:
                self._state = State.MOVING
                return Velocity()
            return Velocity(angular=self._angular_speed(angle))
        if abs(angle) > self._config.reorient_threshold:
            self._clearance_prepared = False
            self._state = State.ROTATING
            return Velocity()
        return Velocity(
            linear=min(
                self._config.max_linear_speed,
                max(self._config.min_linear_speed, self._config.linear_gain * distance),
            ),
            angular=self._angular_speed(angle),
        )

    def _reposition(self, distance: float) -> Velocity:
        """Make room to turn without changing the policy's final waypoint."""
        if distance >= self._config.turn_clearance:
            self._state = State.ROTATING
            return Velocity()
        return Velocity(
            linear=self._retreat_sign
            * min(
                self._config.max_linear_speed,
                self._config.linear_gain * self._config.turn_clearance,
            ),
            angular=self._angular_speed(
                pose.angle_difference(self._retreat_heading, self._pose.yaw)
            ),
        )

    def _angular_speed(self, error: float) -> float:
        limit = self._config.max_angular_speed
        return max(-limit, min(limit, self._config.angular_gain * error))

    def _settle(self, now: float, distance: float) -> Velocity:
        if distance > self._config.position_tolerance:
            self._clearance_prepared = False
            self._state = State.ROTATING
            self._stable_since = None
            return Velocity()
        if (
            self._speed > self._config.stop_linear_speed
            or self._yaw_rate > self._config.stop_angular_speed
        ):
            self._stable_since = None
            return Velocity()
        if self._stable_since is None:
            self._stable_since = now
        if (
            self._pose is not None
            and self._pose.received_at > self._stable_since
            and now - self._stable_since >= self._config.settle_duration
        ):
            self._state = State.HOLDING
        return Velocity()

    def _finish(
        self,
        now: float,
        outcome: Literal["target_reached", "stopped", "fault"],
        reason: str,
    ) -> None:
        self._reason = reason
        if (
            self._target is not None
            and self._start is not None
            and self._pose is not None
        ):
            self._result = MotionResult(
                command_id=self._command_id,
                outcome=outcome,
                reason=reason,
                start_pose=self._start,
                final_pose=self._pose,
                target=self._target,
                started_at=self._started_at,
                finished_at=now,
            )
        self._target = None
