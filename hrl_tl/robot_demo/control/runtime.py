"""One-command-at-a-time orchestration with a nonblocking action handshake."""

from __future__ import annotations

import math
from typing import Any

from hrl_tl.robot_demo import config, pose
from hrl_tl.robot_demo.control import action, motion


class ActionRequest(config.Settings):
    """A unique inference request using a measured pose after stopping."""

    command_id: int
    pose: pose.Pose2D
    requested_at: float


class DemoRuntime:
    """A motion controller and action handshake; no policy runs inside tick."""

    def __init__(self, settings: config.DemoConfig, now: float) -> None:
        self._config: config.DemoConfig = settings
        self._motion: motion.MotionExecutor = motion.MotionExecutor(
            settings.motion, settings.bounds
        )
        self._created_at: float = now
        self._request: ActionRequest | None = None
        self._next_id: int = 1
        self._after_stamp: float = float("-inf")
        self._events: list[dict[str, Any]] = []
        self._last_state: motion.State = self._motion.state

    @property
    def state(self) -> motion.State:
        """Current motion state."""
        return self._motion.state

    @property
    def current_pose(self) -> pose.Pose2D | None:
        """Latest accepted measurement for logging and observation consumers."""
        return self._motion.current_pose

    @property
    def reason(self) -> str:
        """Current terminal/completion reason."""
        return self._motion.reason

    @property
    def finished(self) -> bool:
        """Whether motion is latched off for this episode."""
        return self.state in (motion.State.STOPPED, motion.State.FAULT)

    @property
    def request(self) -> ActionRequest | None:
        """Pending inference request, unchanged until submitted or stopped."""
        return self._request

    def update_pose(self, measured: pose.Pose2D) -> bool:
        """Pass live measurements to the controller, including while moving."""
        return self._motion.update_pose(measured)

    def stop(self, now: float, reason: str, *, fault: bool = False) -> None:
        """Cancel pending inference and latch motion off."""
        self._request = None
        self._motion.stop(now, reason, fault=fault)

    def submit_action(
        self,
        command_id: int,
        command: action.MovementAction | pose.Point2D | None,
        now: float,
        *,
        decision: dict[str, Any] | None = None,
    ) -> bool:
        """Accept only the active request ID; discard late/duplicate replies."""
        request = self._request
        if request is None or command_id != request.command_id or self.finished:
            return False
        if now - request.requested_at > self._config.motion.action_timeout:
            self.stop(now, "action_timeout", fault=True)
            return False
        self._request = None
        if command is None:
            self.stop(now, "actions_exhausted")
            return True
        measured = self._motion.current_pose
        if measured is None or now - measured.received_at > (
            self._config.motion.pose_timeout
        ):
            self.stop(now, "pose_timeout", fault=True)
            return False
        target = (
            action.target_from_action(command, measured, self._config)
            if isinstance(command, action.MovementAction)
            else command
        )
        try:
            self._motion.submit(command_id, target, now)
        except ValueError as error:
            self.stop(now, str(error), fault=True)
            return False
        self._events.append(
            {
                "event": "command",
                "command_id": command_id,
                "time": now,
                "action": command.model_dump(),
                "policy": decision,
                "distance_m": math.hypot(
                    target.x - measured.x, target.y - measured.y
                ),
                "world_heading_deg": math.degrees(
                    math.atan2(target.y - measured.y, target.x - measured.x)
                ),
                "target": target.model_dump(),
                "start_pose": measured.model_dump(),
            }
        )
        return True

    def tick(self, now: float) -> motion.Velocity:
        """Advance feedback and produce a request only after fresh stopping."""
        if self.state == motion.State.WAIT_POSE and now - self._created_at > (
            self._config.motion.startup_timeout
        ):
            self.stop(now, "startup_pose_timeout", fault=True)
        if self._request is not None and now - self._request.requested_at > (
            self._config.motion.action_timeout
        ):
            self.stop(now, "action_timeout", fault=True)
        velocity = self._motion.tick(now)
        result = self._motion.take_result()
        if result is not None:
            self._events.append({"event": "result", **result.model_dump()})
            self._after_stamp = result.final_pose.stamp
        if self.finished:
            self._request = None
        measured = self._motion.current_pose
        if (
            self.state == motion.State.IDLE
            and self._request is None
            and result is None
            and measured is not None
            and measured.stamp > self._after_stamp
        ):
            self._request = ActionRequest(
                command_id=self._next_id, pose=measured, requested_at=now
            )
            self._next_id += 1
            self._events.append(
                {"event": "request", **self._request.model_dump()}
            )
        if self.state != self._last_state:
            self._events.append(
                {
                    "event": "state",
                    "state": self.state.value,
                    "reason": self.reason,
                    "time": now,
                }
            )
            self._last_state = self.state
        return velocity

    def take_events(self) -> list[dict[str, Any]]:
        """Drain structured events for a JSONL log."""
        events, self._events = self._events, []
        return events
