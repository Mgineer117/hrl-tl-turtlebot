"""Serial trained-policy inference with independent live robot feedback."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import queue
import threading
from typing import Any, TextIO

from pydantic import dataclasses
from stable_baselines3.common import base_class

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action
from hrl_tl.robot_demo.inference import hierarchy, policy
from hrl_tl.robot_demo.world import arena

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(config={"arbitrary_types_allowed": True})
class _Request:
    measured: pose.Pose2D
    result: concurrent.futures.Future[action.NativeMovementAction | None]


class LearnedPolicyProvider(policy.ActionProvider):
    """A worker resuming the training policy after each measured robot stop."""

    def __init__(
        self,
        layout: arena.Arena,
        upper_model: base_class.BaseAlgorithm,
        wrapper_kwargs: dict[str, Any],
        log: TextIO,
        *,
        max_actions: int | None = None,
        require_arena: bool = True,
    ) -> None:
        if not layout.physical_walls and not require_arena:
            raise ValueError(
                "Virtual-wall replay requires the matching Gazebo arena"
            )
        self._layout: arena.Arena = layout
        self._controller: hierarchy.RobotHierarchy = hierarchy.RobotHierarchy(
            layout, upper_model, wrapper_kwargs, log, max_actions
        )
        self._require_arena: bool = require_arena
        self._stable_since: float | None = None
        self._last_pose: pose.Pose2D | None = None
        self._feedback: policy.Feedback = policy.Feedback(ready=False)
        self._lock: threading.Lock = threading.Lock()
        self._events: queue.Queue[_Request] = queue.Queue(1)
        self._cancel: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._work, name="zone-policy", daemon=True
        )
        self._thread.start()

    @property
    def arena_id(self) -> str | None:
        """Gate simulated motion on the matching rendered layout."""
        return self._layout.identity if self._require_arena else None

    def receive_pose(self, measured: pose.Pose2D) -> None:
        """Track real motion and freshness without advancing the trained env."""
        with self._lock:
            previous = self._last_pose
            if previous is not None and measured.stamp <= previous.stamp:
                return
            speed = yaw_rate = float("inf")
            if previous is not None:
                dt = measured.stamp - previous.stamp
                speed = (
                    math.hypot(measured.x - previous.x, measured.y - previous.y)
                    / dt
                )
                yaw_rate = (
                    abs(pose.angle_difference(measured.yaw, previous.yaw)) / dt
                )
            limits = self._layout.robot.motion
            if (
                speed > limits.stop_linear_speed
                or yaw_rate > limits.stop_angular_speed
            ):
                self._stable_since = None
            elif self._stable_since is None:
                self._stable_since = measured.received_at
            self._last_pose = measured
            ready = self._stable_since is not None and (
                measured.received_at - self._stable_since
                >= limits.settle_duration
            )
            self._feedback = self._feedback.model_copy(
                update={
                    "ready": ready,
                    "observed_at": measured.received_at,
                }
            )

    def feedback(self) -> policy.Feedback:
        """Read an immutable snapshot without accessing native env state."""
        with self._lock:
            return self._feedback

    def next_action(
        self, measured: pose.Pose2D
    ) -> action.NativeMovementAction | None:
        """Request one primitive after the previous robot movement completed."""
        if self._cancel.is_set() or self.feedback().reason:
            return None
        self.receive_pose(measured)
        result: concurrent.futures.Future[
            action.NativeMovementAction | None
        ] = concurrent.futures.Future()
        self._events.put_nowait(_Request(measured, result))
        return result.result(timeout=self._layout.robot.motion.action_timeout)

    def close(self) -> None:
        """Cancel queued work and bound shutdown even if inference stalls."""
        self._cancel.set()
        self._thread.join(timeout=1)

    def _work(self) -> None:
        pending: _Request | None = None
        try:
            while not self._cancel.is_set():
                try:
                    pending = self._events.get(timeout=0.1)
                except queue.Empty:
                    continue
                selected = self._controller.next_action(pending.measured)
                with self._lock:
                    self._feedback = self._feedback.model_copy(
                        update={
                            "display": self._controller.display,
                            "decision": self._controller.decision,
                            "reason": self._controller.reason,
                        }
                    )
                pending.result.set_result(
                    None if self._cancel.is_set() else selected
                )
                pending = None
        except Exception:
            # This is the top-level boundary of the inference worker.
            _LOGGER.exception("Hierarchical policy worker failed")
            with self._lock:
                self._feedback = self._feedback.model_copy(
                    update={
                        "reason": "policy_error",
                        "fault": True,
                    }
                )
        finally:
            if pending is not None and not pending.result.done():
                pending.result.set_result(None)
            self._controller.close()
            while not self._events.empty():
                self._events.get_nowait().result.set_result(None)
