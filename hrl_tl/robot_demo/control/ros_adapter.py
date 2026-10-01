"""ROS transport and background inference around the ROS-free runtime."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import pathlib
import time
from typing import TextIO

import rclpy
from geometry_msgs import msg as geometry
from rclpy import node, publisher, qos, signals, subscription, timer
from sensor_msgs import msg as sensor
from std_msgs import msg as standard

from hrl_tl.robot_demo import config, pose, recording
from hrl_tl.robot_demo.control import action, motion, runtime
from hrl_tl.robot_demo.inference import policy

_LOGGER: logging.Logger = logging.getLogger(__name__)


def decode_pose(
    message: geometry.PoseStamped,
    settings: config.RosConfig,
    received_at: float,
) -> pose.Pose2D:
    """Decode the selected mocap format and apply marker-to-body yaw offset.

    Raises:
        ValueError: The frame or numeric pose is invalid.
    """
    if settings.pose_frame and message.header.frame_id != settings.pose_frame:
        raise ValueError(f"Expected pose frame {settings.pose_frame!r}")
    rotation = message.pose.orientation
    yaw = (
        rotation.z
        if settings.pose_encoding == "legacy_euler"
        else pose.quaternion_yaw(rotation.x, rotation.y, rotation.z, rotation.w)
    )
    raw = pose.Pose2D(
        x=message.pose.position.x,
        y=message.pose.position.y,
        yaw=yaw,
        stamp=message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
        received_at=received_at,
    )
    return pose.mocap_to_world(raw, settings)


class RobotNode(node.Node):
    """A feedback node whose timer never waits for policy inference."""

    def __init__(
        self,
        settings: config.DemoConfig,
        provider: policy.ActionProvider,
        log: TextIO,
        *,
        episode_recorder: recording.EpisodeRecorder | None = None,
        gazebo_image_topic: str | None = None,
    ) -> None:
        super().__init__("zone_robot_demo")
        self._config: config.DemoConfig = settings
        self._provider: policy.ActionProvider = provider
        self._log: TextIO = log
        self._recorder = episode_recorder
        self._runtime: runtime.DemoRuntime = runtime.DemoRuntime(
            settings, time.monotonic()
        )
        self._pool: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
        )
        self._future: (
            concurrent.futures.Future[
                action.MovementAction | pose.Point2D | None
            ]
            | None
        ) = None
        self._request_id: int = 0
        self._last_sample: float = 0.0
        self._arena_ready: bool = provider.arena_id is None
        self._last_display: str = ""
        retained = qos.QoSProfile(
            depth=1,
            durability=qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=qos.ReliabilityPolicy.RELIABLE,
        )
        self._display_publisher = self.create_publisher(
            standard.String, "/robot_demo/zone_state", retained
        )
        self._arena_subscription = self.create_subscription(
            standard.String, "/robot_demo/arena", self._on_arena, retained
        )
        self._publisher: publisher.Publisher = self.create_publisher(
            geometry.Twist
            if settings.ros.cmd_vel_type == "twist"
            else geometry.TwistStamped,
            settings.ros.cmd_vel_topic,
            1,
        )
        self._pose_subscription: subscription.Subscription = (
            self.create_subscription(
                geometry.PoseStamped,
                settings.ros.pose_topic,
                self._on_pose,
                qos.qos_profile_sensor_data,
            )
        )
        self._image_subscription: subscription.Subscription | None = None
        if episode_recorder is not None and gazebo_image_topic is not None:
            self._image_subscription = self.create_subscription(
                sensor.Image,
                gazebo_image_topic,
                episode_recorder.capture_gazebo,
                qos.qos_profile_sensor_data,
            )
        self._stop_subscription: subscription.Subscription = (
            self.create_subscription(
                standard.Bool, settings.ros.stop_topic, self._on_stop, 1
            )
        )
        self._timer: timer.Timer = self.create_timer(
            1 / settings.ros.control_hz, self._on_timer
        )
        self._log.write(
            json.dumps(
                {
                    "event": "config",
                    "settings": settings.model_dump(),
                }
            )
            + "\n"
        )
        self._log.flush()

    @property
    def finished(self) -> bool:
        """Whether the current episode has stopped or faulted."""
        return self._runtime.finished

    @property
    def reason(self) -> str:
        """Final episode reason for CLI exit status and logs."""
        return self._runtime.reason

    def publish_velocity(self, velocity: motion.Velocity) -> None:
        """Publish either Twist or TwistStamped as configured."""
        twist = geometry.Twist()
        twist.linear.x, twist.angular.z = velocity.linear, velocity.angular
        if self._config.ros.cmd_vel_type == "twist_stamped":
            stamped = geometry.TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = "base_link"
            stamped.twist = twist
            self._publisher.publish(stamped)
        else:
            self._publisher.publish(twist)

    def stop(self, reason: str) -> None:
        """Latch a stop and publish zero immediately."""
        self._runtime.stop(time.monotonic(), reason)
        self._on_timer()

    def close(self) -> None:
        """Cancel unstarted inference and close transport resources."""
        measured = self._runtime.current_pose
        if self._recorder is not None and measured is not None:
            self._recorder.capture_board(
                measured,
                self._provider.feedback().display,
                time.monotonic(),
                force=True,
            )
        self._provider.close()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.destroy_node()

    def _on_pose(self, message: geometry.PoseStamped) -> None:
        try:
            measured = decode_pose(message, self._config.ros, time.monotonic())
            if self._runtime.update_pose(measured):
                self._provider.receive_pose(measured)
        except ValueError as error:
            self._runtime.stop(
                time.monotonic(), f"invalid_pose: {error}", fault=True
            )
            self.publish_velocity(motion.Velocity())

    def _on_arena(self, message: standard.String) -> None:
        expected = self._provider.arena_id
        if expected is not None:
            self._arena_ready = message.data == expected
            if not self._arena_ready:
                self._runtime.stop(
                    time.monotonic(), "arena_mismatch", fault=True
                )

    def _on_stop(self, message: standard.Bool) -> None:
        if message.data:
            self.stop("manual_stop")

    def _on_timer(self) -> None:
        now = time.monotonic()
        self._advance_policy(now)
        # Report the accepted command and its policy before any cmd_vel for it.
        self._write_events()
        velocity = self._runtime.tick(now)
        self.publish_velocity(velocity)
        if now - self._last_sample >= 0.1:
            measured = self._runtime.current_pose
            self._log.write(
                json.dumps(
                    {
                        "event": "control",
                        "time": now,
                        "state": self._runtime.state.value,
                        "velocity": velocity.model_dump(),
                        "pose": measured.model_dump()
                        if measured is not None
                        else None,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            self._last_sample = now
        self._write_events()
        self._log.flush()
        measured = self._runtime.current_pose
        if self._recorder is not None and measured is not None:
            self._recorder.capture_board(
                measured,
                self._provider.feedback().display,
                now,
            )

    def _write_events(self) -> None:
        for event in self._runtime.take_events():
            self._log.write(json.dumps(event, allow_nan=False) + "\n")
            if event["event"] in ("command", "state", "result"):
                self.get_logger().info(json.dumps(event, allow_nan=False))

    def _advance_policy(self, now: float) -> None:
        if self.count_publishers(self._config.ros.pose_topic) > 1:
            self._runtime.stop(now, "multiple_pose_publishers", fault=True)
            return
        feedback = self._provider.feedback()
        if feedback.display is not None:
            display = json.dumps(feedback.display, sort_keys=True)
            if display != self._last_display:
                self._display_publisher.publish(standard.String(data=display))
                self._last_display = display
        if feedback.reason:
            self._runtime.stop(now, feedback.reason, fault=feedback.fault)
            return
        if (
            feedback.observed_at is not None
            and now - feedback.observed_at > self._config.motion.pose_timeout
            and self._runtime.state
            in (
                motion.State.ROTATING,
                motion.State.REPOSITIONING,
                motion.State.MOVING,
                motion.State.SETTLING,
            )
        ):
            self._runtime.stop(now, "observation_timeout", fault=True)
            return
        if self._future is not None and self._future.done():
            completed, self._future = self._future, None
            try:
                command = completed.result()
                feedback = self._provider.feedback()
                if feedback.reason:
                    self._runtime.stop(
                        now, feedback.reason, fault=feedback.fault
                    )
                else:
                    self._runtime.submit_action(
                        self._request_id,
                        command,
                        now,
                        decision=feedback.decision,
                    )
            except Exception:
                # Inference is an external worker/process boundary.
                _LOGGER.exception("Policy failed")
                self._runtime.stop(now, "policy_error", fault=True)
        request = self._runtime.request
        if (
            request is not None
            and self._future is None
            and feedback.ready
            and self._arena_ready
        ):
            self._request_id = request.command_id
            self._future = self._pool.submit(
                self._provider.next_action, request.pose
            )


def run(
    settings: config.DemoConfig,
    provider: policy.ActionProvider,
    log_path: pathlib.Path,
    *,
    episode_recorder: recording.EpisodeRecorder | None = None,
    gazebo_image_topic: str | None = None,
) -> int:
    """Run an episode, flush stop commands, and return a process exit status.

    A unique log file is required to preserve previous runs. Inference is
    outside callbacks; pose reception and stop output continue while it runs.

    Args:
        settings: Robot motion and ROS settings.
        provider: Source of one action after each completed movement.
        log_path: New path for structured motion events.
        episode_recorder: Optional synchronized GIF sample collector.
        gazebo_image_topic: Camera topic subscribed to while recording.

    Returns:
        Zero for an expected episode ending; one for a fault or interruption.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        rclpy.init(
            args=[], signal_handler_options=signals.SignalHandlerOptions.NO
        )
        robot: RobotNode | None = None
        try:
            robot = RobotNode(
                settings,
                provider,
                log,
                episode_recorder=episode_recorder,
                gazebo_image_topic=gazebo_image_topic,
            )
            while rclpy.ok() and not robot.finished:
                rclpy.spin_once(robot, timeout_sec=0.1)
        except KeyboardInterrupt:
            if robot is not None:
                robot.stop("interrupted")
        finally:
            if robot is not None:
                for _ in range(5):
                    robot.publish_velocity(motion.Velocity())
                    time.sleep(0.04)
                log.write(
                    json.dumps(
                        {
                            "event": "shutdown",
                            "reason": robot.reason,
                            "velocity": motion.Velocity().model_dump(),
                        }
                    )
                    + "\n"
                )
                log.flush()
                robot.close()
            rclpy.try_shutdown()
    return (
        0
        if robot is not None
        and robot.reason
        in ("actions_exhausted", "task_success", "episode_limit")
        else 1
    )
