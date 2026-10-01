"""Run the supplied Zone hierarchy from Qualisys poses on one TurtleBot3."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import socket
import time
from pathlib import Path

import torch
import yaml
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action
from hrl_tl.robot_demo.inference.hierarchy import RobotHierarchy
from hrl_tl.robot_demo.inference.learned_policy import LearnedPolicyProvider
from hrl_tl.robot_demo.inference.manual import SingleActionProvider
from hrl_tl.robot_demo.world.arena import Arena, load

ROOT = Path(__file__).resolve().parent
DEFAULT_ROBOT_IP = "192.168.0.77"


def interpret_manual_action(
    layout: Arena, angle_deg: float, distance_m: float
) -> tuple[action.MovementAction, float, float]:
    """Map one supported Zone-frame angle and stride to discrete action bins."""
    if not math.isfinite(angle_deg) or not math.isfinite(distance_m):
        raise ValueError("angle and distance must be finite numbers")
    direction = angle_deg % 360.0
    direction_index = round(direction / 45.0) % 8
    interpreted_angle = direction_index * 45.0
    angle_error = abs((direction - interpreted_angle + 180.0) % 360.0 - 180.0)
    if angle_error > 1e-6:
        raise ValueError("angle must be a multiple of 45 degrees")
    available_steps = [
        value / layout.robot.frame.sim_units_per_meter
        for value in layout.robot.step_lengths_sim
    ]
    magnitude_index = min(
        range(len(available_steps)),
        key=lambda index: abs(available_steps[index] - distance_m),
    )
    if abs(available_steps[magnitude_index] - distance_m) > 0.001:
        choices = ", ".join(f"{step:.5f}" for step in available_steps)
        raise ValueError(
            f"distance must match a configured step within 1 mm; available: {choices}"
        )
    return (
        action.MovementAction(
            direction_index=direction_index,
            magnitude_index=magnitude_index,
        ),
        interpreted_angle,
        available_steps[magnitude_index],
    )


def preflight(robot_ip: str, layout) -> None:
    """Require a corrected live pose and a ROS command subscriber before motion."""
    try:
        with socket.create_connection((robot_ip, 22), timeout=2):
            pass
    except OSError as error:
        raise RuntimeError(f"Robot {robot_ip}: SSH port 22 is unreachable: {error}") from error

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data
    from hrl_tl.robot_demo.control.ros_adapter import decode_pose

    settings = layout.robot.ros
    pose_topic, cmd_topic, cmd_type = (
        settings.pose_topic, settings.cmd_vel_topic, settings.cmd_vel_type
    )
    rclpy.init(args=[])
    node = rclpy.create_node("zone_robot_preflight")
    received: list[PoseStamped] = []
    subscription = node.create_subscription(
        PoseStamped, pose_topic, received.append, qos_profile_sensor_data
    )
    try:
        deadline = time.monotonic() + 10.0
        topics = dict(node.get_topic_names_and_types())
        expected = (
            "geometry_msgs/msg/Twist" if cmd_type == "twist"
            else "geometry_msgs/msg/TwistStamped"
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            topics = dict(node.get_topic_names_and_types())
            if (
                pose_topic in topics
                and "geometry_msgs/msg/PoseStamped" in topics[pose_topic]
                and received
                and expected in topics.get(cmd_topic, [])
                and node.count_subscribers(cmd_topic) > 0
            ):
                try:
                    corrected = decode_pose(received[-1], settings, time.monotonic())
                except ValueError as error:
                    raise RuntimeError(f"Invalid mocap pose: {error}") from error
                if not layout.robot.bounds.contains(
                    corrected.x, corrected.y, layout.robot.motion.wall_stop_margin
                ):
                    raise RuntimeError(
                        f"Corrected mocap pose ({corrected.x:.3f}, {corrected.y:.3f}) "
                        "is outside the calibrated wall buffer"
                    )
                zone = pose.world_to_zone(corrected, layout.robot.frame)
                print(
                    f"Preflight OK: {robot_ip}:22, {pose_topic}, {cmd_topic} ({expected})\n"
                    f"Raw QTM: ({received[-1].pose.position.x:.3f}, "
                    f"{received[-1].pose.position.y:.3f}) m; "
                    f"corrected world: ({corrected.x:.3f}, {corrected.y:.3f}) m; "
                    f"Zone: ({zone.x:.3f}, {zone.y:.3f})"
                )
                return
        pose_types = topics.get(pose_topic, [])
        cmd_types = topics.get(cmd_topic, [])
        pose_publishers = node.count_publishers(pose_topic)
        cmd_subscribers = node.count_subscribers(cmd_topic)
        problems = []
        if "geometry_msgs/msg/PoseStamped" not in pose_types:
            problems.append(
                f"{pose_topic} type missing (observed {pose_types or 'no topic'})"
            )
        elif not received:
            problems.append(
                f"no PoseStamped message arrived on {pose_topic} during the 10 s wait"
            )
        if expected not in cmd_types:
            problems.append(
                f"{cmd_topic} type must be {expected} (observed {cmd_types or 'no topic'})"
            )
        elif cmd_subscribers == 0:
            problems.append(f"{cmd_topic} has no subscribers")
        raise RuntimeError(
            "ROS preflight failed: "
            + "; ".join(problems)
            + f". Diagnostics: pose publishers={pose_publishers}, "
            f"pose messages received={len(received)}, "
            f"{cmd_topic} subscribers={cmd_subscribers}, visible topics={topics}"
        )
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "preflight", "move", "run"))
    parser.add_argument("--arena", type=Path, default=Path("configs/arena.json"))
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument(
        "--angle-deg",
        type=float,
        help="one Zone-frame direction (0, 45, ..., 315 degrees) for move mode",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        help="one configured primitive step distance in metres for move mode",
    )
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.mode == "move":
        if args.angle_deg is None or args.distance_m is None:
            parser.error("move mode requires both --angle-deg and --distance-m")
        if not math.isfinite(args.angle_deg) or not math.isfinite(args.distance_m):
            parser.error("move angle and distance must be finite numbers")
    elif args.angle_deg is not None or args.distance_m is not None:
        parser.error("--angle-deg and --distance-m are only valid in move mode")
    if args.mode in ("preflight", "move", "run") and os.environ.get("ROS_DOMAIN_ID") != "40":
        parser.error("Set ROS_DOMAIN_ID=40 before connecting to the robot")
    arena_path = args.arena if args.arena.is_absolute() else ROOT / args.arena
    os.chdir(ROOT)  # Wrapper and formula paths are repo relative.

    layout = load(arena_path)
    manual_movement = None
    manual_metadata = None
    if args.mode == "move":
        try:
            manual_movement, interpreted_angle, actual_distance = (
                interpret_manual_action(layout, args.angle_deg, args.distance_m)
            )
        except ValueError as error:
            parser.error(str(error))
        manual_metadata = {
            "event": "manual_action",
            "zone_direction_deg": interpreted_angle,
            "requested_distance_m": args.distance_m,
            "interpreted_distance_m": actual_distance,
            "movement": manual_movement.model_dump(),
        }
    if args.mode in ("check", "run"):
        for filename in ("best_model.zip", "final_model_8.j.b_30.0M_rep_2.zip"):
            if not (ROOT / "models" / filename).is_file():
                parser.error(f"Missing robot policy checkpoint: {ROOT / 'models' / filename}")
    if args.mode in ("preflight", "move", "run"):
        try:
            preflight(args.robot_ip, layout)
        except RuntimeError as error:
            parser.error(str(error))
    if args.mode == "preflight":
        return 0

    if args.mode == "move":
        assert manual_movement is not None and manual_metadata is not None
        log_dir = ROOT / "logs" / str(time.time_ns())
        log_dir.mkdir(parents=True)
        (log_dir / "manual_action.json").write_text(
            json.dumps(manual_metadata, indent=2, allow_nan=False) + "\n"
        )
        provider = SingleActionProvider(
            manual_movement,
            angle_deg=manual_metadata["zone_direction_deg"],
            distance_m=manual_metadata["interpreted_distance_m"],
        )
        print(
            f"One action: Zone direction {manual_metadata['zone_direction_deg']:g} deg, "
            f"step {manual_metadata['interpreted_distance_m']:.5f} m "
            f"(direction_index={manual_movement.direction_index}, "
            f"magnitude_index={manual_movement.magnitude_index})"
        )
        print(f"Run data: {log_dir}")
        from hrl_tl.robot_demo.control.ros_adapter import run as ros_run

        return ros_run(layout.robot, provider, log_dir / "motion.jsonl")

    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load((ROOT / "configs/wrapper.yaml").read_text())
    )
    reader.low_level_policy_args["model_path"] = str(ROOT / "models/best_model.zip")
    reader.max_episode_steps = layout.max_steps
    wrapper = reader.to_config()  # Loads the CPC/SDSAC primitive checkpoint.
    torch.set_num_threads(1)
    upper = PrimitiveStepPPO.load(
        ROOT / "models/final_model_8.j.b_30.0M_rep_2.zip", device="cpu"
    )

    if args.mode == "check":
        with io.StringIO() as log:
            hierarchy = RobotHierarchy(
                layout, upper, wrapper.wrapper_kwargs, log, args.max_actions
            )
            try:
                start = pose.zone_to_world(layout.start, layout.robot.frame)
                now = time.monotonic()
                command = hierarchy.next_action(
                    pose.Pose2D(
                        x=start.x, y=start.y,
                        yaw=-layout.robot.frame.rotation_rad,
                        stamp=now, received_at=now,
                    )
                )
                print(json.dumps({
                    "decision": hierarchy.decision,
                    "primitive_action": command.model_dump() if command else None,
                    "observation": json.loads(log.getvalue())["observation"],
                    "cmd_vel_topic": layout.robot.ros.cmd_vel_topic,
                    "cmd_vel_type": layout.robot.ros.cmd_vel_type,
                }, indent=2))
                return int(command is None)
            finally:
                hierarchy.close()

    from hrl_tl.robot_demo.control.ros_adapter import run

    log_dir = ROOT / "logs" / str(time.time_ns())
    log_dir.mkdir(parents=True)
    with (log_dir / "policy.jsonl").open("x") as policy_log:
        provider = LearnedPolicyProvider(
            layout, upper, wrapper.wrapper_kwargs, policy_log,
            max_actions=args.max_actions, require_arena=False,
        )
        try:
            print(f"Policy log: {log_dir / 'policy.jsonl'}")
            print(f"Motion log: {log_dir / 'motion.jsonl'}")
            return run(layout.robot, provider, log_dir / "motion.jsonl")
        finally:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
