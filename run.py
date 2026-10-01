"""Run the supplied Zone hierarchy from Qualisys poses on one TurtleBot3."""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import time
from pathlib import Path

import torch
import yaml
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.inference.hierarchy import RobotHierarchy
from hrl_tl.robot_demo.inference.learned_policy import LearnedPolicyProvider
from hrl_tl.robot_demo.world.arena import load

ROOT = Path(__file__).resolve().parent
DEFAULT_SIM_ROOT = ROOT.parent / "hrl-tl-zone-sim"
DEFAULT_ROBOT_IP = "192.168.0.77"


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
        raise RuntimeError(
            f"ROS preflight failed: {pose_topic} must publish PoseStamped, "
            f"and {cmd_topic} must have a {expected} subscriber. "
            f"Visible topics: {topics}"
        )
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "preflight", "run"))
    parser.add_argument("--arena", type=Path, default=Path("configs/arena.json"))
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.mode in ("preflight", "run") and os.environ.get("ROS_DOMAIN_ID") != "40":
        parser.error("Set ROS_DOMAIN_ID=40 before connecting to the robot")
    arena_path = args.arena if args.arena.is_absolute() else ROOT / args.arena
    sim_root = args.sim_root.resolve()
    if not (sim_root / "configs/wrapper.yaml").is_file():
        parser.error(f"Simulation repo not found: {sim_root}")
    os.chdir(sim_root)  # Upstream wrapper and formula paths are repo relative.

    layout = load(arena_path)
    for filename in ("best_model.zip", "final_model_8.j.b_30.0M_rep_2.zip"):
        if not (ROOT / "models" / filename).is_file():
            parser.error(f"Missing robot policy checkpoint: {ROOT / 'models' / filename}")
    if args.mode in ("preflight", "run"):
        try:
            preflight(args.robot_ip, layout)
        except RuntimeError as error:
            parser.error(str(error))
    if args.mode == "preflight":
        return 0
    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load((sim_root / "configs/wrapper.yaml").read_text())
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
            print(f"Motion log: {log_dir / 'motion.jsonl'}")
            return run(layout.robot, provider, log_dir / "motion.jsonl")
        finally:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
