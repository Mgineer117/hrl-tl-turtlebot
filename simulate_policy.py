"""Simulate the learned Zone policy from a saved Qualisys pose sample."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

import torch
import yaml
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import motion
from hrl_tl.robot_demo.inference.hierarchy import RobotHierarchy
from hrl_tl.robot_demo.world import arena, arena_figure

ROOT = pathlib.Path(__file__).resolve().parent


def _advance(
    current: pose.Pose2D, command: motion.Velocity, dt: float, stamp: float
) -> pose.Pose2D:
    """Integrate the commanded unicycle velocity for one fixed time step."""
    yaw = current.yaw
    turn = command.angular * dt
    if abs(command.angular) < 1e-10:
        x = current.x + command.linear * math.cos(yaw) * dt
        y = current.y + command.linear * math.sin(yaw) * dt
    else:
        radius = command.linear / command.angular
        x = current.x + radius * (math.sin(yaw + turn) - math.sin(yaw))
        y = current.y - radius * (math.cos(yaw + turn) - math.cos(yaw))
    return pose.Pose2D(
        x=x,
        y=y,
        yaw=yaw + turn,
        stamp=stamp,
        received_at=stamp,
    )


def _append_position(
    trajectory: list[pose.Point2D], point: pose.Point2D
) -> None:
    if not trajectory or math.hypot(
        trajectory[-1].x - point.x, trajectory[-1].y - point.y
    ) >= 0.0001:
        trajectory.append(point)


def simulate(
    layout: arena.Arena,
    upper: PrimitiveStepPPO,
    wrapper_kwargs: dict,
    start: pose.Pose2D,
    max_actions: int,
) -> dict:
    """Run model waypoints through the repository controller and ideal motion."""
    motion_controller = motion.MotionExecutor(
        layout.robot.motion, layout.robot.bounds
    )
    if not motion_controller.update_pose(start):
        raise RuntimeError("Could not initialize the simulated pose")

    import io

    hierarchy = RobotHierarchy(
        layout, upper, wrapper_kwargs, io.StringIO(), max_actions
    )
    sample_every = max(1, round(layout.robot.ros.control_hz / 10))
    trajectory = [pose.Point2D(x=start.x, y=start.y)]
    actions: list[dict] = []
    dt = 1.0 / layout.robot.ros.control_hz
    sim_time = start.stamp
    current = start
    command_id = 0
    reason = ""
    try:
        command = hierarchy.next_action(current)
        while command is not None:
            command_id += 1
            decision = hierarchy.decision or {}
            if not motion_controller.submit(command_id, command.target, sim_time):
                raise RuntimeError(f"Controller rejected action {command_id}")

            ticks = 0
            while True:
                velocity = motion_controller.tick(sim_time)
                sim_time += dt
                current = _advance(current, velocity, dt, sim_time)
                motion_controller.update_pose(current)
                ticks += 1
                if ticks % sample_every == 0:
                    _append_position(
                        trajectory, pose.Point2D(x=current.x, y=current.y)
                    )
                result = motion_controller.take_result()
                if result is not None:
                    if result.outcome != "target_reached":
                        reason = f"motion_{result.outcome}:{result.reason}"
                    break
                if motion_controller.state in (motion.State.FAULT, motion.State.STOPPED):
                    reason = f"motion_{motion_controller.state.value}:{motion_controller.reason}"
                    break
                if ticks > math.ceil(
                    layout.robot.motion.motion_timeout / dt
                ) + 2:
                    reason = "simulation_timeout"
                    break

            _append_position(
                trajectory, pose.Point2D(x=current.x, y=current.y)
            )
            actions.append(
                {
                    "command_id": command_id,
                    "policy_decision": decision,
                    "action": command.model_dump(),
                    "target_world": command.target.model_dump(),
                    "actual_end_world": {"x": current.x, "y": current.y},
                    "motion_result": (
                        result.model_dump() if result is not None else None
                    ),
                }
            )
            if reason:
                break
            command = hierarchy.next_action(current)
            if command is None:
                reason = hierarchy.reason or "policy_ended"
        if not reason:
            reason = hierarchy.reason or "policy_ended"
    finally:
        hierarchy.close()
    return {
        "reason": reason,
        "actions": actions,
        "trajectory_world_m": [point.model_dump() for point in trajectory],
        "start_pose_world": start.model_dump(),
        "end_pose_world": current.model_dump(),
        "simulated_seconds": sim_time - start.stamp,
        "controller_dt_seconds": dt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", type=pathlib.Path, default=ROOT / "configs/arena.json")
    parser.add_argument("--raw-x", type=float, default=-2.5447392578125)
    parser.add_argument("--raw-y", type=float, default=0.023682369)
    parser.add_argument("--yaw", type=float, default=-1.53085881)
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "artifacts/policy_trajectory.png",
    )
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    os.chdir(ROOT)
    layout = arena.load(args.arena if args.arena.is_absolute() else ROOT / args.arena)
    for filename in ("best_model.zip", "final_model_8.j.b_30.0M_rep_2.zip"):
        if not (ROOT / "models" / filename).is_file():
            parser.error(f"Missing policy checkpoint: {ROOT / 'models' / filename}")

    raw_start = pose.Pose2D(
        x=args.raw_x,
        y=args.raw_y,
        yaw=args.yaw,
        stamp=0.0,
        received_at=0.0,
    )
    start = pose.mocap_to_world(raw_start, layout.robot.ros)
    if not layout.robot.bounds.contains(
        start.x, start.y, layout.robot.motion.wall_stop_margin
    ):
        parser.error("Corrected start pose is outside the controller wall buffer")

    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load((ROOT / "configs/wrapper.yaml").read_text())
    )
    reader.low_level_policy_args["model_path"] = str(ROOT / "models/best_model.zip")
    reader.max_episode_steps = layout.max_steps
    wrapper = reader.to_config()
    torch.set_num_threads(1)
    upper = PrimitiveStepPPO.load(
        ROOT / "models/final_model_8.j.b_30.0M_rep_2.zip", device="cpu"
    )
    result = simulate(layout, upper, wrapper.wrapper_kwargs, start, args.max_actions)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    start_zone = pose.world_to_zone(start, layout.robot.frame)
    end = result["end_pose_world"]
    end_zone = pose.world_to_zone(pose.Point2D(x=end["x"], y=end["y"]), layout.robot.frame)
    summary = {
        "reason": result["reason"],
        "actions": len(result["actions"]),
        "start_world": [start.x, start.y],
        "end_world": [end["x"], end["y"]],
    }
    sampled = [pose.Point2D.model_validate(p) for p in result["trajectory_world_m"]]
    arena_figure.save(
        layout, output, trajectory=sampled, summary=summary, overwrite=True
    )
    result["source_qtm_pose"] = {
        "raw_x_m": args.raw_x,
        "raw_y_m": args.raw_y,
        "yaw_rad": args.yaw,
        "corrected_zone_position": {"x": start_zone.x, "y": start_zone.y},
        "final_zone_position": {"x": end_zone.x, "y": end_zone.y},
    }
    result["arena_identity"] = layout.identity
    result["image"] = str(output)
    trace_path = output.with_suffix(".json")
    trace_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"First QTM sample raw: ({args.raw_x:.6f}, {args.raw_y:.6f}) m, yaw={args.yaw:.6f} rad")
    print(f"Corrected world: ({start.x:.6f}, {start.y:.6f}) m; Zone: ({start_zone.x:.3f}, {start_zone.y:.3f})")
    print(f"Rollout: {result['reason']}; actions: {len(result['actions'])}")
    print(f"Final world: ({end['x']:.6f}, {end['y']:.6f}) m; Zone: ({end_zone.x:.3f}, {end_zone.y:.3f})")
    print(f"Arena image: {output}")
    print(f"Rollout data: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
