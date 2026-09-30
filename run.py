"""Run the supplied Zone hierarchy from Qualisys poses on one TurtleBot3."""

from __future__ import annotations

import argparse
import io
import json
import os
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run"))
    parser.add_argument("--arena", type=Path, default=Path("configs/arena.json"))
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--max-actions", type=int, default=250)
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    arena_path = args.arena if args.arena.is_absolute() else ROOT / args.arena
    sim_root = args.sim_root.resolve()
    if not (sim_root / "configs/wrapper.yaml").is_file():
        parser.error(f"Simulation repo not found: {sim_root}")
    os.chdir(sim_root)  # Upstream wrapper and formula paths are repo relative.

    layout = load(arena_path)
    reader = TLMetaOptionWrapperConfigReader.model_validate(
        yaml.safe_load((sim_root / "configs/wrapper.yaml").read_text())
    )
    reader.max_episode_steps = layout.max_steps
    wrapper = reader.to_config()  # Loads the CPC/SDSAC primitive checkpoint.
    torch.set_num_threads(1)
    upper = PrimitiveStepPPO.load(
        sim_root / "models/final_model_8.j.b_30.0M_rep_2.zip", device="cpu"
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
