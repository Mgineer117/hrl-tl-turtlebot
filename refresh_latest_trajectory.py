"""Refresh the bundled trajectories from the newest robot run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import numpy as np
import torch
import yaml
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world.arena import load
from latest_trajectory.plot_trajectories import plot_trajectories
from simulate_policy import simulate


def latest_run(run_id: str | None) -> Path:
    root = ROOT / "logs"
    if run_id is not None:
        candidates = [root / run_id]
    else:
        candidates = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name), reverse=True,
        )
    for folder in candidates:
        if (folder / "motion.jsonl").is_file() and (folder / "policy.jsonl").is_file():
            return folder
    raise FileNotFoundError("No run with both motion.jsonl and policy.jsonl")


def read_run(folder: Path) -> tuple[dict, dict, list[dict], int, str]:
    first_policy = None
    with (folder / "policy.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "policy":
                first_policy = event
                break
    if first_policy is None:
        raise ValueError(f"No policy action in {folder}")
    settings = None
    measured = [first_policy["pose"]]
    commands = 0
    stop_reason = "snapshot_running"
    first_stamp = first_policy["pose"]["stamp"]
    last_stamp = first_stamp
    with (folder / "motion.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            kind = event.get("event")
            if kind == "config":
                settings = event["settings"]
            elif kind == "command":
                commands += 1
            elif kind == "control" and event.get("pose"):
                sample = event["pose"]
                if sample["stamp"] > last_stamp:
                    measured.append(sample)
                    last_stamp = sample["stamp"]
            elif kind == "state" and event.get("state") in ("stopped", "fault"):
                stop_reason = event.get("reason", "")
            elif kind == "shutdown":
                stop_reason = event.get("reason", stop_reason)
    if settings is None or commands == 0:
        raise ValueError(f"Incomplete motion log in {folder}")
    return settings, first_policy, measured, commands, stop_reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Plot a specific logs/<run-id> folder")
    args = parser.parse_args()
    folder = latest_run(args.run_id)
    settings, first, measured, commands, stop_reason = read_run(folder)
    layout = load(ROOT / "configs/arena.json")
    if first["arena_id"] != layout.identity or settings != json.loads(layout.robot.model_dump_json()):
        raise ValueError("Current arena/config differs from the logged run; restore its configuration first")
    start = pose.Pose2D.model_validate(first["pose"]).model_copy(
        update={"stamp": 0.0, "received_at": 0.0}
    )
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
    random.seed(layout.seed)
    np.random.seed(layout.seed)
    torch.manual_seed(layout.seed)
    result = simulate(layout, upper, wrapper.wrapper_kwargs, start, commands,
                      continuous=True)
    ideal = result["trajectory_world_m"]
    summary = {
        "run_id": folder.name,
        "seed": layout.seed,
        "measured_commands": commands,
        "measured_samples": len(measured),
        "stop_reason": stop_reason,
        "open_loop_actions": len(result["actions"]),
        "open_loop_reason": result["reason"],
        "start_world_m": {"x": start.x, "y": start.y, "yaw": start.yaw},
        "measured_end_world_m": {key: measured[-1][key] for key in ("x", "y")},
        "open_loop_end_world_m": {key: ideal[-1][key] for key in ("x", "y")},
    }
    data = {
        **summary,
        "arena_world_m": {
            "bounds": [
                layout.robot.bounds.min_x, layout.robot.bounds.max_x,
                layout.robot.bounds.min_y, layout.robot.bounds.max_y,
            ],
            "zones": [
                {
                    "color": zone.color,
                    "center": pose.zone_to_world(zone.center, layout.robot.frame).model_dump(),
                    "radius": zone.radius / layout.robot.frame.sim_units_per_meter,
                }
                for zone in layout.zones
            ],
        },
        "measured_trajectory_world_m": [
            {"x": point["x"], "y": point["y"]} for point in measured
        ],
        "open_loop_trajectory_world_m": ideal,
    }
    output = ROOT / "latest_trajectory/trajectories.json"
    output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    plot_trajectories(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
