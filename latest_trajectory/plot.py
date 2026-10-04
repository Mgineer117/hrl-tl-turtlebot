"""Plot the newest logged robot path beside an ideal policy rollout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Circle, Rectangle
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO

from hrl_tl.config.meta_option import TLMetaOptionWrapperConfigReader
from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world.arena import load
from simulate_policy import simulate

FEET_PER_METER = 3.280839895013123


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


def draw(layout, measured: list[dict], ideal: list[dict], folder: Path,
         commands: int, reason: str) -> None:
    ft = FEET_PER_METER
    bounds = layout.robot.bounds
    figure, axes = plt.subplots(figsize=(9, 9))
    axes.add_patch(Rectangle(
        (bounds.min_x * ft, bounds.min_y * ft),
        (bounds.max_x - bounds.min_x) * ft,
        (bounds.max_y - bounds.min_y) * ft,
        fill=False, edgecolor="#555", linewidth=2,
    ))
    colors = {"yellow": "#e6c100", "white": "white", "red": "#d45b50", "black": "#222"}
    for zone in layout.zones:
        center = pose.zone_to_world(zone.center, layout.robot.frame)
        axes.add_patch(Circle(
            (center.x * ft, center.y * ft),
            zone.radius / layout.robot.frame.sim_units_per_meter * ft,
            facecolor=colors[zone.color], edgecolor="#333", linewidth=0.8,
        ))
    axes.plot([p["x"] * ft for p in measured], [p["y"] * ft for p in measured],
              color="#0072b2", linewidth=1.5, label="Measured robot")
    axes.plot([p["x"] * ft for p in ideal], [p["y"] * ft for p in ideal],
              color="#e69f00", linewidth=1.3, label="Ideal open-loop policy")
    for points, color, marker, label in (
        (measured, "#0072b2", "o", "Measured end"),
        (ideal, "#e69f00", "x", "Open-loop end"),
    ):
        axes.plot(points[-1]["x"] * ft, points[-1]["y"] * ft,
                  marker=marker, color=color, markersize=8, label=label)
    axes.plot(measured[0]["x"] * ft, measured[0]["y"] * ft,
              marker="*", color="black", markersize=12, label="Same start")
    axes.set(xlabel="World x (ft)", ylabel="World y (ft)",
             title=f"Latest TurtleBot run {folder.name} · seed {layout.seed}")
    axes.set_aspect("equal")
    axes.set_xlim(bounds.min_x * ft - 0.5, bounds.max_x * ft + 0.5)
    axes.set_ylim(bounds.min_y * ft - 0.5, bounds.max_y * ft + 0.5)
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right")
    figure.text(0.5, 0.02, f"{commands} logged commands · stop: {reason}", ha="center")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("png", "pdf"):
        figure.savefig(Path(__file__).with_name(f"latest_trajectory.{extension}"), dpi=180)
    plt.close(figure)


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
    draw(layout, measured, ideal, folder, commands, stop_reason)
    summary = {
        "run_id": folder.name,
        "measured_commands": commands,
        "measured_samples": len(measured),
        "stop_reason": stop_reason,
        "open_loop_actions": len(result["actions"]),
        "open_loop_reason": result["reason"],
        "start_world_m": {"x": start.x, "y": start.y, "yaw": start.yaw},
        "measured_end_world_m": {key: measured[-1][key] for key in ("x", "y")},
        "open_loop_end_world_m": {key: ideal[-1][key] for key in ("x", "y")},
    }
    output = Path(__file__).with_name("latest_trajectory_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
