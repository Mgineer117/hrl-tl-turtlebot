"""Read trajectories.json beside this file and plot both paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

FEET_PER_METER = 3.280839895013123


def plot_trajectories(data_path: Path | None = None) -> None:
    """Plot saved measured and open-loop paths without loading robot code."""
    source = data_path or Path(__file__).with_name("trajectories.json")
    data = json.loads(source.read_text())
    measured = data["measured_trajectory_world_m"]
    ideal = data["open_loop_trajectory_world_m"]
    if not measured or not ideal:
        raise ValueError("Both saved trajectories must contain positions")
    ft = FEET_PER_METER
    min_x, max_x, min_y, max_y = data["arena_world_m"]["bounds"]
    figure, axes = plt.subplots(figsize=(9, 9))
    axes.add_patch(Rectangle(
        (min_x * ft, min_y * ft), (max_x - min_x) * ft, (max_y - min_y) * ft,
        fill=False, edgecolor="#555", linewidth=2,
    ))
    colors = {"yellow": "#e6c100", "white": "white", "red": "#d45b50", "black": "#222"}
    for zone in data["arena_world_m"]["zones"]:
        center = zone["center"]
        axes.add_patch(Circle(
            (center["x"] * ft, center["y"] * ft), zone["radius"] * ft,
            facecolor=colors[zone["color"]], edgecolor="#333", linewidth=0.8,
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
             title=f"Latest TurtleBot run {data['run_id']} · seed {data['seed']}")
    axes.set_aspect("equal")
    axes.set_xlim(min_x * ft - 0.5, max_x * ft + 0.5)
    axes.set_ylim(min_y * ft - 0.5, max_y * ft + 0.5)
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right")
    figure.text(
        0.5, 0.02,
        f"{data['measured_commands']} logged commands · stop: {data['stop_reason']}",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("png", "pdf"):
        figure.savefig(source.with_name(f"latest_trajectory.{extension}"), dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    plot_trajectories()
