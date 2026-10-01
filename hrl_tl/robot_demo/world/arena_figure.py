"""Draw a coordinate-accurate diagram from the Gazebo arena snapshot."""

from __future__ import annotations

import pathlib

import matplotlib
from matplotlib import patches, pyplot

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world import arena, gazebo_layout


def save(layout: arena.Arena, destination: pathlib.Path) -> None:
    """Save an English annotated PNG or SVG without running a policy."""
    frame = layout.robot.frame
    scale = frame.sim_units_per_meter
    bounds = layout.robot.bounds
    with matplotlib.rc_context(
        {"font.family": ["DejaVu Sans"], "font.size": 11}
    ):
        figure = pyplot.figure(figsize=(12, 7.6), facecolor="#f5f7fa")
        try:
            figure.text(
                0.055, 0.94, "Zone / Gazebo layout", size=22, weight="bold"
            )
            figure.text(
                0.055,
                0.895,
                "Reach any yellow zone, then the white zone. Orange is forbidden; black is irrelevant.",
                color="#455467",
            )
            axes = figure.add_axes((0.05, 0.11, 0.55, 0.72))
            axes.set_aspect("equal")
            for min_x, max_x, min_y, max_y in layout.walls:
                corners = [
                    pose.zone_to_world(pose.Point2D(x=x, y=y), frame)
                    for x, y in (
                        (min_x, min_y),
                        (max_x, min_y),
                        (max_x, max_y),
                        (min_x, max_y),
                    )
                ]
                axes.add_patch(
                    patches.Polygon(
                        [(p.x, p.y) for p in corners],
                        facecolor="#8b8e92",
                        edgecolor="#535960",
                        linewidth=0.5,
                    )
                )
            for zone in layout.zones:
                center = pose.zone_to_world(zone.center, frame)
                color = tuple(
                    map(float, gazebo_layout.COLORS[zone.color].split())
                )
                axes.add_patch(
                    patches.Circle(
                        (center.x, center.y),
                        zone.radius / scale,
                        facecolor=color,
                        edgecolor="#20252b",
                        linewidth=0.8,
                    )
                )
            start = pose.zone_to_world(layout.start, frame)
            axes.add_patch(
                patches.Circle(
                    (start.x, start.y),
                    bounds.margin,
                    facecolor="#56b4e9",
                    edgecolor="#20252b",
                    linewidth=0.8,
                )
            )
            axes.annotate(
                "Example start position",
                (start.x, start.y),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                size=10,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            axes.autoscale_view()
            axes.margins(0)
            axes.set_xlabel("Gazebo / world x [m]")
            axes.set_ylabel("Gazebo / world y [m]")
            axes.set_title(
                (
                    f"Inside walls: {bounds.max_x - bounds.min_x:g} m × "
                    f"{bounds.max_y - bounds.min_y:g} m"
                )
                if layout.physical_walls
                else "Virtual training walls; no physical walls in Gazebo",
                pad=10,
            )
            figure.text(0.65, 0.79, "Colors and roles", size=15, weight="bold")
            entries = [
                (
                    "#56b4e9",
                    "Start",
                    f"Selected start (blue circle: safety radius {bounds.margin:g} m)",
                ),
                (
                    "#ffda00",
                    "Yellow: first goal",
                    "Enter any one of the three zones",
                ),
                (
                    "white",
                    "White: final goal",
                    "After yellow, move to this layout's white zone",
                ),
                (
                    "#d16100",
                    "Orange: obstacle",
                    "Entering the area fails the task",
                ),
                (
                    "#050505",
                    "Black: irrelevant object",
                    "Passable; does not affect task progress or outcome",
                ),
            ]
            for index, (color, title, detail) in enumerate(entries):
                y = 0.73 - index * 0.105
                figure.add_artist(
                    patches.Circle(
                        (0.662, y),
                        0.009,
                        transform=figure.transFigure,
                        facecolor=color,
                        edgecolor="#20252b",
                        linewidth=0.7,
                    )
                )
                figure.text(0.683, y - 0.005, title, weight="bold")
                figure.text(0.65, y - 0.038, detail, size=10, color="#455467")
            black = [
                pose.zone_to_world(z.center, frame)
                for z in layout.zones
                if z.color == "black"
            ]
            coordinates = "\n".join(f"({p.x:g}, {p.y:g})" for p in black)
            figure.text(
                0.65,
                0.18,
                f"Black world coordinates [m]:\n{coordinates}",
                size=10,
            )
            figure.text(
                0.65,
                0.095,
                "Evaluation: compare the meta-policy's selected spec\n"
                "with the robot's measured trajectory.",
                size=11,
            )
            figure.text(
                0.055,
                0.035,
                f"Layout diagram (not a run result)  |  seed={layout.seed}  |  "
                f"arena={layout.identity[:12]}",
                size=9,
                color="#657286",
            )
            with destination.open("xb") as stream:
                figure.savefig(stream, format=destination.suffix[1:], dpi=160)
        finally:
            pyplot.close(figure)
