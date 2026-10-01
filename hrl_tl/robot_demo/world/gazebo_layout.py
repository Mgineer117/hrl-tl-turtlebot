"""SDF geometry generated from the same fixed arena as the policy."""

from __future__ import annotations

import xml.etree.ElementTree as xml

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world import arena

COLORS: dict[str, str] = {
    "yellow": "1 0.85 0 1",
    "red": "0.82 0.38 0.0 1",  # Native red predicate, orange in the reference.
    "white": "1 1 1 1",
    "black": "0.02 0.02 0.02 1",
}


def goal_color(layout: arena.Arena, stage: int) -> str | None:
    """Resolve the current native subtask goal, including terminal stages."""
    sequence = layout.environment["scenario_config"]["spawn_config"][
        "subtask_seq"
    ]
    return sequence[stage]["goal"] if 0 <= stage < len(sequence) else None


def _model(
    world: xml.Element,
    name: str,
    location: str,
    geometry: str,
    color: str,
    *,
    collision: bool = False,
) -> None:
    model = xml.SubElement(world, "model", name=name)
    xml.SubElement(model, "static").text = "true"
    xml.SubElement(model, "pose").text = location
    link = xml.SubElement(model, "link", name="body")
    visual = xml.SubElement(link, "visual", name="color")
    visual.append(xml.fromstring(geometry))
    material = xml.SubElement(visual, "material")
    xml.SubElement(material, "diffuse").text = color
    xml.SubElement(material, "ambient").text = color
    if collision:
        collider = xml.SubElement(link, "collision", name="wall")
        collider.append(xml.fromstring(geometry))


def append_layout(world: xml.Element, layout: arena.Arena) -> None:
    """Render visit disks and optional physical walls.

    Orange is a forbidden task region (native red), not a solid body. Native
    failure is measured at its visit radius, just as in the training world.
    """
    frame = layout.robot.frame
    scale = frame.sim_units_per_meter
    for index, zone in enumerate(layout.zones):
        point = pose.zone_to_world(zone.center, frame)
        radius = zone.radius / scale
        disk = (
            f"<geometry><cylinder><radius>{radius}</radius>"
            "<length>0.002</length></cylinder></geometry>"
        )
        _model(
            world,
            f"robot_demo_zone_{index}",
            f"{point.x} {point.y} 0.005 0 0 0",
            disk,
            COLORS[zone.color],
        )
        halo = (
            f"<geometry><cylinder><radius>{radius + 0.04 / scale}</radius>"
            "<length>0.002</length></cylinder></geometry>"
        )
        height = 0.002 if zone.color == goal_color(layout, 0) else -2.0
        _model(
            world,
            f"robot_demo_goal_{index}",
            f"{point.x} {point.y} {height} 0 0 0",
            halo,
            "0 0.9 1 1",
        )
    if not layout.physical_walls:
        return
    for index, (min_x, max_x, min_y, max_y) in enumerate(layout.walls):
        point = pose.zone_to_world(
            pose.Point2D(
                x=(min_x + max_x) / 2,
                y=(min_y + max_y) / 2,
            ),
            frame,
        )
        box = (
            f"<geometry><box><size>{(max_x - min_x) / scale} "
            f"{(max_y - min_y) / scale} 0.3</size></box></geometry>"
        )
        _model(
            world,
            f"robot_demo_wall_{index}",
            f"{point.x} {point.y} 0.15 0 0 {-frame.rotation_rad}",
            box,
            "0.25 0.25 0.3 1",
            collision=True,
        )
