"""A fixed Zone layout shared by observations and Gazebo rendering."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from typing import Any, Literal, Self

import pydantic
import yaml

from hrl_tl.robot_demo import config, pose

# The reference room spans 10 native units between the inner wall faces.
DEFAULT_SCALE = 10.0 / 3.048


class Zone(config.Settings):
    """A colored disk in Zone coordinates."""

    color: Literal["yellow", "red", "white", "black"]
    center: pose.Point2D
    radius: float = pydantic.Field(gt=0)


class Arena(config.Settings):
    """A resolved native environment and its matching robot configuration.

    Attributes:
        environment: Native ZoneEnv keyword arguments with fixed spawns.
        zones: Actual post-swap landmarks, used by Gazebo.
        walls: Native wall bounds ordered min_x, max_x, min_y, max_y.
        start: Initial agent position in Zone coordinates.
        robot: World transform, control limits and ROS topics.
        max_steps: Maximum number of issued low-level movements.
        seed: Layout generation seed for provenance.
        physical_walls: Whether Gazebo has solid walls and inset target limits.
    """

    environment: dict[str, Any]
    zones: tuple[Zone, ...]
    walls: tuple[tuple[float, float, float, float], ...]
    start: pose.Point2D
    robot: config.DemoConfig
    max_steps: int = pydantic.Field(default=250, gt=0)
    seed: int = 0
    physical_walls: bool = True

    @pydantic.model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject snapshots whose rendered geometry differs from native state."""
        spawn = self.environment["scenario_config"]["spawn_config"]
        if spawn["spawn_method"]["mode"] != "fixed":
            raise ValueError("Arena must contain resolved fixed spawns")
        if spawn["agent"] != [self.start.x, self.start.y]:
            raise ValueError("Rendered robot start differs from native spawn")
        expected = sorted(
            (color, *item["pos"], spawn["zone_size"][color])
            for color in ("yellow", "red", "white", "black")
            for item in spawn[f"{color}_zone"]
        )
        actual = sorted(
            (z.color, z.center.x, z.center.y, z.radius) for z in self.zones
        )
        if actual != expected:
            raise ValueError("Rendered zones differ from native spawns")
        initial = pose.zone_to_world(self.start, self.robot.frame)
        if not self.robot.bounds.contains(
            initial.x, initial.y, self.robot.motion.wall_stop_margin
        ):
            raise ValueError("Robot start is too close to a wall")
        clearance = spawn["agent_size"]
        if any(
            math.hypot(self.start.x - z.center.x, self.start.y - z.center.y)
            < z.radius + clearance
            for z in self.zones
        ):
            raise ValueError("Robot start overlaps a zone")
        grid = self.environment["world_config"]["grid"]
        cell, rows = grid["cell_size"], grid["layout"]
        walls = sorted(
            (
                (c - 0.5) * cell,
                (c + 0.5) * cell,
                (len(rows) - r - 1.5) * cell,
                (len(rows) - r - 0.5) * cell,
            )
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value == "#"
        )
        if sorted(self.walls) != walls:
            raise ValueError("Rendered walls differ from native grid")
        return self

    @property
    def identity(self) -> str:
        """Content identity shared by the controller and stage display."""
        values = self.model_dump()
        if self.robot.motion.turn_clearance == 0:
            values["robot"]["motion"].pop("turn_clearance")
        if self.physical_walls:
            # Keep identities of existing physical-wall snapshots stable.
            values.pop("physical_walls")
        data = json.dumps(values, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def save(self, path: pathlib.Path) -> None:
        """Write a new snapshot without overwriting an earlier run."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(self.model_dump_json(indent=2) + "\n")


def load(path: pathlib.Path) -> Arena:
    """Load the exact snapshot used to create the Gazebo world."""
    return Arena.model_validate_json(path.read_text())


def prepare(
    path: pathlib.Path, seed: int, scale: float = DEFAULT_SCALE
) -> Arena:
    """Resolve random spawns once using the existing ContGrid environment.

    Args:
        path: Existing native environment YAML.
        seed: Seed applied to the native reset.
        scale: Zone units per world metre; defaults to a 3.048 m reference room.
    """
    from contgrid.envs.zone import env as zone_env

    with path.open() as stream:
        source = yaml.safe_load(stream)
    if source["id"] != "contgrid/Zone-v0":
        raise ValueError("Robot arena requires contgrid/Zone-v0")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    native = zone_env.ZoneEnv(**source["env_kwargs"])
    try:
        native.reset(seed=seed)
        scenario, world = native.scenario, native.env.world
        grid = world.grid
        if (
            any("#" in row[1:-1] for row in grid.layout[1:-1])
            or any(value != "#" for value in grid.layout[0] + grid.layout[-1])
            or any(row[0] != "#" or row[-1] != "#" for row in grid.layout)
        ):
            raise ValueError("Initial robot demo supports an open Zone room")
        cell = grid.cell_size
        min_x = min_y = cell / 2
        max_x = (grid.width_cells - 1.5) * cell
        max_y = (grid.height_cells - 1.5) * cell
        frame = config.FrameConfig(
            origin_x_m=-(min_x + max_x) / (2 * scale),
            origin_y_m=-(min_y + max_y) / (2 * scale),
            sim_units_per_meter=scale,
        )
        lower = pose.zone_to_world(pose.Point2D(x=min_x, y=min_y), frame)
        upper = pose.zone_to_world(pose.Point2D(x=max_x, y=max_y), frame)
        robot = config.DemoConfig(
            frame=frame,
            # Resolve shortened movements while retaining the real footprint.
            motion=config.MotionConfig(
                position_tolerance=(
                    config.MotionConfig().position_tolerance / max(1.0, scale)
                ),
            ),
            bounds=config.Bounds(
                min_x=lower.x,
                min_y=lower.y,
                max_x=upper.x,
                max_y=upper.y,
            ),
        )
        # Native spawning uses a point-mass radius. Reject samples that would
        # put the physical robot on a disk or inside the wall safety margin.
        for _ in range(1000):
            agent = world.agents[0]
            start = pose.Point2D(
                x=float(agent.state.pos[0]), y=float(agent.state.pos[1])
            )
            initial = pose.zone_to_world(start, frame)
            clearance = max(agent.size, robot.bounds.margin * scale)
            landmarks = (
                scenario.yellow + scenario.red + scenario.white + scenario.black
            )
            if robot.bounds.contains(
                initial.x, initial.y, robot.motion.wall_stop_margin
            ) and all(
                math.hypot(start.x - lm.state.pos[0], start.y - lm.state.pos[1])
                >= lm.size + clearance
                for lm in landmarks
            ):
                break
            if scenario.config.spawn_config.agent is not None:
                raise ValueError("Fixed start lacks robot clearance")
            native.reset()
        else:
            raise ValueError("Could not sample a start with robot clearance")
        scenario_config = scenario.config.model_dump(mode="json")
        spawn = scenario_config["spawn_config"]
        spawn["agent"] = [start.x, start.y]
        spawn["spawn_method"] = {"mode": "fixed"}
        zones: list[Zone] = []
        groups = (
            ("yellow", scenario.yellow),
            ("red", scenario.red),
            ("white", scenario.white),
            ("black", scenario.black),
        )
        for color, landmarks in groups:
            spawn[f"{color}_zone"] = []
            for landmark in landmarks:
                x, y = map(float, landmark.state.pos)
                spawn[f"{color}_zone"].append({"pos": [x, y]})
                zones.append(
                    Zone(
                        color=color,
                        center=pose.Point2D(x=x, y=y),
                        radius=float(landmark.size),
                    )
                )
        environment = dict(source["env_kwargs"])
        environment["scenario_config"] = scenario_config
        return Arena(
            environment=environment,
            zones=tuple(zones),
            walls=tuple(map(tuple, scenario.wall_bounds.tolist())),
            start=start,
            robot=robot,
            seed=seed,
            max_steps=source["max_episode_steps"],
        )
    finally:
        native.close()
