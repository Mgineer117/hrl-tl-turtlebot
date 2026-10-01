"""MuJoCo XML model generator for the Fetch Reach-Avoid environment."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import gymnasium_robotics


def get_reach_avoid_xml_path() -> str:
    """Generate or retrieve the path to the Fetch Reach-Avoid MuJoCo XML model.

    Returns:
        The absolute path string to the generated MuJoCo XML file.
    """
    gym_robotics_dir = Path(gymnasium_robotics.__file__).resolve().parent
    fetch_assets_dir = gym_robotics_dir / "envs" / "assets" / "fetch"
    stls_dir = gym_robotics_dir / "envs" / "assets" / "stls" / "fetch"
    textures_dir = gym_robotics_dir / "envs" / "assets" / "textures"

    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco>
\t<compiler angle="radian" coordinate="local" meshdir="{stls_dir}" texturedir="{textures_dir}"/>
\t<option timestep="0.002">
\t\t<flag warmstart="enable"/>
\t</option>

\t<include file="{fetch_assets_dir}/shared.xml"/>

\t<worldbody>
\t\t<geom name="floor0" pos="0.8 0.75 0" size="0.85 0.7 1" type="plane" condim="3" material="floor_mat"/>

\t\t<include file="{fetch_assets_dir}/robot.xml"/>

\t\t<body pos="1.3 0.75 0.2" name="table0">
\t\t\t<geom size="0.25 0.35 0.2" type="box" mass="2000" material="table_mat"/>
\t\t</body>

\t\t<!-- Goal sites (3D spheres) -->
\t\t<site name="yellow0" pos="1.300 0.870 0.500" size="0.035 0.035 0.035" rgba="1.0 0.9 0.0 0.7" type="sphere"/>
\t\t<site name="yellow1" pos="1.196 0.690 0.500" size="0.035 0.035 0.035" rgba="1.0 0.9 0.0 0.7" type="sphere"/>
\t\t<site name="white0" pos="1.404 0.690 0.500" size="0.035 0.035 0.035" rgba="1.0 1.0 1.0 0.9" type="sphere"/>

\t\t<!-- Obstacle sites (3D spheres) -->
\t\t<site name="red0" pos="1.248 0.780 0.500" size="0.035 0.035 0.035" rgba="0.9 0.1 0.1 0.75" type="sphere"/>
\t\t<site name="red1" pos="1.352 0.780 0.500" size="0.035 0.035 0.035" rgba="0.9 0.1 0.1 0.75" type="sphere"/>
\t\t<site name="red2" pos="1.300 0.690 0.500" size="0.035 0.035 0.035" rgba="0.9 0.1 0.1 0.75" type="sphere"/>

\t\t<light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"/>
\t</worldbody>
</mujoco>
"""
    output_dir = Path(__file__).resolve().parent / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "reach_avoid.xml"

    # Validate whether existing model matches active environment asset paths.
    if xml_path.is_file() and xml_path.stat().st_size > 0:
        try:
            content = xml_path.read_text(encoding="utf-8")
            if (
                str(fetch_assets_dir) in content
                and str(stls_dir) in content
                and str(textures_dir) in content
            ):
                return str(xml_path)
        except OSError:
            pass

    temp_path = (
        output_dir
        / f"reach_avoid_{os.getpid()}_{threading.get_ident()}.xml.tmp"
    )
    temp_path.write_text(xml_content, encoding="utf-8")
    temp_path.replace(xml_path)

    return str(xml_path)
