"""Local Gazebo world generation using the installed TurtleBot3 model."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as xml

from ament_index_python import packages

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world import arena, gazebo_layout


def write_world(
    model: str,
    destination: pathlib.Path,
    layout: arena.Arena | None = None,
) -> pathlib.Path:
    """Build a world with an official model and world-pose publisher.

    Returns:
        The model resource directory to add to GZ_SIM_RESOURCE_PATH.

    Raises:
        ValueError: The model is unsupported or lacks its drive plugin.
    """
    if model not in ("burger", "waffle", "waffle_pi"):
        raise ValueError(f"Unsupported TurtleBot3 model: {model}")
    resources = (
        pathlib.Path(packages.get_package_share_directory("turtlebot3_gazebo"))
        / "models"
    )
    robot = xml.parse(resources / f"turtlebot3_{model}/model.sdf").getroot()
    model_element = robot.find("model")
    if model_element is None:
        raise ValueError("TurtleBot3 SDF contains no model")
    model_element.set("name", "robot_demo_turtlebot")
    pose_element = model_element.find("pose")
    if pose_element is None:
        pose_element = xml.SubElement(model_element, "pose")
    pose_element.text = "0 0 0.01 0 0 0"
    if layout is not None:
        start = pose.zone_to_world(layout.start, layout.robot.frame)
        pose_element.text = (
            f"{start.x} {start.y} 0.01 0 0 {-layout.robot.frame.rotation_rad}"
        )
    drive = model_element.find("plugin[@name='gz::sim::systems::DiffDrive']")
    if drive is None:
        raise ValueError(
            "TurtleBot3 model requires a Gazebo Sim DiffDrive plugin"
        )
    topic = drive.find("topic")
    if topic is None:
        topic = xml.SubElement(drive, "topic")
    topic.text = "/robot_demo/cmd_vel"
    truth = xml.SubElement(
        model_element,
        "plugin",
        filename="gz-sim-pose-publisher-system",
        name="gz::sim::systems::PosePublisher",
    )
    values = (
        ("publish_model_pose", "true"),
        ("publish_nested_model_pose", "true"),
        ("publish_link_pose", "false"),
        ("publish_sensor_pose", "false"),
        ("publish_collision_pose", "false"),
        ("publish_visual_pose", "false"),
        ("use_pose_vector_msg", "false"),
        ("update_frequency", "60"),
        ("topic", "/robot_demo/ground_truth"),
    )
    for name, value in values:
        xml.SubElement(truth, name).text = value
    sdf = xml.fromstring(_WORLD)
    world = sdf.find("world")
    if world is None:
        raise ValueError("World template is missing its world")
    world.append(model_element)
    if layout is not None:
        gazebo_layout.append_layout(world, layout)
    xml.indent(sdf)
    xml.ElementTree(sdf).write(destination, encoding="unicode")
    return resources


_WORLD = """<sdf version="1.9">
  <world name="robot_demo">
    <physics name="physics" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"
      name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.8</gravity>
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose><direction>-0.5 0.1 -0.9</direction>
      <diffuse>0.8 0.8 0.8 1</diffuse>
    </light>
    <model name="ground_plane"><static>true</static>
      <link name="ground">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size>
          </plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size>
          </plane></geometry>
          <material><diffuse>0.65 0.65 0.65 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="robot_demo_recording_camera"><static>true</static>
      <pose>3 -3 3 0 0.61548 2.35619</pose>
      <link name="camera_link">
        <sensor name="camera" type="camera">
          <always_on>true</always_on><update_rate>5</update_rate>
          <topic>/robot_demo/gazebo_camera</topic>
          <camera name="robot_demo_camera">
            <horizontal_fov>1.15</horizontal_fov>
            <image><width>640</width><height>480</height>
              <format>R8G8B8</format></image>
            <clip><near>0.1</near><far>100</far></clip>
          </camera>
        </sensor>
      </link>
    </model>
  </world>
</sdf>
"""
