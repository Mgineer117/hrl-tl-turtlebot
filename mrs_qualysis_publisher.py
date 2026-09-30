# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import xml.etree.cElementTree as ET
from threading import Thread

import numpy as np
import qtm_rt as qtm
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.spatial.transform import Rotation


class QualysisPublisher(Node):
    def __init__(self, ip_address, marker_deck_name):
        super().__init__("qualysis_publisher")
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.publisher_ = self.create_publisher(
            PoseStamped, f"qualysis/{marker_deck_name}", qos
        )
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name

        self.qualysis_client = QualisysClient(self.ip_address, self.marker_deck_name)

    def timer_callback(self):
        self.pose = PoseStamped()

        data = self.qualysis_client.data

        self.pose.header.frame_id = "mocap"
        self.pose.header.stamp = self.get_clock().now().to_msg()
        self.pose.pose.position.x = data["x"] if data["x"] else 0.0
        self.pose.pose.position.y = data["y"] if data["y"] else 0.0
        self.pose.pose.position.z = data["z"] if data["z"] else 0.0
        # quat = data['quaternion']
        # self.pose.pose.orientation.x = quat[0] if quat[0] else 0.0
        # self.pose.pose.orientation.y = quat[1] if quat[1] else 0.0
        # self.pose.pose.orientation.z = quat[2] if quat[2] else 0.0
        # self.pose.pose.orientation.w = quat[3] if quat[3] else 0.0
        self.pose.pose.orientation.z = data["yaw"] if data["yaw"] else 0.0
        self.pose.pose.orientation.y = data["pitch"] if data["pitch"] else 0.0
        self.pose.pose.orientation.x = data["roll"] if data["roll"] else 0.0

        self.publisher_.publish(self.pose)
        # qualysis_client.close()


class QualisysClient(Thread):
    def __init__(self, ip_address, marker_deck_name):
        Thread.__init__(self)
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True
        self.data = {
            "time": [],
            "x": [],
            "y": [],
            "z": [],
            # 'quaternion': []
            "yaw": [],
            "pitch": [],
            "roll": [],
        }
        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while self._stay_open:
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        print("QualisysClient: Connect to motion capture system")
        self.connection = await qtm.connect(self.ip_address, version="1.24")
        params = await self.connection.get_parameters(parameters=["6d"])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [
            label.text.strip() for index, label in enumerate(xml.findall("*/Body/Name"))
        ]
        await self.connection.stream_frames(
            components=["6d"],
            on_packet=self._on_packet,
        )

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            print(f"QualisysClient: No rigid bodies found")
            return

        if self.marker_deck_name not in self.qtm_6DoF_labels:
            print(f"QualisysClient: Marker deck {self.marker_deck_name} not found")
            return

        index = self.qtm_6DoF_labels.index(self.marker_deck_name)
        position, orientation = bodies[index]

        # Get time in seconds, with respect to the qualisys clock
        t = packet.timestamp / 1e6

        # Get position of marker deck (x, y, z in meters)
        x, y, z = np.array(position) / 1e3

        # Get orientation of marker deck (yaw, pitch, roll in radians)
        R = Rotation.from_matrix(np.reshape(orientation.matrix, (3, -1), order="F"))
        # quaternion = R.as_quat()#euler('ZYX', degrees=False)
        yaw, pitch, roll = R.as_euler("ZYX", degrees=False)

        # Publish one complete frame to the ROS timer thread at a time.
        self.data = {
            "time": t, "x": x, "y": y, "z": z,
            "yaw": yaw, "pitch": pitch, "roll": roll,
        }

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()
