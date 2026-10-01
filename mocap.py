"""Publish fresh QTM rigid-body poses using the adapted mrs2025 publisher."""

from __future__ import annotations

import argparse
import math
import os

import rclpy

from mrs_qualysis_publisher import QualysisPublisher


class FreshPublisher(QualysisPublisher):
    def __init__(self, ip: str, marker: str) -> None:
        self._last_sample = float("-inf")
        super().__init__(ip, marker)

    def timer_callback(self) -> None:
        data = self.qualysis_client.data
        try:
            sample = float(data["time"])
            valid = all(math.isfinite(float(data[k])) for k in
                        ("x", "y", "z", "yaw", "pitch", "roll"))
        except (TypeError, ValueError):
            return
        if valid and math.isfinite(sample) and sample > self._last_sample:
            self._last_sample = sample
            super().timer_callback()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="128.174.245.64", help="QTM server IP address")
    parser.add_argument("--marker", default="tb3_1", help="QTM rigid-body name")
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "40":
        parser.error("Set ROS_DOMAIN_ID=40 before publishing mocap poses")
    rclpy.init(args=[])
    publisher = FreshPublisher(args.ip, args.marker)
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.qualysis_client.close()
        publisher.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
