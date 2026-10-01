"""Offline checks for marker correction and a Burger drive deadband."""

import math
import unittest

from hrl_tl.robot_demo import config, pose
from hrl_tl.robot_demo.control import runtime


class MotionCalibrationTest(unittest.TestCase):
    def test_marker_offset_uses_corrected_heading(self):
        settings = config.RosConfig(
            heading_offset_rad=math.pi / 2,
            marker_offset_x_m=0.01,
        )
        result = pose.mocap_to_world(
            pose.Pose2D(x=0, y=0, yaw=0, stamp=0, received_at=0), settings
        )
        self.assertAlmostEqual(result.x, 0)
        self.assertAlmostEqual(result.y, -0.01)

    def test_minimum_drive_speed_allows_second_action(self):
        settings = config.MotionConfig(
            min_linear_speed=0.02, position_tolerance=0.01,
            motion_timeout=10,
        )
        demo_settings = config.DemoConfig(
            motion=settings,
            frame=config.FrameConfig(sim_units_per_meter=10 / 3.048),
        )
        demo = runtime.DemoRuntime(demo_settings, 0)
        demo.update_pose(
            pose.Pose2D(x=0, y=0, yaw=math.pi, stamp=0, received_at=0)
        )
        demo.tick(0)
        self.assertEqual(demo.request.command_id, 1)
        demo.submit_action(1, pose.Point2D(x=-0.01524, y=0), 0)
        x = 0.0
        events = demo.take_events()
        for step in range(1, 310):
            now = step / 30
            command = demo.tick(now)
            events.extend(demo.take_events())
            # Model a drive that does not move below 0.012 m/s.
            if abs(command.linear) >= 0.012:
                x -= command.linear / 30
            demo.update_pose(
                pose.Pose2D(x=x, y=0, yaw=math.pi, stamp=now, received_at=now)
            )
            if demo.request is not None:
                break
        self.assertIn("target_reached", [
            event["outcome"] for event in events if event["event"] == "result"
        ])
        self.assertEqual(demo.request.command_id, 2)


if __name__ == "__main__":
    unittest.main()
