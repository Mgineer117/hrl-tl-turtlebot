"""Offline checks for marker correction and a Burger drive deadband."""

import math
import unittest

from hrl_tl.robot_demo import config, pose
from hrl_tl.robot_demo.control import runtime


class MotionCalibrationTest(unittest.TestCase):
    def test_stale_mocap_holds_zero_until_fresh_pose(self):
        demo = runtime.DemoRuntime(config.DemoConfig(), 0)
        demo.update_pose(pose.Pose2D(x=0, y=0, yaw=0, stamp=0, received_at=0))
        demo.tick(0)
        self.assertTrue(demo.submit_action(1, pose.Point2D(x=0.2, y=0), 0))
        self.assertEqual(demo.tick(0.6), runtime.motion.Velocity())
        self.assertFalse(demo.finished)
        self.assertIsNone(demo.request)
        demo.update_pose(pose.Pose2D(x=0, y=0, yaw=0, stamp=0.7, received_at=0.7))
        demo.tick(0.7)
        self.assertEqual(demo.request.command_id, 2)

    def test_deadline_replans_but_wall_violation_still_faults(self):
        settings = config.DemoConfig(motion=config.MotionConfig(motion_timeout=1.0))
        demo = runtime.DemoRuntime(settings, 0)
        demo.update_pose(pose.Pose2D(x=0, y=0, yaw=0, stamp=0, received_at=0))
        demo.tick(0)
        self.assertTrue(demo.submit_action(1, pose.Point2D(x=0.2, y=0), 0))
        demo.update_pose(pose.Pose2D(x=0, y=0, yaw=0, stamp=1.05, received_at=1.05))
        self.assertEqual(demo.tick(1.1), runtime.motion.Velocity())
        self.assertFalse(demo.finished)
        self.assertIn("motion_timeout", [
            event["reason"] for event in demo.take_events()
            if event["event"] == "result"
        ])
        demo.update_pose(pose.Pose2D(x=0, y=0, yaw=0, stamp=1.2, received_at=1.2))
        demo.tick(1.2)
        self.assertEqual(demo.request.command_id, 2)
        demo.update_pose(pose.Pose2D(x=1.9, y=0, yaw=0, stamp=1.3, received_at=1.3))
        self.assertTrue(demo.finished)
        self.assertEqual(demo.reason, "wall_clearance")

    def test_first_pose_alignment_keeps_one_fixed_translation(self):
        alignment = pose.FirstPoseAlignment(pose.Point2D(x=-1.0, y=2.0))
        first = pose.Pose2D(x=3.0, y=4.0, yaw=0.2, stamp=1, received_at=1)
        second = pose.Pose2D(x=3.1, y=4.2, yaw=0.3, stamp=2, received_at=2)
        aligned_first = alignment.apply(first)
        aligned_second = alignment.apply(second)
        self.assertEqual((aligned_first.x, aligned_first.y), (-1.0, 2.0))
        self.assertAlmostEqual(aligned_second.x, -0.9)
        self.assertAlmostEqual(aligned_second.y, 2.2)
        self.assertEqual(aligned_second.yaw, second.yaw)
        self.assertEqual(alignment.first_pose, first)

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
