# Zone policy on TurtleBot3 Burger

This is the small ROS 2 / Qualisys pipeline for one physical robot. It uses the sibling `hrl-tl-zone-sim` repo for the supplied 8.j.b meta policy, 8.i.b primitive policy, Zone observation code, and checkpoints. It contains the physical arena and an adapted Qualisys publisher from `mrs2025-main.zip`.

`mocap.py` receives a QTM rigid body and publishes `/qualysis/tb3_1` (`PoseStamped`, metres, yaw in `orientation.z`, frame `mocap`). `run.py` converts that pose to the trained Zone coordinates, updates the Zone observation and yellow → white task state, runs the meta and primitive policies, converts the selected 8×5 primitive action to a world waypoint, and publishes bounded ROS velocity commands to `/cmd_vel`. It waits for a fresh measured stop before choosing another action. The native ContGrid velocity state is retained because the checkpoint was trained with that state; position and zone visits come from Qualisys.

Each primitive has a **5-second slot** (`robot.motion.motion_timeout` in `configs/arena.json`). The controller turns toward the waypoint before driving forward, stops at the target, then publishes zero velocity for the rest of the slot. If it cannot reach and settle by the deadline, it stops with `motion_timeout`; it does not score an unfinished policy step. Increase this setting if the calibrated speed and turn rate make 5 seconds insufficient. The controller also clips targets and stops on measured entry into a wall buffer: `robot.bounds.margin` covers the robot footprint, and `robot.motion.wall_stop_margin` adds 0.05 m. Calibrate both for the actual wall, robot size, tracking error, and braking distance.

## Setup

Use an Ubuntu 24.04 / ROS 2 Jazzy computer that can reach QTM and the Burger. Source ROS 2 and the TurtleBot3 workspace first. Use Python 3.12 with `rclpy` available. Conda is fine if its Python can import the system ROS packages.

```bash
cd /path/to/hrl-tl-turtlebot
python -m pip install -e ../hrl-tl-zone-sim qtm-rt
python -c 'import rclpy, qtm_rt, spot, contgrid; print("imports OK")'
```

On macOS, for the offline `check` only, install a native Spot after pip: `conda install -c conda-forge spot=2.13.2`. The ROS motion run is for the Ubuntu robot computer.

If the simulation repo is elsewhere, pass `--sim-root=/path/to/hrl-tl-zone-sim` to `run.py`.

## Before motion

1. Edit [configs/arena.json](configs/arena.json): match the physical room and start, set `robot.frame` to the measured QTM origin, rotation, and metres-to-Zone scale, and set `robot.bounds`, `robot.ros.heading_offset_rad`, and motion limits. Keep `environment.scenario_config.spawn_config` and `zones` synchronized if any landmark moves. The included arena is the seed-0 example, not a measured lab calibration.
2. Start the Burger's ROS driver, as in `mrs2025-main`. Set the same `ROS_DOMAIN_ID` on the robot and computer. Check `ros2 topic type /cmd_vel`; the included arena selects `geometry_msgs/msg/Twist`, matching `mrs2025-main`. If the Burger reports `TwistStamped`, set `robot.ros.cmd_vel_type` to `twist_stamped`. Check that only the intended robot receives that topic.
3. Start QTM and define rigid body `tb3_1`. In one terminal:

   ```bash
   python mocap.py --ip=<QTM_SERVER_IP> --marker=tb3_1
   ```

4. Confirm `/qualysis/tb3_1` contains fresh, correct x/y/yaw and `mocap` frame. The old publisher is adapted to publish only advancing QTM frames, so lost tracking causes a pose timeout and zero velocity.
5. Run the no-motion checkpoint and state check:

   ```bash
   python run.py check
   ```

   It prints the initial policy observation, selected TL option, primitive action and world waypoint. This uses the configured arena start, not a live QTM pose.

## Physical run

Start with one short action while the robot has a clear path and a physical stop is available:

```bash
python run.py run --max-actions=1
```

After verifying heading, waypoint, and stopping on the real Burger, run the full episode:

```bash
python run.py run --max-actions=250
```

Press Ctrl-C or publish `std_msgs/msg/Bool` with `data: true` to `/robot_demo/stop` to stop. Each run writes `logs/<timestamp>/policy.jsonl` and `motion.jsonl` with observations, actions, targets, measured poses, and stop reasons.

## Source notes

`mrs_qualysis_publisher.py` comes from `mrs2025-main.zip`, with an atomic QTM frame update. `mocap.py` adds a freshness gate and configurable QTM address. The Zone state, waypoint conversion, and ROS feedback controller live in `hrl-tl-zone-sim` and were extracted from `hrl-tl-feat-demo`; they replace the grid-only `MultiRobotSystem`/`PIDController` because its five tile actions do not match this policy's eight directions and five magnitudes. No hardware run has been performed in this repo.
