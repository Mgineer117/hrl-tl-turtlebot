# Zone policy on TurtleBot3 Burger

This is the ROS 2 / Qualisys pipeline for one physical robot. A clone contains the `hrl_tl/` source package, the policy wrapper configuration and formulae, and both trained checkpoints in `models/`: `best_model.zip` (primitive) and `final_model_8.j.b_30.0M_rep_2.zip` (meta). It also contains the physical arena and an adapted Qualisys publisher from `mrs2025-main.zip`.

The `.zip` files are Stable-Baselines3 checkpoint archives. Each contains `policy.pth` (PyTorch weights) plus saved algorithm metadata and optimizer state. Keep them zipped: `run.py` loads the meta policy with `PrimitiveStepPPO.load(...)` and the configured primitive policy with `SDSAC.load(...)`.

`mocap.py` connects to QTM at `128.174.245.64` by default and publishes the `tb3_1` rigid body on `/qualysis/tb3_1` (`PoseStamped`, metres, yaw in `orientation.z`, frame `mocap`). `run.py` checks that `192.168.0.77` accepts SSH, a fresh mocap pose arrives, and `/cmd_vel` has a subscriber of the configured type. It corrects the raw QTM pose into arena-world coordinates before building the trained Zone observation, updating the yellow → white task state, running both policies, and commanding a world waypoint through `/cmd_vel`. It waits for a fresh measured stop before choosing another action. The native ContGrid velocity state is retained because the checkpoint was trained with that state; position and zone visits come from the corrected Qualisys pose.

Each primitive has a **30-second slot** (`robot.motion.motion_timeout` in `configs/arena.json`). The controller turns toward the waypoint before driving forward, stops at the target, then publishes zero velocity for the rest of the slot. At the deadline, an unfinished command stops and the policy replans from the next fresh measured pose. Stale mocap also stops the current command and waits for a fresh pose. The controller clips targets and stops on measured entry into a wall buffer: `robot.bounds.margin` covers the robot footprint, and `robot.motion.wall_stop_margin` adds 0.05 m. Calibrate these values for the actual wall, robot size, tracking error, and braking distance.

## Installation and lab run order (ROS domain 40)

Use an Ubuntu 24.04 / ROS 2 Jazzy laptop with network access to QTM and the Burger. Clone this repo alone. Before any motion, calibrate [configs/arena.json](configs/arena.json) to the measured lab: `robot.frame`, `robot.bounds`, `robot.ros.heading_offset_rad`, motion limits, and the zone positions in both `environment.scenario_config.spawn_config` and `zones`. The default seed 383 layout is an example lab placement, not a measured placement. This procedure assumes `192.168.0.77` is the Burger carrying the `tb3_1` marker.

The configured origin correction is `robot.ros.mocap_offset_x_m: 2.34` and `mocap_offset_y_m: -0.011`, based on the reported raw QTM pose `(-2.340, 0.011)` m while the robot is at arena-world `(0, 0)`. `heading_offset_rad` is set to `1.57079632679` (+90°). The turning trace is consistent with that yaw correction and shows the tracked marker moving on a circle about 1.8 cm from the robot's turn center; `marker_offset_x_m` and `marker_offset_y_m` encode the fitted body-frame displacement `(-0.01099, -0.01356)` m from the robot reference point to the marker. The pose transform subtracts the rotated marker displacement before control and policy state updates. These values are fitted to the current logs and should be confirmed with a short, supervised movement before a full policy episode. The optional `mocap_rotation_rad` is currently `0`. The state and motion controller receive the corrected robot-reference point in arena-world coordinates; the policy then receives `Zone_xy = robot.frame.sim_units_per_meter × R(robot.frame.rotation_rad) × (world_xy - robot.frame.origin_xy)`. `robot.frame.origin_xy` is a separate world-to-Zone offset. QTM yaw is corrected by `mocap_rotation_rad + heading_offset_rad`.

With `robot.ros.align_first_pose_to_start: true`, preflight maps its first valid QTM reading to seed 383's configured start at world `(-1.01717, -1.02527)` m. The same fixed translation is used throughout `run` or `move`, including policy observations and wall checks; `motion.jsonl` records it as `mocap_alignment`. **Place the robot physically at that start before preflight**. This translation corrects a coordinate origin mismatch; it cannot correct a robot placed elsewhere. Set the flag to `false` to use only the static mocap offsets above.

### 1. Install from a fresh clone (once)

If `hrl-zone` does not exist on the Ubuntu laptop, create it first with `conda create -n hrl-zone python=3.12 -y`.

```bash
source /opt/ros/jazzy/setup.bash
conda activate hrl-zone
export ROS_DOMAIN_ID=40
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
git clone https://github.com/Mgineer117/hrl-tl-turtlebot.git
cd hrl-tl-turtlebot
python -m pip install -r requirements.txt
python -c 'import rclpy, geometry_msgs.msg, sensor_msgs.msg, std_msgs.msg, qtm_rt, spot, contgrid; print("imports OK")'
python run.py check
```

Stop here unless the import check and offline policy check pass. `check` prints an observation, meta option, primitive action, and waypoint from the configured example start; it does not use live QTM. On macOS, only the offline `check` is supported; the motion run requires the Ubuntu ROS computer.

`requirements.txt` installs this repo's `hrl_tl` package via `-e .`, which also installs PyTorch (`torch`) and Stable-Baselines3 (`stable-baselines3[extra]`) from `pyproject.toml`, plus the Qualisys SDK. Install ROS 2 Jazzy separately; `rclpy`, `geometry_msgs`, `sensor_msgs`, and `std_msgs` come from ROS and must be importable by the selected Conda Python.

### 2. Start the robot driver (robot terminal)

```bash
ssh ubuntu@192.168.0.77
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=40
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export TURTLEBOT3_MODEL=burger
ros2 pkg prefix turtlebot3_bringup
ros2 launch turtlebot3_bringup robot.launch.py use_sim_time:=False
```

Leave this terminal running. If `ros2 pkg prefix` cannot find the package, source the TurtleBot3 workspace on the robot and retry. The launch command follows `mrs2025-main`.

### 3. Publish Qualisys poses (new laptop terminal)

Start QTM with rigid body `tb3_1`, then run:

```bash
source /opt/ros/jazzy/setup.bash
conda activate hrl-zone
export ROS_DOMAIN_ID=40
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
cd /path/to/hrl-tl-turtlebot
python mocap.py --ip=128.174.245.64 --marker=tb3_1
```

Leave this terminal running. The script publishes only advancing QTM frames on `/qualysis/tb3_1`.

### 4. Verify, then command one action (new laptop terminal)

```bash
source /opt/ros/jazzy/setup.bash
conda activate hrl-zone
export ROS_DOMAIN_ID=40
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
cd /path/to/hrl-tl-turtlebot
ros2 topic list
ros2 topic type /qualysis/tb3_1
ros2 topic echo /qualysis/tb3_1 --once
ros2 topic type /cmd_vel
ros2 topic info /cmd_vel --verbose
python run.py preflight --robot-ip=192.168.0.77
python run.py move --angle-deg=90 --distance-m=0.015
```

The list must include `/qualysis/tb3_1` and `/cmd_vel`; the raw pose must be fresh, in metres, and in frame `mocap`. The included physical arena expects `geometry_msgs/msg/TwistStamped` on `/cmd_vel`; preflight checks the actual topic type and subscriber before motion. Confirm that the `/cmd_vel` subscriber is the intended robot. `preflight` checks SSH reachability, a live pose, command topic type, and a subscriber without moving; it prints raw QTM, corrected world, and Zone x/y and rejects a corrected pose outside the wall buffer. Compare those printed coordinates with the robot's known physical location before moving.

`run.py move` executes exactly one manually selected discrete movement through the same `MovementAction` interpreter and feedback controller as policy actions. The angle is a Zone-frame direction from +x, in 45° increments; it is not relative to the robot's current yaw. Distance must match one of `robot.step_lengths_sim / robot.frame.sim_units_per_meter` within 1 mm. For the included arena, `--angle-deg=90 --distance-m=0.064` means direction index 2 and magnitude index 4. The command preflights first and then moves the robot, so use a clear test area and start with the shortest configured distance when checking the interpreter.

After checking the measured heading, path, and stop behavior from that action, run the episode in the same terminal:

```bash
python run.py run --robot-ip=192.168.0.77
```

Press Ctrl-C or publish `std_msgs/msg/Bool` with `data: true` to `/robot_demo/stop` to stop. Trained-policy runs write `logs/<timestamp>/policy.jsonl` and `motion.jsonl`; manual one-action runs write `manual_action.json` and `motion.jsonl`. `motion.jsonl` has a trajectory sample at each control tick with raw QTM and corrected poses, controller state and velocity, active action and target, policy option/decision, and native observation. `policy.jsonl` stores each action decision and its completed transition, including next observation, reward, task info, and terminal flags. The final transition is recorded even when that step succeeds or fails the task.

`ros_adapter.decode_pose` applies `mocap_to_world` once to each raw QTM pose. `PrimitiveZone.start/complete` in this repo's `hrl_tl/` package transforms that corrected world pose to the trained Zone coordinates and constructs the observation from the saved environment, including zone and wall distances. `RobotHierarchy.next_action` runs both checkpoints. `PrimitiveZone.plan` maps direction (0–7) and magnitude (0–4) to a short world waypoint in the same corrected world frame. The policy indices are not direct velocity commands.

## Action interpreter and modes

The environment's `discrete_ang_directional` action mode gives the primitive policy `MultiDiscrete([8, 5])`: direction index `0..7` means `0°, 45°, ..., 315°` in Zone coordinates; magnitude index `0..4` selects native magnitudes `1.0, 1.8, 2.6, 3.4, 4.2`. The robot maps those magnitudes through `robot.step_lengths_sim = [0.05, 0.09, 0.13, 0.17, 0.21]` and `robot.frame.sim_units_per_meter` to waypoint distances. The separate meta policy selects a temporal-logic option and CPC parameters.

`MotionExecutor` uses a feedback state machine and **proportional** gains; it has no derivative term in the velocity command. In `rotating`, it commands only yaw rate, `clip(angular_gain × heading_error, ±max_angular_speed)`, until the heading error is within `heading_tolerance`. In `moving`, it commands forward speed `min(max_linear_speed, max(min_linear_speed, linear_gain × distance_to_target))` and keeps correcting yaw. The minimum speed avoids stalling below the Burger's drive deadband near a target. If the heading error exceeds `reorient_threshold`, it returns to `rotating`. At the target it enters `settling`, waits for measured linear and angular motion to stop for `settle_duration`, then enters `holding` and sends zero velocity until the 30-second slot ends. An unfinished command or stale mocap stops the command, then the policy replans after a fresh pose. Wall-buffer violations remain terminal faults. The 0.05 m `turn_clearance` lets the controller make space before a close turn.

For example, `[2, 4]` requests direction 90° and a `0.21` Zone-unit step, which is `0.064` m with the included scale. The robot first turns toward the resulting world waypoint, then drives toward it under mocap feedback. Edit `robot.motion.motion_timeout` to change the per-action slot; `robot.motion.action_timeout` separately limits policy inference time.

The command modes in `run.py` are: `check` (load both checkpoints and predict once from the configured start, without ROS or motion), `preflight` (verify SSH reachability, a live mocap pose, and a `/cmd_vel` subscriber, without motion), `move` (preflight and execute one manually selected discrete action), and `run` (continue policy inference and robot movement until the user stops it or a safety fault occurs). On task completion or the native episode horizon, `run` restarts policy state at the current measured robot pose. `--max-actions` is accepted for compatibility but does not cap `run`.

## Seed 383 arena and offline trajectory

The default [physical arena](configs/arena.json) and [offline arena](configs/robot_demo/arenas/seed_383.json) use the same fixed seed 383 task, start, zones, and walls. The physical arena retains the measured Qualisys transform, 0.02 m/s minimum drive speed, and the previously accepted 0.01 m target tolerance. The offline arena uses an identity Mocap transform and the guide's 0.003048 m target tolerance. Both use a 30 s action slot. The original saved JSON was not supplied, so other fields use this repo's settings. For a live run, place the robot at the listed world start facing +x; `run.py run` reads its actual Qualisys pose and does not move it to the saved start automatically.

| Object | Zone (x, y) | World (x, y), m |
|---|---|---|
| Robot start, yaw 0 | (2.1628, 2.1362) | (-1.0172, -1.0253) |
| Yellow 1 | (1.5, 5.5) | (-1.2192, 0) |
| Yellow 2 | (9.5, 5.5) | (1.2192, 0) |
| Yellow 3 | (5.5, 1.5) | (0, -1.2192) |
| White | (5.5, 9.5) | (0, 1.2192) |
| Red 1 | (5.5, 5.5) | (0, 0) |
| Red 2 | (3.5, 3.5) | (-0.6096, -0.6096) |
| Red 3 | (7.5, 3.5) | (0.6096, -0.6096) |
| Red 4 | (3.5, 7.5) | (-0.6096, 0.6096) |
| Red 5 | (7.5, 7.5) | (0.6096, 0.6096) |
| Black 1 | (7, 8) | (0.4572, 0.7620) |
| Black 2 | (4, 10) | (-0.4572, 1.3716) |

One Zone unit is 0.3048 m. All zones have radius 0.25 Zone units (0.0762 m). The inner wall faces are at x/y = ±1.524 m. Run `python simulate_policy.py` to regenerate the [offline seed 383 plot](artifacts/fixed_seed/seed_383_trajectory.png) and its adjacent JSON trajectory. The saved ideal-motion rollout reached `task_success` in 140 actions; it is not a Gazebo or robot run.

## Source notes

`mrs_qualysis_publisher.py` comes from `mrs2025-main.zip`, with an atomic QTM frame update. `mocap.py` adds a freshness gate and configurable QTM address. The bundled `hrl_tl/` code was copied from `hrl-tl-zone-sim`, which extracted the Zone state, waypoint conversion, and ROS feedback controller from `hrl-tl-feat-demo`. The seed 383 configuration has not been validated on the physical robot.
