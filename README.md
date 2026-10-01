# Zone policy on TurtleBot3 Burger

This is the small ROS 2 / Qualisys pipeline for one physical robot. Its `models/` folder contains both trained checkpoints: `best_model.zip` (primitive policy) and `final_model_8.j.b_30.0M_rep_2.zip` (meta policy). It still installs the sibling `hrl-tl-zone-sim` package for the trained Zone environment, observation logic, and TL wrappers. It contains the physical arena and an adapted Qualisys publisher from `mrs2025-main.zip`.

`mocap.py` connects to QTM at `128.174.245.64` by default and publishes the `tb3_1` rigid body on `/qualysis/tb3_1` (`PoseStamped`, metres, yaw in `orientation.z`, frame `mocap`). `run.py` checks that `192.168.0.77` accepts SSH, a fresh mocap pose arrives, and `/cmd_vel` has a subscriber of the configured type. It corrects the raw QTM pose into arena-world coordinates before building the trained Zone observation, updating the yellow → white task state, running both policies, and commanding a world waypoint through `/cmd_vel`. It waits for a fresh measured stop before choosing another action. The native ContGrid velocity state is retained because the checkpoint was trained with that state; position and zone visits come from the corrected Qualisys pose.

Each primitive has a **10-second slot** (`robot.motion.motion_timeout` in `configs/arena.json`). The controller turns toward the waypoint before driving forward, stops at the target, then publishes zero velocity for the rest of the slot. If it cannot reach and settle by the deadline, it stops with `motion_timeout`; it does not score an unfinished policy step. Adjust this setting if the calibrated speed and turn rate require a different duration. The controller also clips targets and stops on measured entry into a wall buffer: `robot.bounds.margin` covers the robot footprint, and `robot.motion.wall_stop_margin` adds 0.05 m. Calibrate both for the actual wall, robot size, tracking error, and braking distance.

## Lab run order (ROS domain 40)

Use an Ubuntu 24.04 / ROS 2 Jazzy laptop with network access to QTM and the Burger. Keep `hrl-tl-turtlebot` and `hrl-tl-zone-sim` side by side. Before any motion, calibrate [configs/arena.json](configs/arena.json) to the measured lab: `robot.frame`, `robot.bounds`, `robot.ros.heading_offset_rad`, motion limits, and the zone positions in both `environment.scenario_config.spawn_config` and `zones`. The supplied seed-0 layout is an example, not a measured lab calibration. This procedure assumes `192.168.0.77` is the Burger carrying the `tb3_1` marker.

The requested origin correction is `robot.ros.mocap_offset_x_m: 2.1336` (**+7 ft**) and `mocap_offset_y_m: 0`. The optional `mocap_rotation_rad` is currently `0`. The state and motion controller both receive `world_xy = R(mocap_rotation_rad) × raw_QTM_xy + mocap_offset_xy`; the policy then receives `Zone_xy = robot.frame.sim_units_per_meter × R(robot.frame.rotation_rad) × (world_xy - robot.frame.origin_xy)`. `robot.frame.origin_xy` is a separate world-to-Zone offset; do not add the +7 ft there again. QTM yaw is corrected by `mocap_rotation_rad + heading_offset_rad`.

### 1. Prepare the laptop (once)

If `hrl-zone` does not exist on the Ubuntu laptop, create it first with `conda create -n hrl-zone python=3.12 -y`.

```bash
source /opt/ros/jazzy/setup.bash
conda activate hrl-zone
export ROS_DOMAIN_ID=40
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
cd /path/to/hrl-tl-turtlebot
python -m pip install -e ../hrl-tl-zone-sim qtm-rt
python -c 'import rclpy, qtm_rt, spot, contgrid; print("imports OK")'
python run.py check
```

Stop here unless the import check and offline policy check pass. `check` prints an observation, meta option, primitive action, and waypoint from the configured example start; it does not use live QTM. If the simulation repo is elsewhere, pass `--sim-root=/path/to/hrl-tl-zone-sim` to `run.py`. On macOS, only the offline `check` is supported; the motion run requires the Ubuntu ROS computer.

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
python run.py run --robot-ip=192.168.0.77 --max-actions=1
```

The list must include `/qualysis/tb3_1` and `/cmd_vel`; the raw pose must be fresh, in metres, and in frame `mocap`. The included arena expects `geometry_msgs/msg/Twist` on `/cmd_vel`. If the robot reports `TwistStamped`, set `robot.ros.cmd_vel_type` to `twist_stamped` in the arena JSON before preflight. Confirm that the `/cmd_vel` subscriber is the intended robot. `preflight` checks SSH reachability, a live pose, command topic type, and a subscriber without moving; it prints raw QTM, corrected world, and Zone x/y and rejects a corrected pose outside the wall buffer. Compare those printed coordinates with the robot's known physical location before the one-action command. The one-action command is the first motion command.

After checking the measured heading, path, and stop behavior from that action, run the episode in the same terminal:

```bash
python run.py run --robot-ip=192.168.0.77 --max-actions=250
```

Press Ctrl-C or publish `std_msgs/msg/Bool` with `data: true` to `/robot_demo/stop` to stop. Each run writes `logs/<timestamp>/policy.jsonl` and `motion.jsonl` with observations, actions, targets, measured poses, and stop reasons.

`ros_adapter.decode_pose` applies `mocap_to_world` once to each raw QTM pose. `PrimitiveZone.start/complete` in the sibling sim repo transforms that corrected world pose to the trained Zone coordinates and constructs the observation from the saved environment, including zone and wall distances. `RobotHierarchy.next_action` runs both checkpoints. `PrimitiveZone.plan` maps direction (0–7) and magnitude (0–4) to a short world waypoint in the same corrected world frame. The policy indices are not direct velocity commands.

## Action interpreter and modes

The environment's `discrete_ang_directional` action mode gives the primitive policy `MultiDiscrete([8, 5])`: direction index `0..7` means `0°, 45°, ..., 315°` in Zone coordinates; magnitude index `0..4` selects native magnitudes `1.0, 1.8, 2.6, 3.4, 4.2`. The robot maps those magnitudes through `robot.step_lengths_sim = [0.05, 0.09, 0.13, 0.17, 0.21]` and `robot.frame.sim_units_per_meter` to waypoint distances. The separate meta policy selects a temporal-logic option and CPC parameters.

`MotionExecutor` uses a feedback state machine and **proportional** gains; it has no derivative term in the velocity command. In `rotating`, it commands only yaw rate, `clip(angular_gain × heading_error, ±max_angular_speed)`, until the heading error is within `heading_tolerance`. In `moving`, it commands forward speed `min(max_linear_speed, linear_gain × distance_to_target)` and keeps correcting yaw. If the heading error exceeds `reorient_threshold`, it returns to `rotating`. At the target it enters `settling`, waits for measured linear and angular motion to stop for `settle_duration`, then enters `holding` and sends zero velocity until the 10-second slot ends. An unreached target, stale mocap, or wall-buffer violation enters `fault` and sends zero velocity. The optional `repositioning` state for making space to turn is inactive here because `turn_clearance` is `0`.

For example, `[2, 4]` requests direction 90° and a `0.21` Zone-unit step, which is `0.064` m with the included scale. The robot first turns toward the resulting world waypoint, then drives toward it under mocap feedback. Edit `robot.motion.motion_timeout` to change the per-action slot; `robot.motion.action_timeout` separately limits policy inference time.

The command modes in `run.py` are: `check` (load both checkpoints and predict once from the configured start, without ROS or motion), `preflight` (verify SSH reachability, a live mocap pose, and a `/cmd_vel` subscriber, without motion), and `run` (repeat policy inference and robot movement until the task or action limit ends).

## Source notes

`mrs_qualysis_publisher.py` comes from `mrs2025-main.zip`, with an atomic QTM frame update. `mocap.py` adds a freshness gate and configurable QTM address. The Zone state, waypoint conversion, and ROS feedback controller live in `hrl-tl-zone-sim` and were extracted from `hrl-tl-feat-demo`; they replace the grid-only `MultiRobotSystem`/`PIDController` because its five tile actions do not match this policy's eight directions and five magnitudes. No hardware run has been performed in this repo.
