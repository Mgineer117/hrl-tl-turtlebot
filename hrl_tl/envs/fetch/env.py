"""Gymnasium environment extending Fetch Reach with reach-avoid subtasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from gymnasium_robotics.envs.fetch import MujocoFetchEnv

from hrl_tl.envs.fetch.config import FetchReachAvoidConfig
from hrl_tl.envs.fetch.model import get_reach_avoid_xml_path
from hrl_tl.envs.fetch.patch import apply_patch

apply_patch()


def _decode_action(action: np.ndarray, num_bins: int) -> np.ndarray:
    """Convert multi-discrete action indices to continuous [-1, 1] range."""
    return -1.0 + np.asarray(action, dtype=np.float32) * (2.0 / (num_bins - 1))


def _sort_by_distance(displacement_vectors: np.ndarray) -> np.ndarray:
    """Sorts relative displacement vectors in ascending order of Euclidean distance.

    Args:
        displacement_vectors: Array of shape (N, 3) representing 3D displacements.

    Returns:
        Array of shape (N, 3) sorted such that index 0 is closest to the origin.
    """
    if displacement_vectors.shape[0] <= 1:
        return displacement_vectors
    distances = np.linalg.norm(displacement_vectors, axis=1)
    return displacement_vectors[np.argsort(distances)]


class FetchReachAvoidEnv(MujocoFetchEnv, EzPickle):
    """An extended Fetch Reach environment with 3D reach-avoid subtasks."""

    def __init__(
        self,
        config: FetchReachAvoidConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            self.reach_avoid_config = FetchReachAvoidConfig()
        elif isinstance(config, FetchReachAvoidConfig):
            self.reach_avoid_config = config
        else:
            self.reach_avoid_config = FetchReachAvoidConfig(**config)

        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        self._init_task_state()
        self.xml_path = get_reach_avoid_xml_path()
        self._initialized = False
        super().__init__(
            model_path=self.xml_path,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=self.reach_avoid_config.zone_config.zone_size,
            initial_qpos=initial_qpos,
            reward_type="sparse",
            **kwargs,
        )
        EzPickle.__init__(self, config=config, **kwargs)
        self._initialized = True
        if hasattr(self, "goal"):
            delattr(self, "goal")
        self._setup_spaces()

    @property
    def yellow_positions(self) -> np.ndarray:
        return self.yellow_pos

    @property
    def red_positions(self) -> np.ndarray:
        return self.red_pos

    @property
    def current_subtask(self) -> int:
        return self.current_subtask_idx

    @current_subtask.setter
    def current_subtask(self, value: int) -> None:
        self.current_subtask_idx = value

    def set_agent_pos(self, pos: np.ndarray) -> None:
        """Teleports gripper mocap to a given 3D position."""
        target = pos[0] if pos.ndim > 1 else pos
        self._utils.set_mocap_pos(
            self.model, self.data, "robot0:mocap", np.asarray(target)
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def _setup_spaces(self) -> None:
        """Initialize action and observation spaces."""
        act_cfg = self.reach_avoid_config.action_config
        bins = act_cfg.num_bins
        self.action_space = (
            spaces.MultiDiscrete([bins, bins, bins])
            if act_cfg.action_mode == "multidiscrete_3d"
            else spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        )
        subtask_count = len(self.reach_avoid_config.subtask_seq)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-np.inf, np.inf, (3,), np.float64),
                "agent_vel": spaces.Box(-np.inf, np.inf, (3,), np.float64),
                "gripper_state": spaces.Box(-np.inf, np.inf, (2,), np.float64),
                "gripper_vel": spaces.Box(-np.inf, np.inf, (2,), np.float64),
                "yellow_dist": spaces.Box(-np.inf, np.inf, (2, 3), np.float64),
                "red_dist": spaces.Box(-np.inf, np.inf, (3, 3), np.float64),
                "white_dist": spaces.Box(-np.inf, np.inf, (1, 3), np.float64),
                "visit_counts": spaces.Box(0, np.inf, (3,), np.int32),
                "subtask": spaces.Box(0, max(subtask_count, 1), (1,), np.int32),
                "observation": spaces.Box(-np.inf, np.inf, (10,), np.float64),
            }
        )

    def _init_task_state(self) -> None:
        """Initialize zone coordinates and subtask tracking attributes."""
        z_cfg = self.reach_avoid_config.zone_config
        self.yellow_pos = np.array(z_cfg.yellow_zone[:2], dtype=np.float64)
        self.white_pos = np.array(z_cfg.yellow_zone[2:], dtype=np.float64)
        self.red_pos = np.array(z_cfg.red_zone, dtype=np.float64)
        self.current_subtask_idx = 0
        self.is_success = False
        self.yellow_visits = self.red_visits = self.white_visits = 0
        self._in_yellow = self._in_red = self._in_white = False

    def _set_action(self, action: np.ndarray) -> None:
        act_cfg = self.reach_avoid_config.action_config
        cont_action = (
            _decode_action(action, act_cfg.num_bins)
            if act_cfg.action_mode == "multidiscrete_3d"
            else action
        )
        cont_action = np.clip(cont_action, -1.0, 1.0)
        full_action = np.concatenate([cont_action, [0.0]])
        super()._set_action(full_action)

    def reset_model(self) -> dict[str, np.ndarray]:
        self.current_subtask_idx = 0
        self.is_success = False
        self.yellow_visits = self.red_visits = self.white_visits = 0
        self._in_yellow = self._in_red = self._in_white = False

        candidates = np.array(
            self.reach_avoid_config.zone_config.yellow_zone, dtype=np.float64
        )
        white_idx = int(self.np_random.integers(0, len(candidates)))
        yellow_mask = np.ones(len(candidates), dtype=bool)
        yellow_mask[white_idx] = False

        self.white_pos = candidates[white_idx : white_idx + 1]
        self.yellow_pos = candidates[yellow_mask]

        center = np.array(
            self.reach_avoid_config.zone_config.center, dtype=np.float64
        )
        perturb = self.reach_avoid_config.zone_config.agent_perturbation
        spawn_pos = center + self.np_random.uniform(-perturb, perturb, size=3)

        self._utils.set_mocap_pos(
            self.model, self.data, "robot0:mocap", spawn_pos
        )
        self._utils.set_mocap_quat(
            self.model,
            self.data,
            "robot0:mocap",
            np.array([1.0, 0.0, 1.0, 0.0]),
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

        return self._get_obs()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to initial state, bypassing Farama GoalEnv assertion."""
        gym.Env.reset(self, seed=seed, options=options)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self.reset_model()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def _get_obs(self) -> dict[str, np.ndarray]:
        grip_pos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        dt = self.n_substeps * self.model.opt.timestep
        grip_vel = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
            * dt
        )
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        obs_vec = np.concatenate(
            [grip_pos, grip_vel, robot_qpos[-2:], robot_qvel[-2:] * dt]
        )
        obs = {
            "agent_pos": grip_pos,
            "agent_vel": grip_vel,
            "gripper_state": robot_qpos[-2:].copy(),
            "gripper_vel": (robot_qvel[-2:] * dt).copy(),
            "yellow_dist": _sort_by_distance(self.yellow_pos - grip_pos),
            "red_dist": _sort_by_distance(self.red_pos - grip_pos),
            "white_dist": _sort_by_distance(self.white_pos - grip_pos),
            "visit_counts": np.array(
                [self.yellow_visits, self.red_visits, self.white_visits],
                dtype=np.int32,
            ),
            "subtask": np.array([self.current_subtask_idx], dtype=np.int32),
            "observation": obs_vec,
        }
        if not getattr(self, "_initialized", False):
            obs["achieved_goal"] = grip_pos.copy()
            obs["desired_goal"] = np.zeros(3)
        return obs

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        obs = self._get_obs()
        reward, terminated = self._evaluate_transitions(obs["agent_pos"])
        info = self._build_info(terminated)
        return obs, reward, terminated, False, info

    def _evaluate_transitions(self, grip_pos: np.ndarray) -> tuple[float, bool]:
        z_size = self.reach_avoid_config.zone_config.zone_size
        in_yellow = bool(
            np.any(np.linalg.norm(self.yellow_pos - grip_pos, axis=1) < z_size)
        )
        in_red = bool(
            np.any(np.linalg.norm(self.red_pos - grip_pos, axis=1) < z_size)
        )
        in_white = bool(
            np.any(np.linalg.norm(self.white_pos - grip_pos, axis=1) < z_size)
        )

        if in_yellow and not self._in_yellow:
            self.yellow_visits += 1
        if in_red and not self._in_red:
            self.red_visits += 1
        if in_white and not self._in_white:
            self.white_visits += 1
        self._in_yellow, self._in_red, self._in_white = (
            in_yellow,
            in_red,
            in_white,
        )

        reward = -self.reach_avoid_config.reward_config.step_penalty
        terminated = False
        subtasks = self.reach_avoid_config.subtask_seq

        if self.current_subtask_idx < len(subtasks):
            subtask = subtasks[self.current_subtask_idx]
            if subtask.obstacle == "red" and in_red:
                reward += subtask.penalty
                if subtask.obstacle_absorbing:
                    terminated = True

            if not terminated:
                goal_entered = (subtask.goal == "yellow" and in_yellow) or (
                    subtask.goal == "white" and in_white
                )
                if goal_entered:
                    reward += subtask.reward
                    is_last = self.current_subtask_idx == (len(subtasks) - 1)
                    if is_last:
                        self.is_success = True
                    if subtask.goal_absorbing:
                        terminated = True
                    else:
                        self.current_subtask_idx += 1

        return reward, terminated

    def _build_info(self, terminated: bool) -> dict[str, Any]:
        grip_pos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        )
        d_y = (
            float(np.min(np.linalg.norm(self.yellow_pos - grip_pos, axis=1)))
            if len(self.yellow_pos)
            else np.inf
        )
        d_r = (
            float(np.min(np.linalg.norm(self.red_pos - grip_pos, axis=1)))
            if len(self.red_pos)
            else np.inf
        )
        d_w = (
            float(np.min(np.linalg.norm(self.white_pos - grip_pos, axis=1)))
            if len(self.white_pos)
            else np.inf
        )
        return {
            "is_success": self.is_success,
            "terminated": terminated,
            "subtask": self.current_subtask_idx,
            "distances": {"yellow": d_y, "red": d_r, "white": d_w},
            "visit_counts": {
                "yellow": self.yellow_visits,
                "red": self.red_visits,
                "white": self.white_visits,
            },
        }

    def _render_callback(self) -> None:
        offset = (self.data.site_xpos - self.model.site_pos)[0]
        z_size = self.reach_avoid_config.zone_config.zone_size
        size_arr = np.array([z_size, z_size, z_size], dtype=np.float64)

        for idx, pos in enumerate(self.yellow_pos):
            sid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, f"yellow{idx}"
            )
            if sid != -1:
                self.model.site_pos[sid] = pos - offset
                self.model.site_size[sid] = size_arr
        if len(self.white_pos) > 0:
            sid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, "white0"
            )
            if sid != -1:
                self.model.site_pos[sid] = self.white_pos[0] - offset
                self.model.site_size[sid] = size_arr
        for idx, pos in enumerate(self.red_pos):
            sid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, f"red{idx}"
            )
            if sid != -1:
                self.model.site_pos[sid] = pos - offset
                self.model.site_size[sid] = size_arr
        self._mujoco.mj_forward(self.model, self.data)

    def render_top_down(self) -> np.ndarray:
        """Render top-down camera view looking directly down at table.

        Returns:
            RGB image array of shape (height, width, 3).
        """
        self._render_callback()
        viewer = self.mujoco_renderer._get_viewer(self.render_mode)
        cam = viewer.cam
        orig_dist, orig_az, orig_el, orig_lookat = (
            cam.distance,
            cam.azimuth,
            cam.elevation,
            cam.lookat.copy(),
        )

        cam.distance = 1.3
        cam.azimuth = self.reach_avoid_config.zone_config.top_down_azimuth
        cam.elevation = -89.9
        center = self.reach_avoid_config.zone_config.center
        cam.lookat[:] = np.array([center[0], center[1], 0.45], dtype=np.float64)

        img = viewer.render(self.render_mode, camera_id=-1)

        cam.distance = orig_dist
        cam.azimuth = orig_az
        cam.elevation = orig_el
        cam.lookat[:] = orig_lookat
        return img

    def render_multiview(self) -> dict[str, np.ndarray]:
        """Render default, top-down, and combined side-by-side frames.

        Returns:
            Dictionary containing 'default', 'top_down', and 'combined' RGB arrays.
        """
        img_default = np.asarray(super().render())
        img_topdown = self.render_top_down()
        combined = np.hstack([img_default, img_topdown])
        return {
            "default": img_default,
            "top_down": img_topdown,
            "combined": combined,
        }

    def render(
        self,
        render_view: Literal["default", "top_down", "combined"] | None = None,
    ) -> np.ndarray:
        """Render environment frame according to configured or requested view.

        Args:
            render_view: Override for view mode ('default', 'top_down', or 'combined').

        Returns:
            RGB image array of shape (H, W, 3) or (H, 2W, 3) if combined.
        """
        view_mode = (
            render_view or self.reach_avoid_config.zone_config.render_view
        )
        if view_mode == "top_down":
            return self.render_top_down()
        if view_mode == "combined":
            return self.render_multiview()["combined"]
        return super().render()
