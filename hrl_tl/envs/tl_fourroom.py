import os
from typing import Any, TypedDict

import imageio
import numpy as np
from gym_multigrid.envs.rooms import RoomsEnv
from gym_multigrid.typing import Position
from gym_multigrid.utils.map import distance_area_point, distance_points
from gym_tl_tools import (
    BaseVarValueInfoGenerator,
    TLObservationReward,
    replace_special_characters,
)
from gymnasium import Env, Wrapper
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


class FourRoomVarValueInfoGenerator(
    BaseVarValueInfoGenerator[NDArray[np.int64], np.int64]
):
    def get_var_values(
        self,
        env: Env[NDArray[np.int64], np.int64]
        | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64],
        obs: NDArray[np.int64],
        info: dict[str, Any],
    ) -> dict[str, Any]:
        left_doorway: Position = (2, 6)
        bottom_doorway: Position = (6, 10)
        top_doorway: Position = (6, 3)
        right_doorway: Position = (9, 7)

        bl_pocket: Position = (1, 8)
        tl_pocket: Position = (4, 2)

        match env:
            case RoomsEnv():
                # For RoomsEnv, we can directly access the positions
                lavas: list[Position] = env.lava_pos
                holes: list[Position] = env.hole_pos
                goal: Position = env.goal_pos
                agent: Position = env.agents[0].pos
            case Wrapper():
                # For wrapped environments, we need to extract the positions from the observation
                lavas: list[Position] = env.unwrapped.lava_pos
                holes: list[Position] = env.unwrapped.hole_pos
                goal: Position = env.unwrapped.goal_pos
                agent: Position = env.unwrapped.agents[0].pos
            case _:
                raise ValueError("Unsupported environment type")

        d_ld: float = distance_points(agent, left_doorway)
        d_bd: float = distance_points(agent, bottom_doorway)
        d_td: float = distance_points(agent, top_doorway)
        d_rd: float = distance_points(agent, right_doorway)
        d_gl: float = distance_points(agent, goal)
        d_lv: float = distance_area_point(agent, lavas)
        d_hl: float = distance_area_point(agent, holes)

        d_blp: float = distance_points(agent, bl_pocket)
        d_tlp: float = distance_points(agent, tl_pocket)

        return {
            "d_ld": d_ld,
            "d_bd": d_bd,
            "d_td": d_td,
            "d_rd": d_rd,
            "d_gl": d_gl,
            "d_lv": d_lv,
            "d_hl": d_hl,
            "d_blp": d_blp,
            "d_tlp": d_tlp,
        }


class ContRoomsVarValueInfoGenerator(
    BaseVarValueInfoGenerator[NDArray[np.int64], np.int64]
):
    """
    For ContGrid's RoomsEnv,
    """

    def get_var_values(
        self,
        env: Env[NDArray[np.int64], np.int64]
        | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64],
        obs: NDArray[np.int64],
        info: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        env : Env or Wrapper
            The environment or wrapped environment.
        obs : NDArray[np.int64]
            The current observation.
        info : dict[str, Any]
            Additional information from the environment.
            Given in the following format (example):
            ```
            {
                'distances': {
                    'bd': np.float64(2.852197716112439),
                    'goal': np.float64(7.458885942468939),
                    'hole': np.float64(1.6324200222078775),
                    'lava': np.float64(1.6366731309271614),
                    'ld': np.float64(3.202200037510091),
                    'rd': np.float64(5.939015094160368),
                    'td': np.float64(6.453594603889719)
                    },
                'terminated': False
            }
            ```
        Returns
        -------
        var_values : dict[str, Any]
            A dictionary containing the variable values.
            The dictionary contains the following keys:
            - 'd_ld': Distance to the left doorway.
            - 'd_bd': Distance to the bottom doorway.
            - 'd_td': Distance to the top doorway.
            - 'd_rd': Distance to the right doorway.
            - 'd_gl': Distance to the goal.
            - 'd_lv': Distance to the nearest lava.
            - 'd_hl': Distance to the nearest hole.

        """

        return {
            "d_ld": info["distances"]["ld"],
            "d_bd": info["distances"]["bd"],
            "d_td": info["distances"]["td"],
            "d_rd": info["distances"]["rd"],
            "d_gl": info["distances"]["goal"],
            "d_lv": info["distances"]["lava"],
            "d_hl": info["distances"]["hole"],
            "d_lv_c": info["distances"]["closest_lava"],
            "d_hl_c": info["distances"]["closest_hole"],
        }


class ContRoomsObsVarValueInfoGenerator(
    BaseVarValueInfoGenerator[dict[str, NDArray[np.int64]], np.int64]
):
    """
    For ContGrid's RoomsEnv,
    """

    def get_var_values(
        self,
        env: Env[dict[str, NDArray[np.int64]], np.int64]
        | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64]
        | None,
        obs: dict[str, NDArray[np.int64]],
        info: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        env : Env or Wrapper
            The environment or wrapped environment.
        obs : NDArray[np.int64]
            The current observation.
            It's given in the following format (example):
            ```
            {'agent_pos': array([3., 3.]),
            'aut_state': 1,
            'doorway_dist': array([3.16227766, 3.16227766, 6.32455532, 6.70820393]),
            'doorway_pos': array([[ 3., -1.],
                [-1.,  3.],
                [ 6.,  2.],
                [ 3.,  6.]]),
            'goal_dist': array([7.81024968]),
            'goal_pos': array([6., 5.]),
            'hole_dist': array([2.]),
            'hole_pos': array([[ 5.,  6.],
                [ 6.,  4.],
                [ 2.,  5.],
                [ 1.,  6.],
                [-2.,  2.],
                [ 2.,  0.],
                [ 4.,  1.],
                [ 6., -1.]]),
            'lava_dist': array([1.41421356]),
            'lava_pos': array([[ 4.,  5.],
                [ 6.,  6.],
                [ 2.,  7.],
                [ 0.,  4.],
                [-1.,  1.],
                [ 0.,  2.],
                [ 7.,  1.],
                [ 5.,  0.]]),
            'wall_dist': array([3.])}
            ```
        info : dict[str, Any] = {}
            Additional information from the environment.
            Assume it's always empty.

        Returns
        -------
        var_values : dict[str, Any]
            A dictionary containing the variable values.
            The dictionary contains the following keys:
            - 'd_ld': Distance to the left doorway.
            - 'd_bd': Distance to the bottom doorway.
            - 'd_td': Distance to the top doorway.
            - 'd_rd': Distance to the right doorway.
            - 'd_gl': Distance to the goal.
            - 'd_lv': Distance to the nearest lava.
            - 'd_hl': Distance to the nearest hole.

        """

        room_scale: float = 13

        d_ld: float = (
            obs["doorway_dist"].flatten()[0]
            if "doorway_dist" in obs
            else float(np.linalg.norm(obs["doorway_pos"][0] * room_scale))
        )
        d_td: float = (
            obs["doorway_dist"].flatten()[1]
            if "doorway_dist" in obs
            else float(np.linalg.norm(obs["doorway_pos"][1] * room_scale))
        )
        d_rd: float = (
            obs["doorway_dist"].flatten()[2]
            if "doorway_dist" in obs
            else float(np.linalg.norm(obs["doorway_pos"][2] * room_scale))
        )
        d_bd: float = (
            obs["doorway_dist"].flatten()[3]
            if "doorway_dist" in obs
            else float(np.linalg.norm(obs["doorway_pos"][3] * room_scale))
        )
        d_gl: float = (
            obs["goal_dist"].flatten()[0]
            if "goal_dist" in obs
            else float(np.linalg.norm(obs["goal_pos"] * room_scale))
        )
        d_lv: float = (
            obs["lava_dist"].flatten()[0]
            if "lava_dist" in obs
            else float(
                np.min(np.linalg.norm(obs["lava_pos"] * room_scale, axis=1))
            )
        )
        d_hl: float = (
            obs["hole_dist"].flatten()[0]
            if "hole_dist" in obs
            else float(
                np.min(np.linalg.norm(obs["hole_pos"] * room_scale, axis=1))
            )
        )

        return {
            "d_ld": d_ld,
            "d_td": d_td,
            "d_rd": d_rd,
            "d_bd": d_bd,
            "d_gl": d_gl,
            "d_lv": d_lv,
            "d_hl": d_hl,
        }


def var_value_info_generator(
    env: Env[NDArray[np.int64], np.int64]
    | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64],
    obs: NDArray[np.int64],
    info: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate variable value information for the FourRoom environment.
    """
    left_doorway: Position = (2, 6)
    bottom_doorway: Position = (6, 10)
    top_doorway: Position = (6, 3)
    right_doorway: Position = (9, 7)

    match env:
        case RoomsEnv():
            # For RoomsEnv, we can directly access the positions
            lavas: list[Position] = env.lava_pos
            holes: list[Position] = env.hole_pos
            goal: Position = env.goal_pos
            agent: Position = env.agents[0].pos
        case Wrapper():
            # For wrapped environments, we need to extract the positions from the observation
            lavas: list[Position] = env.unwrapped.lava_pos
            holes: list[Position] = env.unwrapped.hole_pos
            goal: Position = env.unwrapped.goal_pos
            agent: Position = env.unwrapped.agents[0].pos
        case _:
            raise ValueError("Unsupported environment type")

    d_ld: float = distance_points(agent, left_doorway)
    d_bd: float = distance_points(agent, bottom_doorway)
    d_td: float = distance_points(agent, top_doorway)
    d_rd: float = distance_points(agent, right_doorway)
    d_gl: float = distance_points(agent, goal)
    d_lv: float = distance_area_point(agent, lavas)
    d_hl: float = distance_area_point(agent, holes)

    return {
        "d_ld": d_ld,
        "d_bd": d_bd,
        "d_td": d_td,
        "d_rd": d_rd,
        "d_gl": d_gl,
        "d_lv": d_lv,
        "d_hl": d_hl,
    }


class PolicyArgsDict(TypedDict):
    """
    A dictionary to hold the arguments for the low-level policy.
    This can be extended with additional parameters as needed.
    """

    algorithm: type[BaseAlgorithm]
    algo_config: dict[str, Any]
    model_save_dir: str
    model_prefix: str
    model_name: str
    training_config: dict[str, Any]
    device: str


class TrainingConfig(BaseModel):
    total_timesteps: int = 50_000
    n_envs: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PolicyArgs(BaseModel):
    """
    A Pydantic model to hold the arguments for the low-level policy.
    This can be extended with additional parameters as needed.
    """

    algorithm: type[BaseAlgorithm] = PPO
    algo_config: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "learning_rate": 0.0003,
        "n_steps": 1000,
        "batch_size": 1000,
        "n_epochs": 40,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": None,
        "stats_window_size": 100,
        "policy_kwargs": {"net_arch": [128, 128]},
    }
    model_save_dir: str = "out/maze/ltl_ll/ll_policies"
    model_prefix: str = "maze_tl_ppo_stay_"
    model_name: str = "final_model"
    training_config: TrainingConfig = TrainingConfig()
    device: str = "cuda:0"

    model_config = ConfigDict(arbitrary_types_allowed=True)


def maze_low_level_policy(
    obs: NDArray[np.int64],
    aut_state: int,
    low_level_env: TLObservationReward[NDArray[np.int64], np.int64],
    args: PolicyArgsDict = {
        "algorithm": PPO,
        "algo_config": {
            "policy": "MultiInputPolicy",
            "learning_rate": 0.0003,
            "n_steps": 1000,
            "batch_size": 1000,
            "n_epochs": 40,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "rollout_buffer_class": None,
            "rollout_buffer_kwargs": None,
            "target_kl": None,
            "stats_window_size": 100,
            "policy_kwargs": {"net_arch": [128, 128]},
        },
        "model_save_dir": "out/maze/ltl_ll/ll_policies",
        "model_prefix": "maze_tl_ppo_stay_",
        "model_name": "final_model",
        "training_config": {
            "total_timesteps": 50_000,
            "n_envs": 10,
        },
        "device": "cuda:0",
    },
) -> np.int64:
    """
    A simple low-level policy for the FourRoom environment.
    This policy is a placeholder and should be replaced with a proper implementation.
    """
    # For simplicity, we return a random action
    policy_args: PolicyArgs = PolicyArgs.model_validate(args)
    policy_args.algo_config["n_steps"] = int(
        policy_args.algo_config["batch_size"]
        / policy_args.training_config.n_envs
    )
    tl_spec_name: str = policy_args.model_prefix + replace_special_characters(
        low_level_env.automaton.tl_spec
    )
    model_path = os.path.join(
        policy_args.model_save_dir,
        tl_spec_name,
        policy_args.model_name + ".zip",
    )
    # If the model exists, load it; otherwise, train a new one
    if os.path.exists(model_path):
        # with (
        #     open(os.devnull, "w") as devnull,
        #     contextlib.redirect_stdout(devnull),
        #     contextlib.redirect_stderr(devnull),
        # ):
        model = policy_args.algorithm.load(
            model_path, env=low_level_env, device=policy_args.device
        )
    else:
        os.makedirs(policy_args.model_save_dir, exist_ok=True)
        vec_env: VecEnv = SubprocVecEnv(
            [lambda: low_level_env] * policy_args.training_config.n_envs
        )
        model = policy_args.algorithm(
            **policy_args.algo_config,
            tensorboard_log=os.path.join(
                policy_args.model_save_dir, tl_spec_name, "tb"
            ),
            env=vec_env,
            verbose=1,
            device=policy_args.device,
        )
        model.learn(total_timesteps=policy_args.training_config.total_timesteps)
        model.save(model_path.replace(".zip", ""))

        video_save_path: str = os.path.join(
            policy_args.model_save_dir, tl_spec_name + ".gif"
        )

        rep_obs, _ = low_level_env.reset()
        terminated: bool = False
        truncated: bool = False
        frames = [low_level_env.render()]
        while not (terminated or truncated):
            rep_action, _ = model.predict(rep_obs)  # type: ignore
            # Ensure action is a numpy int64 scalar
            rep_obs, reward, terminated, truncated, info = low_level_env.step(
                np.int64(rep_action)
            )
            frame = low_level_env.render()
            frames.append(frame)

        imageio.mimsave(video_save_path, frames, fps=10, dpi=300, loop=10)  # type: ignore

    # Predict the action using the model
    obs_input: dict[str, Any] = {"obs": obs, "aut_state": aut_state}
    action, _ = model.predict(obs_input)
    # Ensure action is a numpy int64 scalar
    return np.int64(action)
