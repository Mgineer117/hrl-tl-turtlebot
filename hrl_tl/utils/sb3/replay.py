from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Any

import imageio
import numpy as np
from gymnasium import Env
from sb3_hrl.option.vis import record_option_replay
from stable_baselines3.common.base_class import BaseAlgorithm

from ..render import DiscreteActionVectorRenderer

__all__ = [
    "record_option_replay",
    "record_replay",
    "record_replay_with_actions",
]


def record_replay(
    demo_env: Env[Any, Any],
    model: BaseAlgorithm,
    animation_save_path: str,
    verbose: bool = True,
    close_env: bool = True,
    fps: int = 10,
) -> None:
    """Records an evaluation episode replay and saves it as an animation.

    Args:
        demo_env: Evaluation environment to rollout.
        model: Policy algorithm used to sample actions.
        animation_save_path: Filepath where the animation should be saved.
        verbose: Whether to print rollout transition details.
        close_env: Whether to close the environment upon recording completion.
        fps: Frames per second for the saved animation.
    """
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    first_frame = demo_env.render()
    frames: list[Any] = []
    if isinstance(first_frame, list):
        frames.extend(first_frame)
    elif first_frame is not None:
        frames.append(first_frame)

    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs)  # type: ignore[assignment]
        obs, reward, terminated, truncated, info = demo_env.step(action)
        if verbose:
            print(
                f" - Reward: {float(reward):.2f}, Terminated: {terminated}, "
                f"Truncated: {truncated}, Success: {info.get('is_success', 'N/A')}"
            )
        rewards.append(float(reward))
        frame = demo_env.render()
        if isinstance(frame, list):
            frames.extend(frame)
        elif frame is not None:
            frames.append(frame)

    if close_env:
        demo_env.close()
    if verbose:
        print(f" - Total reward: {sum(rewards):.2f}")

    save_path = Path(animation_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        duration_ms = 1000.0 / max(1, fps)
        imageio.mimsave(save_path, frames, duration=duration_ms, loop=0)  # type: ignore[call-overload]
    if verbose:
        print(f" - Replay saved to {animation_save_path}")


# def record_replay_hl(
#     demo_env: Env, model: BaseAlgorithm, animation_save_path: str, verbose: bool = True
# ) -> None:


def record_replay_with_actions(
    demo_env: Any,
    model: Any,
    animation_save_path: str,
    verbose: bool = True,
) -> None:

    # Initialize action renderer for multi-discrete actions
    composite_action_renderer = DiscreteActionVectorRenderer(
        arrow_width=0.05,
        arrow_color="green",
        num_directions=demo_env.action_space.n,
    )
    goal_action_renderer = DiscreteActionVectorRenderer(
        arrow_color="blue", num_directions=demo_env.action_space.n
    )
    constraint_action_renderer = DiscreteActionVectorRenderer(
        arrow_color="red", num_directions=demo_env.action_space.n
    )

    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=False)

        # Update renderer with latest probabilities from CPC policy
        if (
            model.action_combinations is not None
            and model.last_joint_prob is not None
        ):
            # If the action dimension is 1, we need to add velocity to match the expected shape
            if model.action_combinations.shape[1] == 1:
                action_combinations = np.hstack(
                    [
                        model.action_combinations,
                        np.ones(
                            (model.action_combinations.shape[0], 1),
                        )
                        * 5,
                    ]
                )
            else:
                action_combinations = model.action_combinations
            composite_action_renderer.set_multi_discrete_probabilities(
                action_combinations,
                model.last_joint_prob,
            )
            goal_action_renderer.set_multi_discrete_probabilities(
                action_combinations,
                model.last_goal_prob,
            )
            constraint_action_renderer.set_multi_discrete_probabilities(
                action_combinations,
                model.last_constraint_prob,
            )

        # Ensure action is a numpy int64 scalar
        obs, reward, terminated, truncated, info = demo_env.step(action)
        if verbose:
            print(f"Step {len(rewards) + 1}:")
            print(
                f" - Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info.get('is_success', 'N/A')}"
            )
            print(" - Obs: ")
            pprint(obs)
            print(" - Info: ")
            pprint(info)
        rewards.append(reward)
        _ = demo_env.render()

        # Post-render action probabilities onto the frame
        frame_with_actions = constraint_action_renderer.render(
            demo_env.unwrapped.env
        )
        frame_with_actions = goal_action_renderer.render(demo_env.unwrapped.env)
        frame_with_actions = composite_action_renderer.render(
            demo_env.unwrapped.env
        )
        frames.append(frame_with_actions)
