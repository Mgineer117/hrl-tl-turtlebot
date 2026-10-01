import os

import imageio
import numpy as np
from gymnasium import Env

from hrl_tl.baseline.algorithms.HIRO import HIRO


def record_replay(
    demo_env: Env, model: HIRO, animation_save_path: str, verbose: bool = True
) -> None:
    obs, _ = demo_env.reset()
    terminated: bool = False
    truncated: bool = False
    frames = [demo_env.render()]
    rewards: list[float] = []
    step: int = 0
    model.env.reset()
    while not (terminated or truncated):
        action = model.predict(obs, curr_step=step)
        # Take argmax if the action is discrete
        if isinstance(action, np.ndarray) and action.ndim > 0:
            action = np.argmax(action) if model.policy.is_discrete else action
        obs, reward, terminated, truncated, info = demo_env.step(action)
        if verbose:
            print(
                f" - Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Success: {info['is_success']}"
            )
        rewards.append(reward)  # type: ignore
        frame = demo_env.render()
        frames.append(frame)
        step += 1

    demo_env.close()
    print(f" - Total reward: {sum(rewards)}")
    print(f" - Total steps: {step}")

    os.makedirs(os.path.dirname(animation_save_path), exist_ok=True)
    imageio.mimsave(animation_save_path, frames, fps=10, dpi=300, loop=10)  # type: ignore
    if verbose:
        print(f" - Replay saved to {animation_save_path}")
