import os
from collections import deque
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb

from ..log.wandb_logger import WandbLogger
from ..models.hiro import HIRO_Learner
from ..utils.wrappers import HIROWrapper


class HiroTrainer:
    def __init__(
        self,
        env: HIROWrapper,
        policy: HIRO_Learner,
        logger: WandbLogger,
        writer: SummaryWriter,
        total_timesteps: int = 20000,
        eval_freq: int = 1000,
        eval_num: int = 10,
        seed: int = 0,
    ):
        self.env = env
        self.policy = policy
        self.logger = logger
        self.writer = writer

        self.total_timesteps = total_timesteps
        self.eval_interval = int(self.total_timesteps // eval_freq)
        self.eval_num = eval_num

        self.last_min_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.seed = seed

    def train(self):
        global_step = 0
        eval_idx = 0

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        with tqdm(
            total=self.total_timesteps,
            desc=f"{self.policy.name} Training (timesteps)",
        ) as pbar:
            while pbar.n < self.total_timesteps:
                e = pbar.n + 1  # + 1 to avoid zero division

                #### EVALUATIONS ####
                if e >= self.eval_interval * eval_idx:
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, supp_dict = self.evaluate()

                    self.write_log(eval_dict, step=global_step, eval_log=True)
                    self.write_image(
                        supp_dict["progression"],
                        step=global_step,
                        logdir=f"images",
                        name="goal_progression",
                    )
                    self.write_video(
                        supp_dict["rendering"],
                        step=global_step,
                        logdir=f"videos",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict["eval/mean_reward"])
                    self.last_return_std.append(eval_dict["eval/std_reward"])

                    self.save_model(f"model_{global_step}.pth")

                obs, _ = self.env.reset()

                fg = obs["desired_goal"]
                s = obs["obs"].flatten()

                done: bool = False

                step: int = 0
                episode_reward: float = 0.0

                self.policy.set_final_goal(fg)

                prev_policy = deepcopy(self.policy)
                loss_dict_list: list[dict] = []
                while not done:
                    # Take action
                    a, r, n_s, done, _ = self.policy.step(
                        s, self.env, step, global_step, explore=True
                    )

                    # Append
                    self.policy.append(step, s, a, n_s, r, done)

                    # Train
                    loss_dict = self.policy.learn(global_step)
                    loss_dict_list.append(loss_dict)

                    # Updates
                    s = n_s
                    episode_reward += float(r)
                    step += 1
                    global_step += 1
                    self.policy.end_step()
                    pbar.update(1)

                # Compare weights between previous and current policy
                param_diff = {}
                for (name1, p1), (name2, p2) in zip(
                    prev_policy.low_con.named_parameters(),
                    self.policy.low_con.named_parameters(),
                ):
                    assert name1 == name2, "Parameter names do not match"
                    diff = (p1.data - p2.data).abs().mean().item()
                    param_diff[name1] = diff

                # Optionally log or print the parameter changes
                for name, diff in param_diff.items():
                    print(f"Parameter change in '{name}': {diff:.6f}")

                self.policy.end_episode()
                self.write_log(
                    {
                        "train/episode_reward": episode_reward,
                        "train/episode_length": step,
                    },
                    step=global_step,
                )
                self.write_log(
                    self.average_dict_values(loss_dict_list), step=global_step
                )

                torch.cuda.empty_cache()

    def evaluate(self):
        return evaluate(self.env, self.policy, self.eval_num, self.seed)

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image, step: int, logdir: str, name: str):
        image_path = os.path.join(logdir, name)

        if isinstance(image, list):
            image_list = image
            self.logger.write_images(
                step=step, images=image_list, logdir=image_path
            )
        elif isinstance(image, np.ndarray):
            image_list = [image]
            self.logger.write_images(
                step=step, images=image_list, logdir=image_path
            )
        elif image is None:
            return
        else:
            # assuming fig
            wandb.log({f"{image_path}": wandb.Image(image)}, step=int(step))

    def write_video(self, image: list, step: int, logdir: str, name: str):
        tensor = np.stack(image, axis=0)
        video_path = os.path.join(logdir, name)
        self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, name: str):
        ### save checkpoint
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = self.policy

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) < self.last_min_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = "best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_min_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {
            key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()
        }

        return avg_dict


def evaluate(
    env: HIROWrapper,
    policy: HIRO_Learner,
    n_eval_episodes: int = 100,
    seed: int | None = None,
    add_figure: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Evaluate the policy in the environment.
    """
    ep_buffer = []
    image_array = []
    fig = None
    for num_episodes in range(n_eval_episodes):
        step = 0
        ep_reward = []

        # Env initialization
        state, infos = env.reset(seed=seed)

        fg = state["desired_goal"]
        state = state["obs"].flatten()

        policy.set_final_goal(fg)
        subgoal = []
        transitions = []
        for t in range(env.max_episode_steps):
            with torch.no_grad():
                a, rew, next_state, done, infos = policy.step(state, env, step)
                # if t == 0:
                #     print(state, a)

            if num_episodes == 0:
                # Plotting
                image = env.render()
                image_array.append(image)
                # record subgoal track
                subgoal.append(policy.sg + state)
                transitions.append(next_state)

            state = next_state
            ep_reward.append(rew)

            policy.end_step()
            step += 1

            if done:
                policy.end_episode()
                ep_buffer.append(
                    {
                        "discounted_return": discounted_returns(
                            ep_reward, policy.gamma
                        ),
                        "return": np.sum(ep_reward),
                        "ep_length": step,
                        "is_success": infos.get("is_success", False),
                    }
                )
                if add_figure and len(subgoal) > 0 and len(transitions) > 0:
                    fig, ax = plt.subplots()
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                    # --------------------
                    # Transition setup
                    # --------------------
                    transitions = np.array(transitions)  # shape: (T, 2)
                    T = len(transitions)
                    transition_segments = np.stack(
                        [transitions[:-1], transitions[1:]], axis=1
                    )  # shape: (T-1, 2, 2)

                    # --------------------
                    # Subgoal setup
                    # --------------------
                    subgoal = np.array(subgoal)  # shape: (N, 2)
                    N = len(subgoal)

                    # --------------------
                    # Shared temporal scale: total steps
                    # --------------------
                    total_steps = T + N - 2
                    shared_norm = plt.Normalize(0, total_steps)

                    # --------------------
                    # Apply color to transitions
                    # --------------------
                    transition_colors = np.linspace(
                        0, T - 2, T - 1
                    )  # steps 0 to T-2
                    lc = LineCollection(
                        transition_segments, cmap="viridis", norm=shared_norm
                    )
                    lc.set_array(transition_colors)
                    lc.set_linewidth(2)
                    ax.add_collection(lc)

                    # Optional: scatter the transition points
                    ax.scatter(
                        transitions[:, 0],
                        transitions[:, 1],
                        c=plt.cm.viridis(shared_norm(np.arange(T))),
                        s=10,
                    )

                    # --------------------
                    # Plot subgoals with same colormap
                    # --------------------
                    subgoal_steps = np.arange(
                        T - 1, T - 1 + N
                    )  # continue time index from transitions
                    subgoal_colors = plt.cm.viridis(shared_norm(subgoal_steps))
                    ax.plot(
                        subgoal[:, 0],
                        subgoal[:, 1],
                        linestyle="--",
                        color="orange",
                        alpha=0.5,
                    )
                    ax.scatter(
                        subgoal[:, 0],
                        subgoal[:, 1],
                        c=subgoal_colors,
                        marker="x",
                        s=100,
                        label="Subgoals",
                    )

                    # Annotate subgoals with time index
                    for i, (x, y) in enumerate(subgoal):
                        ax.text(
                            x,
                            y,
                            str(i),
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black",
                        )

                    # --------------------
                    # Final Goal
                    # --------------------
                    fg = np.array(policy.fg)
                    ax.scatter(
                        fg[0],
                        fg[1],
                        marker="*",
                        color="green",
                        s=150,
                        label="Final Goal",
                    )

                    # --------------------
                    # Finishing touches
                    # --------------------
                    ax.set_title("Temporally Colored Transitions and Subgoals")
                    ax.legend()
                    ax.set_aspect("equal")
                    ax.grid(True)

                    plt.close()

                break

    # discounted_return_list: list[float] = [
    #     ep_info["discounted_return"] for ep_info in ep_buffer
    # ]
    return_list: list[float] = [ep_info["return"] for ep_info in ep_buffer]
    is_success_list: list[bool] = [
        ep_info["is_success"] for ep_info in ep_buffer
    ]
    return_mean, return_std = (
        float(np.mean(return_list)),
        float(np.std(return_list)),
    )
    # discounted_return_mean, discounted_return_std = (
    #     float(np.mean(discounted_return_list)),
    #     float(np.std(discounted_return_list)),
    # )
    success_rate = float(np.mean(is_success_list)) if is_success_list else 0.0
    ep_lengths = [ep_info["ep_length"] for ep_info in ep_buffer]
    mean_ep_length: float = float(np.mean(ep_lengths))

    eval_dict: dict[str, float] = {
        "eval/mean_ep_length": mean_ep_length,
        "eval/mean_reward": return_mean,
        "eval/std_reward": return_std,
        # "eval/mean_discounted_reward": discounted_return_mean,
        # "eval/std_discounted_reward": discounted_return_std,
        "eval/success_rate": success_rate,
    }

    return eval_dict, {
        "rendering": image_array,
        "progression": fig,
        "returns": return_list,
        "successes": is_success_list,
        "ep_lengths": ep_lengths,
    }


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
