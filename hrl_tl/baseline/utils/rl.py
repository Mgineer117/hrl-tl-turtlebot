import gym_multigrid
import gymnasium as gym
import numpy as np
import torch


def call_env(args):
    """
    Call the environment based on the given name.
    """
    env = None
    env_name, version = args.env_name.split("-")
    version = int(version[-1]) if version[-1].isdigit() else version[-1]

    if env_name == "fourrooms":
        env_kwargs = {
            "max_episode_steps": 100,
            "spawn_type": 0,
            "random_init_pos": False,
            "layout_config": {
                "field_map": [
                    "#############",
                    "#     #     #",
                    "#     #     #",
                    "#           #",
                    "#     #     #",
                    "#     #     #",
                    "## ####     #",
                    "#     ### ###",
                    "#     #     #",
                    "#     #     #",
                    "#           #",
                    "#     #     #",
                    "#############",
                ],
                "spawn_configs": [
                    {
                        "agent": (3, 9),
                        "goal": {"pos": (9, 4), "reward": 1.0},
                        "lavas": [
                            {"pos": (8, 4), "reward": 0},
                            {"pos": (9, 2), "reward": 0},
                            {"pos": (11, 1), "reward": 0},
                            {"pos": (5, 3), "reward": 0},
                            {"pos": (3, 5), "reward": 0},
                            {"pos": (3, 2), "reward": 0},
                            {"pos": (5, 9), "reward": -1},
                            {"pos": (3, 8), "reward": -1},
                            {"pos": (2, 11), "reward": -1},
                            {"pos": (10, 8), "reward": -1},
                            {"pos": (8, 9), "reward": -1},
                            {"pos": (7, 11), "reward": -1},
                        ],
                        "holes": [
                            {"pos": (7, 3), "reward": 0},
                            {"pos": (10, 5), "reward": 0},
                            {"pos": (8, 6), "reward": 0},
                            {"pos": (4, 4), "reward": -1},
                            {"pos": (2, 3), "reward": -1},
                            {"pos": (1, 1), "reward": -1},
                            {"pos": (2, 7), "reward": 0},
                            {"pos": (1, 9), "reward": 0},
                            {"pos": (4, 10), "reward": 0},
                            {"pos": (7, 8), "reward": -1},
                            {"pos": (9, 10), "reward": -1},
                            {"pos": (11, 11), "reward": -1},
                        ],
                    },
                ],
            },
        }

        env = gym.make("multigrid-rooms-v0", **env_kwargs)
    else:
        raise ValueError(f"Environment {env_name} is not supported.")

    args.state_dim = env.observation_space.shape
    args.action_dim = env.action_space.n
    args.episode_len = env._max_episode_steps
    args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    return env


def estimate_advantages(
    rewards, terminals, values, gamma=0.99, gae=0.95, device=torch.device("cpu")
):
    rewards, terminals, values = (
        rewards.to(torch.device("cpu")),
        terminals.to(torch.device("cpu")),
        values.to(torch.device("cpu")),
    )
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = (
            rewards[i] + gamma * prev_value * (1 - terminals[i]) - values[i]
        )
        advantages[i] = deltas[i] + gamma * gae * prev_advantage * (
            1 - terminals[i]
        )

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    # advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = advantages.to(device), returns.to(device)
    return advantages, returns


def get_flat_params_from(model):
    # pdb.set_trace()
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(torch.zeros(param.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad
