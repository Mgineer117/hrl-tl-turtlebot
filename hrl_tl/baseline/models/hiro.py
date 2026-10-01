import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from ..models.layers.hiro_networks import HL_Policy, LL_policy
from ..utils.hiro_utils import HighReplayBuffer, LowReplayBuffer, Subgoal
from ..utils.wrappers import HIROWrapper


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


class HIRO_Learner(nn.Module):
    def __init__(
        self,
        state_dim: tuple | int,
        action_dim: tuple | int,
        goal_dim: tuple | int,
        subgoal_dim: tuple | int,
        actor_fc_dim: tuple | list,
        critic_fc_dim: tuple | list,
        actor_lr: float,
        critic_lr: float,
        is_discrete: bool,
        start_training_steps: int = 1000,
        buffer_size: int = 100_000,
        batch_size: int = 512,
        buffer_freq: int = 10,
        train_freq: int = 10,
        policy_freq_high: int = 2,
        policy_freq_low: int = 2,
        gamma: float = 0.99,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.subgoal_dim = np.prod(subgoal_dim)
        self.is_discrete = is_discrete

        self.subgoal = Subgoal(subgoal_dim)
        scale_high = (
            self.subgoal.action_space.high * np.ones(subgoal_dim).flatten()
        )

        self.high_con = HL_Policy(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            actor_fc_dim=actor_fc_dim,
            critic_fc_dim=critic_fc_dim,
            action_min=self.subgoal.action_space.low,
            action_max=self.subgoal.action_space.high,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            is_discrete=False,
            policy_freq=policy_freq_high,
            scale_high=scale_high,
            device=device,
        )

        self.low_con = LL_policy(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            actor_fc_dim=actor_fc_dim,
            critic_fc_dim=critic_fc_dim,
            # action_min=self.env,
            # action_max=self.subgoal.action_space.high,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            is_discrete=is_discrete,
            policy_freq=policy_freq_low,
            device=device,
        )

        self.replay_buffer_low = LowReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
        )

        self.replay_buffer_high = HighReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq,
            device=device,
        )

        self.buffer_freq = buffer_freq
        self.train_freq = train_freq
        self.gamma = gamma
        self.episode_subreward = 0
        self.sr = 0

        self.buf = [None, None, None, 0, None, None, [], []]
        self.fg = np.zeros(int(np.prod(goal_dim)))
        self.sg = self.subgoal.action_space.sample()

        self.start_training_steps = start_training_steps
        self.name = "HIRO"

    def get_random_action(self):
        if self.is_discrete:
            # self.action_dim is a tuple like (n,) for Discrete(n)
            n = (
                self.action_dim[0]
                if isinstance(self.action_dim, tuple)
                else self.action_dim
            )
            idx = np.random.randint(n)  # sample integer in [0, n-1]
            one_hot = np.zeros(n, dtype=np.float32)
            one_hot[idx] = 1.0
            return one_hot[np.newaxis, :]
        else:
            # self.action_dim is a tuple like (n,) for Box(-1, 1, (n,))
            shape = (
                self.action_dim
                if isinstance(self.action_dim, tuple)
                else (self.action_dim,)
            )
            return np.random.uniform(low=-1.0, high=1.0, size=shape).astype(
                np.float32
            )[np.newaxis, :]

    def set_final_goal(self, fg):
        self.fg = fg

    def step(self, s, env: HIROWrapper, step, global_step=0, explore=False):
        ## Lower Level Controller
        if explore:
            # Take random action for start_training_steps
            if global_step < self.start_training_steps:
                a = self.get_random_action()

            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)

        # Take action
        if isinstance(a, np.ndarray):
            a = a.squeeze(0) if a.shape[-1] > 1 else [a]
        elif isinstance(a, torch.Tensor):
            a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]
        else:
            raise ValueError("Unknown action type")

        if len(a) >= 1:
            env_a = np.argmax(a) if self.is_discrete else a
        else:
            env_a = a
        obs, r, done, info = env.step(env_a)
        n_s = obs["obs"].flatten()

        ## Higher Level Controller
        # Take random action for start_training steps
        if explore:
            if global_step < self.start_training_steps:
                n_sg = self.subgoal.action_space.sample()
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)

        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)

        self.n_sg = n_sg

        return a, r, n_s, done, info

    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)

        # Low Replay Buffer
        self.replay_buffer_low.append(
            s, self.sg, a, n_s, self.n_sg, self.sr, float(d)
        )

        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.replay_buffer_high.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    n_state=self.buf[4],
                    reward=self.buf[3],
                    done=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=np.array(self.buf[7]),
                )
            self.buf = [s, self.fg, self.sg, 0, None, None, [], []]

        self.buf[3] += r
        self.buf[6].append(s)
        self.buf[7].append(a)

    def learn(self, global_step):
        loss_dict = {}

        if global_step >= self.start_training_steps:
            loss = self.low_con.learn(self.replay_buffer_low)
            loss_dict.update(loss)

            if global_step % self.train_freq == 0:
                loss = self.high_con.learn(
                    self.low_con, self.replay_buffer_high
                )

                loss_dict.update(loss)

        return loss_dict

    def _choose_action_with_noise(self, s, sg):
        with torch.no_grad():
            a, _ = self.low_con(s, sg, deterministic=False)
        return a

    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:  # Should be zero
            with torch.no_grad():
                sg, _ = self.high_con(s, self.fg, deterministic=False)
            sg = sg.reshape(self.subgoal_dim).cpu().numpy()
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def _choose_action(self, s, sg):
        with torch.no_grad():
            a, _ = self.low_con(s, sg, deterministic=True)
        return a

    def _choose_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:
            with torch.no_grad():
                sg, _ = self.high_con(s, self.fg, deterministic=True)
            # sg = sg.reshape((sg.shape[0],) + self.subgoal_dim).shape
            sg = sg.reshape(self.subgoal_dim).cpu().numpy()
        else:
            sg = self.subgoal_transition(s, sg, n_s)

        return sg

    def subgoal_transition(self, s, sg, n_s):
        # print(s[: sg.shape[0]].shape, sg.shape, n_s[: sg.shape[0]].shape)
        return s[: sg.shape[0]] + sg - n_s[: sg.shape[0]]

    def low_reward(self, s: NDArray, sg: NDArray, n_s: NDArray) -> float:
        abs_s = s[: sg.shape[0]].flatten() + sg.flatten()
        return -np.sqrt(np.sum((abs_s - n_s[: sg.shape[0]].flatten()) ** 2))

    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg

    def end_episode(self):
        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]
