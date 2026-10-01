import numpy as np
import torch
from numpy import signedinteger
from numpy.typing import NDArray


def init_buffer(
    buffer_size: int | tuple[int, ...],
    item_dim: int | tuple | NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    Initialize a buffer with zeros.

    Parameters
    ----------
    buffer_size : int
        The size of the buffer.
    item_dim : int | tuple | NDArray[np.int64]
        The dimension of each item in the buffer.

    Returns
    -------
    NDArray[np.float64]
        A buffer initialized with zeros.
    """
    if isinstance(item_dim, (int, signedinteger)):
        item_dim = (item_dim,)
    if isinstance(buffer_size, int):
        buffer_size = (buffer_size,)
    buffer = np.zeros(buffer_size + item_dim, dtype=np.float64)
    return buffer


class ReplayBuffer:
    def __init__(
        self,
        state_dim: tuple,
        goal_dim: tuple,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        device,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.state = init_buffer(buffer_size, state_dim)
        self.goal = init_buffer(buffer_size, goal_dim)
        self.action = init_buffer(buffer_size, action_dim)
        self.n_state = init_buffer(buffer_size, state_dim)
        self.reward = init_buffer(buffer_size, 1)
        self.terminal = init_buffer(buffer_size, 1)

        self.device = device

    def pre_process(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(x, list):
            temp_list = []
            for element in x:
                if isinstance(element, torch.Tensor):
                    temp_list.append(element.cpu().numpy())
                else:
                    temp_list.append(element)
            x = temp_list
        return x

    def append(self, state, goal, action, n_state, reward, done):
        self.state[self.ptr] = self.pre_process(state)
        self.goal[self.ptr] = self.pre_process(goal)
        self.action[self.ptr] = self.pre_process(action)
        self.n_state[self.ptr] = self.pre_process(n_state)
        self.reward[self.ptr] = self.pre_process(reward)
        self.terminal[self.ptr] = self.pre_process(done)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.terminal[ind]).to(self.device),
        )


class LowReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        state_dim: tuple | int,
        goal_dim: tuple | int,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        device,
    ):
        super(LowReplayBuffer, self).__init__(
            state_dim, goal_dim, action_dim, buffer_size, batch_size, device
        )
        self.n_goal = init_buffer(buffer_size, goal_dim)

    def append(self, state, goal, action, n_state, n_goal, reward, done):
        self.state[self.ptr] = self.pre_process(state)
        self.goal[self.ptr] = self.pre_process(goal)
        self.action[self.ptr] = self.pre_process(action)
        self.n_state[self.ptr] = self.pre_process(n_state)

        self.n_goal[self.ptr] = self.pre_process(n_goal)
        self.reward[self.ptr] = self.pre_process(reward)
        self.terminal[self.ptr] = self.pre_process(done)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.n_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.terminal[ind]).to(self.device),
        )


class HighReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        state_dim: tuple | int,
        goal_dim: tuple | int,
        subgoal_dim: tuple | int,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        freq: int,
        device,
    ):
        if isinstance(state_dim, int):
            state_dim = (state_dim,)
        if isinstance(goal_dim, int):
            goal_dim = (goal_dim,)
        if isinstance(subgoal_dim, int):
            subgoal_dim = (subgoal_dim,)

        super(HighReplayBuffer, self).__init__(
            state_dim, goal_dim, action_dim, buffer_size, batch_size, device
        )
        self.action = init_buffer(buffer_size, subgoal_dim)
        self.state_arr = init_buffer((buffer_size, freq), state_dim)
        self.action_arr = init_buffer((buffer_size, freq), action_dim)

    def append(
        self, state, goal, action, n_state, reward, done, state_arr, action_arr
    ):
        self.state[self.ptr] = self.pre_process(state)
        self.goal[self.ptr] = self.pre_process(goal)
        self.action[self.ptr] = self.pre_process(action)
        self.n_state[self.ptr] = self.pre_process(n_state)
        self.reward[self.ptr] = self.pre_process(reward)
        self.terminal[self.ptr] = self.pre_process(done)
        self.state_arr[self.ptr, :, :] = state_arr
        self.action_arr[self.ptr, :, :] = action_arr

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.terminal[ind]).to(self.device),
            torch.FloatTensor(self.state_arr[ind]).to(self.device),
            torch.FloatTensor(self.action_arr[ind]).to(self.device),
        )


class SubgoalActionSpace(object):
    def __init__(self, dim: tuple):
        self.dim = dim
        self.shape = (1,) + dim
        self.low = -0.25 * np.ones((np.prod(dim),))
        self.high = 0.25 * np.ones((np.prod(dim),))

    def sample(self):
        subgoal = (self.high - self.low) * np.random.sample(
            np.prod(self.dim)
        ) + self.low
        return subgoal.reshape(self.dim)


class Subgoal(object):
    def __init__(self, dim):
        self.action_space = SubgoalActionSpace(dim)
        self.action_dim = np.prod(dim)
