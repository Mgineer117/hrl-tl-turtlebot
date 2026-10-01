"""
Reward wrappers for reach-avoid tasks.
"""

import gymnasium as gym
import numpy as np


class ReachAvoidWrapper(gym.Wrapper):
    """
    Wrapper that modifies rewards to encourage reaching goal while avoiding obstacles.

    Reward structure:
    - Reach goal: large positive reward (episode terminates)
    - Reach obstacle: large negative penalty (episode terminates)
    - Otherwise: shaped reward based on distance to goal and obstacle
    """

    def __init__(
        self,
        env,
        goal_pos: np.ndarray,
        obstacle_pos: np.ndarray,
    ):
        """
        Initialize reach-avoid wrapper.

        Args:
            env: Base environment
            goal_pos: Position of the goal
            obstacle_pos: Position of the obstacle
            goal_reward: Reward for reaching the goal
            obstacle_penalty: Penalty for reaching the obstacle
        """
        super().__init__(env)
        self.goal_pos = np.array(goal_pos)
        self.obstacle_pos = np.array(obstacle_pos)

    def reset(self, **kwargs):
        """Reset environment and force specific goal if applicable."""
        obs, info = self.env.reset(**kwargs)

        # Make sure agent doesn't spawn on the obstacle
        agent_pos = obs[:2]
        max_attempts = 100
        attempts = 0

        while (
            np.array_equal(agent_pos, self.obstacle_pos)
            and attempts < max_attempts
        ):
            obs, info = self.env.reset(**kwargs)
            agent_pos = obs[:2]
            attempts += 1

        if attempts >= max_attempts:
            raise RuntimeError(
                f"Failed to spawn agent away from obstacle after {max_attempts} attempts"
            )

        return obs, info

    def step(self, action):
        """Step environment and modify reward for reach-avoid behavior."""
        obs, _, terminated, truncated, info = self.env.step(action)

        # Get current position
        current_pos = obs[:2]

        reward = 0.0
        # Check if reached obstacle
        if np.array_equal(current_pos, self.obstacle_pos):
            terminated = True  # Episode ends in failure
            info["reached_obstacle"] = True
            info["success"] = False
            reward += -10.0  # Large negative penalty
        # Check if reached goal
        elif np.array_equal(current_pos, self.goal_pos):
            terminated = True
            info["success"] = True
            reward += 10.0  # Large positive reward
        else:
            # Compute shaped reward
            info["success"] = False

        goal_distance = np.abs(current_pos - self.goal_pos).sum()
        obstacle_distance = np.abs(current_pos - self.obstacle_pos).sum()
        # reward += -goal_distance + obstacle_distance

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render only the goal and obstacle, not all goal positions."""
        if self.env.render_mode == "rgb_array":
            return self._render_reach_avoid_rgb()
        else:
            return self.env.render()

    def _render_reach_avoid_rgb(self):
        """Render RGB array showing only goal and obstacle."""
        cell_size = 50
        grid_size = self.env.grid_size
        img = (
            np.ones(
                (grid_size * cell_size, grid_size * cell_size, 3),
                dtype=np.uint8,
            )
            * 255
        )

        # Draw grid lines
        for i in range(grid_size + 1):
            img[i * cell_size : i * cell_size + 1, :] = 0
            img[:, i * cell_size : i * cell_size + 1] = 0

        # Draw goal (green)
        x, y = self.goal_pos[0], self.goal_pos[1]
        img_row = (grid_size - 1 - y) * cell_size
        img_col = x * cell_size
        img[
            img_row + 5 : img_row + cell_size - 5,
            img_col + 5 : img_col + cell_size - 5,
        ] = [0, 255, 0]  # Green

        # Draw obstacle (red)
        x, y = self.obstacle_pos[0], self.obstacle_pos[1]
        img_row = (grid_size - 1 - y) * cell_size
        img_col = x * cell_size
        img[
            img_row + 5 : img_row + cell_size - 5,
            img_col + 5 : img_col + cell_size - 5,
        ] = [255, 0, 0]  # Red

        # Draw agent (cyan)
        x, y = self.env.agent_pos[0], self.env.agent_pos[1]
        img_row = (grid_size - 1 - y) * cell_size
        img_col = x * cell_size
        img[
            img_row + 10 : img_row + cell_size - 10,
            img_col + 10 : img_col + cell_size - 10,
        ] = [0, 255, 255]  # Cyan

        return img
