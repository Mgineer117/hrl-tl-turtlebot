import gymnasium as gym
import numpy as np


class SimpleGridEnv(gym.Env):
    """
    Simple 3x3 grid environment where there are an object at the center and another at the top-right corner (9 states).
    Actions correspond to moving up, down, left, or right in the grid (4 actions).
    The agent randomly spawns in an empty cell and must navigate to one of the objects (goals).
    The observations are the agent's (x, y) coordinates in the grid and goal index (0 for center, 1 for top-right).
    Coordinate system: (0,0) is bottom-left, x increases right, y increases up.
    The reward is dense, calculated as the negative Manhattan distance to the goal.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        # Grid size is 3x3
        self.grid_size = 3

        # Define the two goal positions in (x, y) coordinates
        # x: 0=left, 2=right; y: 0=bottom, 2=top
        self.goal_positions = [
            np.array([1, 1]),  # Center: goal index 0
            np.array([2, 2]),  # Top-right: goal index 1
            np.array([0, 2]),  # Top-left: goal index 2
            np.array([1, 0]),  # Bottom-center: goal index 3
        ]

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: [x, y, goal_index]
        # x, y are in range [0, 2], goal_index is in range [0, 1, 2, 3]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([2, 2, 3]), dtype=np.int32
        )

        self.render_mode = render_mode

        # Initialize state
        self.agent_pos = None
        self.goal_index = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select a goal (0 for center, 1 for top-right, 2 for top-left, 3 for bottom-center)
        self.goal_index = self.np_random.integers(0, len(self.goal_positions))

        # Get all possible positions
        all_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ]

        # Remove goal positions (cells with objects)
        empty_positions = [
            pos
            for pos in all_positions
            if not any(
                np.array_equal(pos, goal_pos)
                for goal_pos in self.goal_positions
            )
        ]

        # Randomly spawn agent in an empty cell
        spawn_idx = self.np_random.integers(0, len(empty_positions))
        self.agent_pos = np.array(empty_positions[spawn_idx])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map action to direction in (x, y) coordinates
        # x increases right, y increases up
        direction_map = {
            0: np.array([0, 1]),  # up (increase y)
            1: np.array([0, -1]),  # down (decrease y)
            2: np.array([-1, 0]),  # left (decrease x)
            3: np.array([1, 0]),  # right (increase x)
        }

        # Calculate new position
        new_pos = self.agent_pos + direction_map[action]

        # Check if new position is within bounds
        if (0 <= new_pos[0] < self.grid_size) and (
            0 <= new_pos[1] < self.grid_size
        ):
            self.agent_pos = new_pos

        # Calculate reward as negative Manhattan distance to the goal
        assert self.goal_index in list(range(len(self.goal_positions)))
        goal_pos = self.goal_positions[self.goal_index]
        manhattan_distance = np.abs(self.agent_pos - goal_pos).sum()
        reward = -manhattan_distance

        # Check if agent reached the goal
        assert self.agent_pos is not None
        terminated = np.array_equal(self.agent_pos, goal_pos)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Return the observation: [x, y, goal_index]"""
        assert self.agent_pos is not None
        return np.array(
            [self.agent_pos[0], self.agent_pos[1], self.goal_index],
            dtype=np.int32,
        )

    def _get_info(self):
        """Return additional information"""
        assert self.agent_pos is not None
        assert self.goal_index in list(range(len(self.goal_positions)))
        goal_pos = self.goal_positions[self.goal_index]
        manhattan_distance = np.abs(self.agent_pos - goal_pos).sum()
        return {
            "agent_position": self.agent_pos.copy(),
            "goal_position": goal_pos.copy(),
            "goal_index": self.goal_index,
            "distance_to_goal": manhattan_distance,
        }

    def render(self):
        if self.render_mode == "human":
            return self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_text(self):
        """Render the grid as text"""
        grid = [
            ["." for _ in range(self.grid_size)] for _ in range(self.grid_size)
        ]

        # Place goals (convert x,y to array indices)
        for idx, goal_pos in enumerate(self.goal_positions):
            x, y = goal_pos[0], goal_pos[1]
            grid[self.grid_size - 1 - y][x] = str(idx)

        # Place agent (convert x,y to array indices)
        assert self.agent_pos is not None
        assert self.goal_index in [0, 1]
        x, y = self.agent_pos[0], self.agent_pos[1]
        grid[self.grid_size - 1 - y][x] = "A"

        # Print grid (top to bottom)
        print("\n" + "=" * (self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("=" * (self.grid_size * 2 + 1))
        print(
            f"Goal: {self.goal_index} at {self.goal_positions[self.goal_index]} (x,y)"
        )
        print(f"Agent: {self.agent_pos} (x,y)")

    def _render_rgb_array(self):
        """Render the grid as RGB array"""
        # Create a simple RGB representation
        # Each cell is 50x50 pixels
        cell_size = 50
        img = (
            np.ones(
                (self.grid_size * cell_size, self.grid_size * cell_size, 3),
                dtype=np.uint8,
            )
            * 255
        )

        # Draw grid lines
        for i in range(self.grid_size + 1):
            img[i * cell_size : i * cell_size + 1, :] = 0
            img[:, i * cell_size : i * cell_size + 1] = 0

        # Draw goals (different colors)
        # Convert (x, y) to image coordinates: img[row, col] where row increases downward
        goal_colors = [
            [255, 0, 0],  # Index 0 (center): Red
            [0, 255, 0],  # Index 1 (top-right): Green
            [0, 0, 255],  # Index 2 (top-left): Blue
            [255, 255, 0],  # Index 3 (bottom-center): Yellow
        ]
        # for idx, goal_pos in enumerate(self.goal_positions):
        #     x, y = goal_pos[0], goal_pos[1]
        #     # Image row = (grid_size - 1 - y) to flip y-axis
        #     img_row = (self.grid_size - 1 - y) * cell_size
        #     img_col = x * cell_size
        #     color = goal_colors[idx]
        #     img[
        #         img_row + 5 : img_row + cell_size - 5,
        #         img_col + 5 : img_col + cell_size - 5,
        #     ] = color

        assert self.goal_index is not None
        goal_pos = self.goal_positions[self.goal_index]
        x, y = goal_pos[0], goal_pos[1]
        # Image row = (grid_size - 1 - y) to flip y-axis
        img_row = (self.grid_size - 1 - y) * cell_size
        img_col = x * cell_size
        color = goal_colors[self.goal_index]
        img[
            img_row + 5 : img_row + cell_size - 5,
            img_col + 5 : img_col + cell_size - 5,
        ] = color

        # Draw agent (cyan)
        assert self.agent_pos is not None
        x, y = self.agent_pos[0], self.agent_pos[1]
        img_row = (self.grid_size - 1 - y) * cell_size
        img_col = x * cell_size
        img[
            img_row + 10 : img_row + cell_size - 10,
            img_col + 10 : img_col + cell_size - 10,
        ] = [0, 255, 255]  # Cyan

        return img

    def close(self):
        pass
