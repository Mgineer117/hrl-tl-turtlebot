from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from contgrid.core.world import World
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Wedge
from numpy.typing import NDArray


class PostRenderer(ABC):
    """
    Abstract base class for post-renderers that add additional elements to the rendering after the main environment's `env.render()` is called.
    """

    @abstractmethod
    def render(self, env: Any, **kwargs: Any) -> Any:
        """
        Render additional elements onto the given matplotlib figure and axes.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to render on.
        ax : Axes
            Matplotlib axes to render on.
        env : Env
            The gymnasium environment being rendered.
        **kwargs : Any
            Additional rendering options specific to the renderer implementation.
        """
        pass

    def get_image(self, fig: Figure) -> NDArray[np.uint8]:
        """
        Extract RGB image array from the figure canvas.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to extract image from.

        Returns
        -------
        NDArray[np.uint8]
            RGB array of the rendered image.
        """
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        img_array = np.asarray(buf)
        # Convert RGBA to RGB
        return img_array[:, :, :3].astype(np.uint8)


class DiscreteActionVectorRenderer(PostRenderer):
    """
    Renderer for environments with discrete action vectors.
    Draws arrows representing action probabilities, centered at the agent's position,
    with arrow lengths proportional to probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> # Define 4 cardinal directions
    >>> vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    >>> renderer = DiscreteActionVectorRenderer(vectors, agent_idx=0)
    >>>
    >>> # Update probabilities and render
    >>> probs = np.array([0.5, 0.3, 0.1, 0.1])
    >>> renderer.set_probabilities(probs)
    >>> renderer.render(fig, ax, env)
    """

    def __init__(
        self,
        action_vectors: NDArray[np.float64] | None = None,
        num_directions: int = 16,
        num_velocities: int = 5,
        agent_u_range: float = 5.0,
        agent_idx: int = 0,
        max_arrow_length: float = 1.5,
        arrow_color: str = "blue",
        arrow_alpha: float = 0.7,
        arrow_width: float = 0.02,
        head_width: float = 0.1,
        head_length: float = 0.1,
        min_prob_threshold: float = 0.01,
    ) -> None:
        """
        Initialize the discrete action vector renderer.

        Parameters
        ----------
        action_vectors : NDArray[np.float64]
            Array of shape (num_actions, 2) where each row is a direction vector (x, y).
            These vectors define the possible discrete actions.
        agent_idx : int
            Index of the agent to render arrows for.
        max_arrow_length : float
            Maximum length of arrows when probability is 1.0.
        arrow_color : str
            Color of the arrows.
        arrow_alpha : float
            Base alpha (transparency) of the arrows.
        arrow_width : float
            Width of the arrow shaft.
        head_width : float
            Width of the arrow head.
        head_length : float
            Length of the arrow head.
        min_prob_threshold : float
            Minimum probability threshold below which arrows are not drawn.
        """
        self.action_vectors = action_vectors
        self.num_directions = num_directions
        self.num_velocities = num_velocities
        self.u_range = agent_u_range
        self.agent_idx = agent_idx
        self.max_arrow_length = max_arrow_length
        self.arrow_color = arrow_color
        self.arrow_alpha = arrow_alpha
        self.arrow_width = arrow_width
        self.head_width = head_width
        self.head_length = head_length
        self.min_prob_threshold = min_prob_threshold

        # Normalize action vectors to unit vectors for consistent scaling
        if action_vectors is not None:
            norms = np.linalg.norm(action_vectors, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms = np.where(norms == 0, 1, norms)
            self.unit_vectors = action_vectors / norms

            # Initialize with uniform probabilities
            self.probabilities: NDArray[np.float64] | None = np.ones(
                len(action_vectors)
            ) / len(action_vectors)
        else:
            self.unit_vectors = None
            self.probabilities = None

        # For multi-discrete actions
        self.action_combinations: NDArray[np.integer] | None = None
        self.joint_probabilities: NDArray[np.float64] | None = None

    def set_probabilities(self, probabilities: NDArray[np.float64]) -> None:
        """
        Update the action probabilities.

        Parameters
        ----------
        probabilities : NDArray[np.float64]
            Array of shape (num_actions,) containing probabilities for each action.

        Raises
        ------
        ValueError
            If probabilities length doesn't match the number of action vectors.
        """
        if self.action_vectors is not None and len(probabilities) != len(
            self.action_vectors
        ):
            raise ValueError(
                f"Probabilities length ({len(probabilities)}) must match "
                f"action_vectors length ({len(self.action_vectors)})"
            )
        self.probabilities = probabilities

    def set_multi_discrete_probabilities(
        self,
        action_combinations: NDArray[np.integer],
        joint_probabilities: NDArray[np.float64],
    ) -> None:
        """
        Update probabilities for multi-discrete action space.

        Parameters
        ----------
        action_combinations : NDArray[np.integer]
            Array of shape (num_combinations, num_action_dims) containing all action combinations.
        joint_probabilities : NDArray[np.float64]
            Array of shape (num_combinations,) containing joint probabilities for each combination.
        """
        self.action_combinations = action_combinations
        self.joint_probabilities = joint_probabilities

        # Action vectors are
        direction_idx = action_combinations[:, 0]
        vel_idx = action_combinations[:, 1]

        angle = (direction_idx / self.num_directions) * 2 * np.pi
        magnitude = (vel_idx / self.num_velocities) * self.u_range

        x_vel = magnitude * np.cos(angle)
        y_vel = magnitude * np.sin(angle)
        self.action_vectors = np.column_stack([x_vel, y_vel])
        self.unit_vectors = self.action_vectors / np.linalg.norm(
            self.action_vectors, axis=1, keepdims=True
        )

    def render(self, env: Any, **kwargs: Any) -> Any:
        """
        Render action probability arrows onto the given matplotlib axes.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to render on.
        ax : Axes
            Matplotlib axes to render on.
        env : Env
            The gymnasium environment, expected to have a `world` attribute with agents.
        **kwargs : Any
            Optional keyword arguments:
            - probabilities: NDArray[np.float64] - Override stored probabilities for this render
            - agent_idx: int - Override stored agent index for this render
        """
        # Allow overriding probabilities and agent_idx per render call
        probabilities = kwargs.get("probabilities", self.probabilities)
        agent_idx = kwargs.get("agent_idx", self.agent_idx)

        # Get agent position from environment
        if not hasattr(env, "world"):
            return

        world: World = env.world  # type: ignore
        if agent_idx >= len(world.agents):
            return

        agent = world.agents[agent_idx]
        agent_pos = agent.state.pos

        fig = env.fig
        ax = env.ax

        # Handle multi-discrete action space
        if (
            self.action_combinations is not None
            and self.joint_probabilities is not None
        ):
            # Show top-k action combinations
            top_k = min(5, len(self.joint_probabilities))
            top_indices = np.argsort(self.joint_probabilities)[-top_k:][::-1]

            for idx in top_indices:
                action_combo = self.action_combinations[idx]
                prob = self.joint_probabilities[idx]

                if prob < self.min_prob_threshold:
                    continue

                # For multi-discrete, show action as text near agent
                ax.text(
                    0.5,
                    0.5,
                    f"{tuple(action_combo.tolist())}: {prob:.3f}",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9
                    ),
                    zorder=11,
                )
        if (
            self.joint_probabilities is not None
            and self.unit_vectors is not None
        ):
            probabilities = self.joint_probabilities
            for unit_vec, prob in zip(self.unit_vectors, probabilities):
                if prob < self.min_prob_threshold:
                    continue

                # Scale arrow length by probability
                scaled_prob = (prob - 0.5) / 3 + 0.5
                arrow_length = scaled_prob * self.max_arrow_length
                dx = unit_vec[0] * arrow_length
                dy = unit_vec[1] * arrow_length

                # Draw arrow from agent position
                ax.arrow(
                    agent_pos[0],
                    agent_pos[1],
                    dx,
                    dy,
                    color=self.arrow_color,
                    alpha=self.arrow_alpha
                    * scaled_prob,  # More transparent for lower probabilities
                    width=self.arrow_width,
                    head_width=self.head_width,
                    head_length=self.head_length,
                    length_includes_head=True,
                    zorder=10,
                )

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgb_array = np.asarray(buf)[:, :, :3]  # Remove alpha channel

        env.fig = fig
        env.ax = ax
        return rgb_array


class DiscreteActionPointCloudRenderer(PostRenderer):
    """Render discrete action probabilities as a point cloud around the agent.

    Each action is represented as a point in the direction of the action vector,
    with distance from the agent proportional to the action probability.
    """

    def __init__(
        self,
        action_vectors: NDArray[np.float64] | None = None,
        num_directions: int = 16,
        num_velocities: int = 5,
        agent_u_range: float = 5.0,
        agent_idx: int = 0,
        max_arrow_length: float = 1.5,
        arrow_color: str = "blue",
        arrow_alpha: float = 0.7,
        arrow_width: float = 0.02,
        head_width: float = 0.1,
        head_length: float = 0.1,
        min_prob_threshold: float = 0.01,
    ) -> None:
        self.action_vectors = action_vectors
        self.num_directions = num_directions
        self.num_vel_discrete = num_velocities
        self.u_range = agent_u_range
        self.agent_idx = agent_idx
        self.max_arrow_length = max_arrow_length
        self.arrow_color = arrow_color
        self.arrow_alpha = arrow_alpha
        self.arrow_width = arrow_width
        self.head_width = head_width
        self.head_length = head_length
        self.min_prob_threshold = min_prob_threshold

        if action_vectors is not None:
            norms = np.linalg.norm(action_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.unit_vectors = action_vectors / norms
            self.probabilities: NDArray[np.float64] | None = np.ones(
                len(action_vectors)
            ) / len(action_vectors)
        else:
            self.unit_vectors = None
            self.probabilities = None

        self.action_combinations: NDArray[np.integer] | None = None
        self.joint_probabilities: NDArray[np.float64] | None = None

    def set_probabilities(self, probabilities: NDArray[np.float64]) -> None:
        if self.action_vectors is not None and len(probabilities) != len(
            self.action_vectors
        ):
            raise ValueError(
                f"Probabilities length ({len(probabilities)}) must match "
                f"action_vectors length ({len(self.action_vectors)})"
            )
        self.probabilities = probabilities

    def set_multi_discrete_probabilities(
        self,
        action_combinations: NDArray[np.integer],
        joint_probabilities: NDArray[np.float64],
    ) -> None:
        self.action_combinations = action_combinations
        self.joint_probabilities = joint_probabilities

        direction_idx = action_combinations[:, 0]
        vel_idx = action_combinations[:, 1]

        angle = (direction_idx / self.num_directions) * 2 * np.pi
        magnitude = (vel_idx / self.num_vel_discrete) * self.u_range

        x_vel = magnitude * np.cos(angle)
        y_vel = magnitude * np.sin(angle)
        self.action_vectors = np.column_stack([x_vel, y_vel])

        norms = np.linalg.norm(self.action_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.unit_vectors = self.action_vectors / norms

    def render(self, env: Any, **kwargs: Any) -> Any:
        probabilities = kwargs.get("probabilities", self.probabilities)
        agent_idx = kwargs.get("agent_idx", self.agent_idx)

        if not hasattr(env, "world"):
            return

        world: World = env.world  # type: ignore
        if agent_idx >= len(world.agents):
            return

        if self.action_vectors is None:
            return

        agent = world.agents[agent_idx]
        agent_pos = np.asarray(agent.state.pos, dtype=np.float64).reshape(-1)
        if agent_pos.size < 2:
            return

        fig = env.fig
        ax = env.ax

        if self.joint_probabilities is not None:
            probabilities = self.joint_probabilities

        if probabilities is None:
            return

        if len(probabilities) != len(self.action_vectors):
            raise ValueError(
                f"Probabilities length ({len(probabilities)}) must match "
                f"action_vectors length ({len(self.action_vectors)})"
            )

        for action_vector, prob in zip(self.action_vectors, probabilities):
            if prob < self.min_prob_threshold:
                continue

            offset = np.asarray(action_vector, dtype=np.float64).reshape(-1) / 5
            if offset.size < 2:
                continue

            alpha = float(np.clip(prob, 0.0, 1.0))
            point_x = agent_pos[0] + offset[0]
            point_y = agent_pos[1] + offset[1]

            # ax.scatter(
            #     [point_x],
            #     [point_y],
            #     s=30.0 + 90.0 * float(prob),
            #     c=[self.arrow_color],
            #     alpha=alpha,
            #     edgecolors="none",
            #     zorder=10,
            # )

            # Draw arrow from agent position
            scaled_prob = (prob - 0.5) / 3 + 0.5
            ax.arrow(
                agent_pos[0],
                agent_pos[1],
                offset[0],
                offset[1],
                color=self.arrow_color,
                alpha=self.arrow_alpha
                * scaled_prob,  # More transparent for lower probabilities
                width=self.arrow_width,
                head_width=self.head_width,
                head_length=self.head_length,
                length_includes_head=True,
                zorder=10,
            )

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgb_array = np.asarray(buf)[:, :, :3]

        env.fig = fig
        env.ax = ax
        return rgb_array


class GaussianActionDistributionRenderer(PostRenderer):
    """Render squashed Gaussian action distributions for continuous policies.

    For each configured distribution, this renderer draws:
    - an arrow for the mean action vector (first two action dimensions), and
    - an axis-aligned ellipse centered at the arrow tip with radii from std.
    """

    def __init__(
        self,
        agent_idx: int = 0,
        arrow_width: float = 0.03,
        head_width: float = 0.12,
        head_length: float = 0.12,
        std_alpha: float = 0.18,
        min_std: float = 0.02,
        angle_arrow_length: float = 1.0,
    ) -> None:
        self.agent_idx = agent_idx
        self.arrow_width = arrow_width
        self.head_width = head_width
        self.head_length = head_length
        self.std_alpha = std_alpha
        self.min_std = min_std
        self.angle_arrow_length = angle_arrow_length
        self.distributions: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.distributions = []

    def set_distribution(
        self,
        mean: NDArray[np.floating],
        std: NDArray[np.floating],
        label: str,
        color: str,
    ) -> None:
        self.distributions.append(
            {
                "mean": np.asarray(mean, dtype=np.float64).reshape(-1),
                "std": np.asarray(std, dtype=np.float64).reshape(-1),
                "label": label,
                "color": color,
            }
        )

    def render(self, env: Any, **kwargs: Any) -> Any:
        if not hasattr(env, "world"):
            return

        world: World = env.world  # type: ignore
        if self.agent_idx >= len(world.agents):
            return

        agent = world.agents[self.agent_idx]
        agent_pos = np.asarray(agent.state.pos, dtype=np.float64).reshape(-1)
        if agent_pos.size < 2:
            return

        fig = env.fig
        ax = env.ax

        info_y = 0.98
        for dist in self.distributions:
            mean = dist["mean"]
            std = np.maximum(dist["std"], self.min_std)
            color = dist["color"]
            label = dist["label"]

            if mean.size == 0:
                continue

            if mean.size == 1:
                theta = float(mean[0])
                theta_std = float(std[0])

                mean_x = self.angle_arrow_length * np.cos(theta)
                mean_y = self.angle_arrow_length * np.sin(theta)

                ax.arrow(
                    agent_pos[0],
                    agent_pos[1],
                    mean_x,
                    mean_y,
                    color=color,
                    alpha=0.9,
                    width=self.arrow_width,
                    head_width=self.head_width,
                    head_length=self.head_length,
                    length_includes_head=True,
                    zorder=10,
                )

                wedge = Wedge(
                    center=(agent_pos[0], agent_pos[1]),
                    r=self.angle_arrow_length,
                    theta1=np.degrees(theta - theta_std),
                    theta2=np.degrees(theta + theta_std),
                    width=0.15 * self.angle_arrow_length,
                    edgecolor=color,
                    facecolor=color,
                    alpha=self.std_alpha,
                    linewidth=1.2,
                    zorder=9,
                )
                ax.add_patch(wedge)
            else:
                mean_x = float(mean[0])
                mean_y = float(mean[1])
                std_x = float(std[0])
                std_y = float(std[1]) if std.size > 1 else self.min_std

                ax.arrow(
                    agent_pos[0],
                    agent_pos[1],
                    mean_x,
                    mean_y,
                    color=color,
                    alpha=0.9,
                    width=self.arrow_width,
                    head_width=self.head_width,
                    head_length=self.head_length,
                    length_includes_head=True,
                    zorder=10,
                )

                ellipse = Ellipse(
                    xy=(agent_pos[0] + mean_x, agent_pos[1] + mean_y),
                    width=2.0 * std_x,
                    height=2.0 * std_y,
                    edgecolor=color,
                    facecolor=color,
                    alpha=self.std_alpha,
                    linewidth=1.5,
                    zorder=9,
                )
                ax.add_patch(ellipse)

            mean_txt = np.array2string(mean[: min(3, mean.size)], precision=2)
            std_txt = np.array2string(std[: min(3, std.size)], precision=2)
            ax.text(
                0.02,
                info_y,
                f"{label}: μ={mean_txt}, σ={std_txt}",
                transform=ax.transAxes,
                fontsize=8,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.8,
                },
                va="top",
                zorder=11,
            )
            info_y -= 0.06

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgb_array = np.asarray(buf)[:, :, :3]

        env.fig = fig
        env.ax = ax
        return rgb_array
