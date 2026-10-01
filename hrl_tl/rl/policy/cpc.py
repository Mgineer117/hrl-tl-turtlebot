"""
Contrastive Policy Composition (CPC) for combining policies.
"""

import numpy as np


def lambda_weight(
    robustness: np.ndarray,
    L_gain: float = 3.0,
    k_steepness: float = 2.0,
    eps_margin: float = 0.5,
) -> np.ndarray:
    """
    Compute the sigmoid weight for the constraints in CPC policy.

    Parameters
    ----------
    robustness : np.ndarray
        The robustness value of the constraint (distance to obstacle).
    L_gain : float
        The gain of the sigmoid function.
    k_steepness : float
        The steepness of the sigmoid function.
    eps_margin : float
        The margin for the sigmoid function.

    Returns
    -------
    weight : np.ndarray
        The computed weight for the constraint.
    """
    weight = L_gain / (1.0 + np.exp(-k_steepness * (robustness + eps_margin)))
    return weight


class CPCTabularPolicy:
    """
    Contrastive Policy Composition wrapper for tabular goal-conditioned policies.

    This composes a policy to reach a goal while avoiding obstacles using CPC:
    log π̃(a|s) = log π(a|s,goal) - λ(s) log π(a|s,obstacle)
    """

    def __init__(
        self,
        base_policy,
        goal_idx: int,
        obstacle_idx: int,
        obstacle_pos: np.ndarray,
        L_gain: float = 3.0,
        k_steepness: float = 2.0,
        eps_margin: float = 0.5,
        use_binary_lambda: bool = True,
        lambda_threshold: float = 1.0,
    ):
        """
        Initialize CPC tabular policy.

        Args:
            base_policy: Trained goal-conditioned policy
            goal_idx: Index of the goal to reach
            obstacle_idx: Index of the obstacle to avoid
            obstacle_pos: Position of obstacle (defaults to [1, 1])
            L_gain: Lambda weight gain parameter
            k_steepness: Lambda weight steepness parameter
            eps_margin: Lambda weight margin parameter
            use_binary_lambda: If True, use binary lambda (0 or 1), else sigmoid
            lambda_threshold: Distance threshold for binary lambda
        """
        self.base_policy = base_policy
        self.goal_idx = goal_idx
        self.obstacle_idx = obstacle_idx
        self.obstacle_pos = obstacle_pos
        self.L_gain = L_gain
        self.k_steepness = k_steepness
        self.eps_margin = eps_margin
        self.use_binary_lambda = use_binary_lambda
        self.lambda_threshold = lambda_threshold

        # For visualization
        self.last_goal_probs = None
        self.last_obstacle_probs = None
        self.last_lambda = None
        self.last_composed_probs = None

    def compute_robustness(self, state: np.ndarray) -> float:
        """
        Compute robustness as the Manhattan distance to the obstacle.
        Negative distance means we're at the obstacle (unsafe).

        Args:
            state: Current state (x, y)

        Returns:
            Robustness value (distance to obstacle)
        """
        # Manhattan distance to obstacle
        distance = np.abs(state - self.obstacle_pos).sum()
        # Robustness is positive when away from obstacle, negative when at it
        # We subtract 0.5 so that being at the obstacle gives negative robustness
        robustness = distance - 0.5
        return robustness

    def compute_lambda(self, state: np.ndarray) -> float:
        """
        Compute lambda weight for the current state.

        Args:
            state: Current state (x, y)

        Returns:
            Lambda weight value
        """
        if self.use_binary_lambda:
            distance = np.abs(state - self.obstacle_pos).sum()
            return 1.0 if distance <= self.lambda_threshold else 0.0
        else:
            robustness = self.compute_robustness(state)
            return lambda_weight(
                np.array([robustness]),
                L_gain=self.L_gain,
                k_steepness=self.k_steepness,
                eps_margin=self.eps_margin,
            )[0]

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get composed action probabilities using CPC.

        Args:
            state: Current state (x, y)

        Returns:
            Action probabilities after composition
        """
        # Get action probabilities for reaching the goal
        goal_probs = self.base_policy.get_action_probs(state, self.goal_idx)

        # Get action probabilities for reaching the obstacle (what to avoid)
        obstacle_probs = self.base_policy.get_action_probs(
            state, self.obstacle_idx
        )

        # Compute lambda weight
        lambda_val = self.compute_lambda(state)

        # Store for visualization
        self.last_goal_probs = goal_probs.copy()
        self.last_obstacle_probs = obstacle_probs.copy()
        self.last_lambda = lambda_val

        # CPC composition in log space
        # log π̃(a|s) = log π(a|s,goal) - λ(s) log π(a|s,obstacle)
        epsilon = 1e-10  # Numerical stability
        goal_log_probs = np.log(goal_probs + epsilon)
        obstacle_log_probs = np.log(obstacle_probs + epsilon)

        composed_log_probs = goal_log_probs - lambda_val * obstacle_log_probs

        # Convert back to probabilities
        composed_probs = np.exp(composed_log_probs)
        composed_probs = composed_probs / composed_probs.sum()

        self.last_composed_probs = composed_probs.copy()

        return composed_probs

    def sample_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> int:
        """
        Sample action from the composed policy.

        Args:
            state: Current state (x, y)
            deterministic: If True, return argmax action

        Returns:
            Selected action
        """
        probs = self.get_action_probs(state)

        if deterministic:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(len(probs), p=probs))
