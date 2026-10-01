"""
Tabular policy implementations for discrete state-action spaces.
"""

import numpy as np


class TabularGoalConditionedPolicy:
    """
    Tabular policy for goal-conditioned tasks using REINFORCE with baseline.

    The policy is represented as a table Q(s, g, a) where:
    - s: state (x, y position)
    - g: goal index
    - a: action
    """

    def __init__(
        self, n_states_x=3, n_states_y=3, n_goals=2, n_actions=4, lr=0.1
    ):
        """
        Initialize tabular policy.

        Args:
            n_states_x: Number of states in x dimension
            n_states_y: Number of states in y dimension
            n_goals: Number of possible goals
            n_actions: Number of actions
            lr: Learning rate for policy updates
        """
        self.n_states_x = n_states_x
        self.n_states_y = n_states_y
        self.n_goals = n_goals
        self.n_actions = n_actions
        self.lr = lr

        # Initialize policy parameters (logits)
        # Shape: (n_states_x, n_states_y, n_goals, n_actions)
        self.logits = np.zeros((n_states_x, n_states_y, n_goals, n_actions))

        # Initialize value function for baseline
        self.values = np.zeros((n_states_x, n_states_y, n_goals))

    def get_action_probs(self, state, goal):
        """Get action probabilities using softmax."""
        x, y = int(state[0]), int(state[1])
        goal_idx = int(goal)
        logits = self.logits[x, y, goal_idx]
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / exp_logits.sum()

    def sample_action(self, state, goal):
        """Sample action from policy."""
        probs = self.get_action_probs(state, goal)
        return np.random.choice(self.n_actions, p=probs)

    def get_value(self, state, goal):
        """Get value estimate for state-goal pair."""
        x, y = int(state[0]), int(state[1])
        goal_idx = int(goal)
        return self.values[x, y, goal_idx]

    def update(self, trajectory, gamma=0.99):
        """
        Update policy using REINFORCE with baseline.

        Args:
            trajectory: List of (state, goal, action, reward) tuples
            gamma: Discount factor
        """
        # Calculate returns
        returns = []
        G = 0
        for _, _, _, reward in reversed(trajectory):
            G = reward + gamma * G
            returns.insert(0, G)

        # Normalize returns for stability
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy and value function
        for t, (state, goal, action, reward) in enumerate(trajectory):
            x, y = int(state[0]), int(state[1])
            goal_idx = int(goal)

            # Get current value and advantage
            value = self.values[x, y, goal_idx]
            advantage = returns[t] - value

            # Update value function
            self.values[x, y, goal_idx] += self.lr * (returns[t] - value)

            # Update policy (REINFORCE with baseline)
            probs = self.get_action_probs(state, goal)
            grad_log_prob = np.zeros(self.n_actions)
            grad_log_prob[action] = 1.0
            grad_log_prob -= probs

            self.logits[x, y, goal_idx] += self.lr * advantage * grad_log_prob

    def save(self, filepath):
        """Save policy to file."""
        np.savez(filepath, logits=self.logits, values=self.values)
        print(f"Policy saved to {filepath}")

    def load(self, filepath):
        """Load policy from file."""
        data = np.load(filepath)
        self.logits = data["logits"]
        self.values = data["values"]
        print(f"Policy loaded from {filepath}")
