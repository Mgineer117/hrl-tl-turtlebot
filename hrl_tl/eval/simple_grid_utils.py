"""
Evaluation and visualization utilities for SimpleGrid environment.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def evaluate_policy(env, policy, n_eval_episodes=100, max_steps=50):
    """
    Evaluate goal-conditioned policy success rate.

    Args:
        env: Gymnasium environment
        policy: Policy with sample_action(state, goal) method
        n_eval_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        success_rate: Fraction of episodes that reached the goal
    """
    successes = 0

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        state = obs[:2]
        goal = obs[2]
        done = False

        for _ in range(max_steps):
            action = policy.sample_action(state, goal)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_obs[:2]

            if done:
                if terminated:  # Goal reached
                    successes += 1
                break

    return successes / n_eval_episodes


def evaluate_reach_avoid_policy(env, policy, n_eval_episodes=100, max_steps=50):
    """
    Evaluate reach-avoid policy.

    Args:
        env: Wrapped environment with reach-avoid rewards
        policy: Policy with sample_action(state, goal) method
        n_eval_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        success_rate: Fraction of episodes that reached the goal
        collision_rate: Fraction of episodes that hit the obstacle
    """
    successes = 0
    collisions = 0

    for _ in range(n_eval_episodes):
        obs, info = env.reset()
        state = obs[:2]
        goal = obs[2]
        done = False

        for _ in range(max_steps):
            action = policy.sample_action(state, goal)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_obs[:2]

            if done:
                if info.get("success", False):
                    successes += 1
                elif info.get("reached_obstacle", False):
                    collisions += 1
                break

    return successes / n_eval_episodes, collisions / n_eval_episodes


def save_animation_for_goal(
    env_class, policy, goal_idx, save_path, max_steps=50, n_attempts=10
):
    """
    Save animation for a specific goal with action probabilities visualization.

    Args:
        env_class: Environment class to instantiate
        policy: Trained goal-conditioned policy
        goal_idx: Goal index to create animation for (0 or 1)
        save_path: Path to save the animation
        max_steps: Maximum steps per episode
        n_attempts: Number of attempts to find episode with this goal
    """
    env = env_class(render_mode="rgb_array")

    # Try to get an episode with the desired goal
    frames = []
    action_probs_list = []
    actions_taken = []
    episode_found = False

    for attempt in range(n_attempts):
        obs, info = env.reset()
        state = obs[:2]
        goal = obs[2]

        if goal == goal_idx:
            episode_found = True
            total_reward = 0

            # Capture initial frame and action probabilities
            frames.append(env.render())
            action_probs = policy.get_action_probs(state, goal)
            action_probs_list.append(action_probs)
            actions_taken.append(None)  # No action taken yet

            for step in range(max_steps):
                action = policy.sample_action(state, goal)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Capture frame and action probabilities
                frames.append(env.render())
                state = next_obs[:2]
                action_probs = policy.get_action_probs(state, goal)
                action_probs_list.append(action_probs)
                actions_taken.append(action)

                if done:
                    # Add a few duplicate frames at the end to pause
                    for _ in range(5):
                        frames.append(env.render())
                        action_probs_list.append(action_probs)
                        actions_taken.append(action)
                    break

            break

    env.close()

    if not episode_found:
        print(
            f"Warning: Could not find episode with goal {goal_idx} after {n_attempts} attempts"
        )
        return

    # Create animation with subplots for environment and action probabilities
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax_env = fig.add_subplot(gs[0])
    ax_probs = fig.add_subplot(gs[1])

    ax_env.axis("off")
    im = ax_env.imshow(frames[0])

    # Action names for bar chart
    action_names = ["Up", "Down", "Left", "Right"]
    action_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def update(frame):
        # Update environment image
        im.set_array(frames[frame])
        goal_names = [
            "Center (1,1)",
            "Top-Right (2,2)",
            "Top-Left (0,2)",
            "Bottom-Center (1,0)",
        ]
        ax_env.set_title(
            f"Goal: {goal_names[goal_idx]} | Step {frame}/{len(frames) - 1}",
            fontsize=12,
            pad=10,
        )

        # Update action probabilities bar chart
        ax_probs.clear()
        probs = action_probs_list[frame]
        action_taken = actions_taken[frame]

        # Color bars based on whether action was taken
        colors = [
            action_colors[i] if i == action_taken else "#1f77b4"
            for i in range(len(action_names))
        ]

        bars = ax_probs.bar(
            action_names,
            probs,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Highlight the action that was taken
        if action_taken is not None:
            bars[action_taken].set_alpha(1.0)
            bars[action_taken].set_linewidth(3)

        ax_probs.set_ylim(0, 1)
        ax_probs.set_ylabel("Probability", fontsize=11)
        ax_probs.set_title("Action Probabilities", fontsize=12, pad=10)
        ax_probs.grid(axis="y", alpha=0.3, linestyle="--")

        # Add probability values on top of bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            label = f"{prob:.3f}"
            if i == action_taken:
                label = f"{prob:.3f} ✓"
            ax_probs.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == action_taken else "normal",
            )

        return [im] + list(bars)

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=200, blit=False
    )

    # Save as GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = PillowWriter(fps=1)
    anim.save(save_path, writer=writer)
    plt.close(fig)

    print(f"Animation for goal {goal_idx} saved to {save_path}")


def save_reach_avoid_animation(
    wrapped_env, policy, goal_idx, save_path, max_steps=50
):
    """
    Save animation for reach-avoid policy using the wrapped environment's renderer.

    Args:
        wrapped_env: ReachAvoidWrapper instance with render_mode='rgb_array'
        policy: Trained goal-conditioned policy
        goal_idx: Goal index
        save_path: Path to save the animation
        max_steps: Maximum steps per episode
    """
    frames = []
    action_probs_list = []
    actions_taken = []

    obs, info = wrapped_env.reset()
    state = obs[:2]
    goal = obs[2]

    # Capture initial frame and action probabilities
    frames.append(wrapped_env.render())
    action_probs = policy.get_action_probs(state, goal)
    action_probs_list.append(action_probs)
    actions_taken.append(None)  # No action taken yet

    for step in range(max_steps):
        action = policy.sample_action(state, goal)
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated

        # Capture frame and action probabilities
        frames.append(wrapped_env.render())
        state = next_obs[:2]
        action_probs = policy.get_action_probs(state, goal)
        action_probs_list.append(action_probs)
        actions_taken.append(action)

        if done:
            # Add a few duplicate frames at the end to pause
            for _ in range(5):
                frames.append(wrapped_env.render())
                action_probs_list.append(action_probs)
                actions_taken.append(action)
            break

    # Create animation with subplots for environment and action probabilities
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax_env = fig.add_subplot(gs[0])
    ax_probs = fig.add_subplot(gs[1])

    ax_env.axis("off")
    im = ax_env.imshow(frames[0])

    # Action names for bar chart
    action_names = ["Up", "Down", "Left", "Right"]
    action_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    def update(frame):
        # Update environment image
        im.set_array(frames[frame])
        ax_env.set_title(
            f"Reach-Avoid Task | Step {frame}/{len(frames) - 1}",
            fontsize=12,
            pad=10,
        )

        # Update action probabilities bar chart
        ax_probs.clear()
        probs = action_probs_list[frame]
        action_taken = actions_taken[frame]

        # Color bars based on whether action was taken
        colors = [
            action_colors[i] if i == action_taken else "#1f77b4"
            for i in range(len(action_names))
        ]

        bars = ax_probs.bar(
            action_names,
            probs,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Highlight the action that was taken
        if action_taken is not None:
            bars[action_taken].set_alpha(1.0)
            bars[action_taken].set_linewidth(3)

        ax_probs.set_ylim(0, 1)
        ax_probs.set_ylabel("Probability", fontsize=11)
        ax_probs.set_title("Action Probabilities", fontsize=12, pad=10)
        ax_probs.grid(axis="y", alpha=0.3, linestyle="--")

        # Add probability values on top of bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            label = f"{prob:.3f}"
            if i == action_taken:
                label = f"{prob:.3f} ✓"
            ax_probs.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if i == action_taken else "normal",
            )

        return [im] + list(bars)

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=200, blit=False
    )

    # Save as GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = PillowWriter(fps=1)
    anim.save(save_path, writer=writer)
    plt.close(fig)

    print(f"Reach-avoid animation saved to {save_path}")
