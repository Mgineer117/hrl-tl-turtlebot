import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from hrl_tl.baseline.utils.rl import estimate_advantages

from .base import Base
from .layers.ppo_networks import PPO_Actor, PPO_Critic
from .tl_hrl import TLHRLLearner


class BaseHRLLearner(Base):
    def __init__(
        self,
        hl_actor: PPO_Actor,
        hl_critic: PPO_Critic,
        ll_actor: PPO_Actor,
        ll_critic: PPO_Critic,
        nupdates: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        frequency: int = 10,
        device: str = "cpu",
    ):
        super(TLHRLLearner, self).__init__()

        # constants
        self.name = "TL_HRL_Learner"
        self.device = device

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        self.nupdates = nupdates

        # trainable networks
        self.hl_actor = hl_actor
        self.hl_critic = hl_critic

        self.ll_actor = ll_actor
        self.ll_critic = ll_critic

        self.hl_optimizer = torch.optim.Adam(
            [
                {"params": self.hl_actor.parameters(), "lr": actor_lr},
                {"params": self.hl_critic.parameters(), "lr": critic_lr},
            ]
        )

        self.ll_optimizer = torch.optim.Adam(
            [
                {"params": self.ll_actor.parameters(), "lr": actor_lr},
                {"params": self.ll_critic.parameters(), "lr": critic_lr},
            ]
        )
        self.hl_lr_scheduler = LambdaLR(self.hl_optimizer, self.lr_lambda)
        self.ll_lr_scheduler = LambdaLR(self.ll_optimizer, self.lr_lambda)

        #
        self.frequency = frequency
        self.count = 0
        self.current_goal = None
        self.to(self.dtype).to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        # change goal every frequency
        # otherwise proceed with the current goal
        state = self.preprocess_state(state)
        if self.count % self.frequency == 0:
            self.current_goal, hl_metaData = self.hl_actor(state, deterministic)
            mask = True
        else:
            hl_metaData = {
                "probs": torch.tensor(float("nan")),
                "logprobs": torch.tensor(float("nan")),
                "entropy": torch.tensor(float("nan")),
                "dist": torch.tensor(float("nan")),
            }
            mask = False

        assert self.current_goal is not None
        state = torch.concatenate((state, self.current_goal), dim=-1)

        action, ll_metaData = self.ll_actor(state, deterministic)
        self.count += 1

        return [self.current_goal, action], {
            "hl_probs": hl_metaData["probs"],
            "hl_logprobs": hl_metaData["logprobs"],
            "hl_entropy": hl_metaData["entropy"],
            "hl_dist": hl_metaData["dist"],
            "ll_probs": ll_metaData["probs"],
            "ll_logprobs": ll_metaData["logprobs"],
            "ll_entropy": ll_metaData["entropy"],
            "ll_dist": ll_metaData["dist"],
            "mask": mask,
        }

    def learn(self, batch):
        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        next_states = self.preprocess_state(batch["next_states"])
        goals = self.preprocess_state(batch["goals"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])
        hl_old_logprobs = self.preprocess_state(batch["hl_logprobs"])
        ll_old_logprobs = self.preprocess_state(batch["ll_logprobs"])
        masks = self.preprocess_state(batch["masks"]).bool().squeeze()

        # === perform hl optimization === #
        # process batch
        hl_states = states[masks]
        hl_actions = goals[masks]
        hl_old_logprobs = hl_old_logprobs[masks]
        hl_rewards = self.hl_rewards(rewards, masks)
        hl_terminals = terminals[masks]

        hl_loss_dict, _, hl_update_time = self.learn_hl_policy(
            hl_states, hl_actions, hl_old_logprobs, hl_rewards, hl_terminals
        )

        # === perform ll optimization === #
        # process batch
        ll_states = torch.concatenate((states, goals), dim=-1)
        ll_actions = actions
        ll_rewards = self.ll_rewards(states, goals, next_states)
        ll_terminals = terminals

        ll_loss_dict, timesteps, ll_update_time = self.learn_ll_policy(
            ll_states, ll_actions, ll_old_logprobs, ll_rewards, ll_terminals
        )

        # === finishing up === #
        loss_dict = hl_loss_dict | ll_loss_dict
        update_time = hl_update_time + ll_update_time

        return loss_dict, timesteps, update_time

    def learn_hl_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        """
        For each subgoal generated, we need to process the rewards achieved by the
        low-level policy into a single scaler. Based on this, the hl-policy is trained.
        """
        self.train()
        t0 = time.time()

        # Compute advantages and returns
        with torch.no_grad():
            values = self.hl_critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Mini-batch training
        batch_size = states.size(0)
        hl_minibatch_size = batch_size // self.num_minibatch

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        l2_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[:hl_minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = (
                    old_logprobs[indices],
                    returns[indices],
                )

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # 1. Critic Loss (with optional regularization)
                value_loss, l2_loss = self.critic_loss(
                    self.hl_critic, mb_states, mb_returns
                )
                # Track value loss for logging
                value_losses.append(value_loss.item())
                l2_losses.append(l2_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = (
                    self.actor_loss(
                        self.hl_actor,
                        mb_states,
                        mb_actions,
                        mb_old_logprobs,
                        mb_advantages,
                    )
                )

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                target_kl.append(kl_div.item())

                if kl_div.item() > self.target_kl:
                    break

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss + l2_loss
                losses.append(loss.item())

                # Update critic parameters
                self.hl_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.hl_actor, self.hl_critic],
                    ["hl_actor", "hl_critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.hl_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/hl_loss/loss": np.mean(losses),
            f"{self.name}/hl_loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/hl_loss/value_loss": np.mean(value_losses),
            f"{self.name}/hl_loss/l2_loss": np.mean(l2_losses),
            f"{self.name}/hl_loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/hl_analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/hl_analytics/klDivergence": target_kl[-1],
            f"{self.name}/hl_analytics/K-epoch": k + 1,
            f"{self.name}/hl_analytics/avg_rewards": torch.mean(rewards).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.hl_actor, self.hl_critic],
            ["hl_actor", "hl_critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0

        return loss_dict, timesteps, update_time

    def learn_ll_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Compute advantages and returns
        with torch.no_grad():
            values = self.ll_critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        l2_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = (
                    old_logprobs[indices],
                    returns[indices],
                )

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # 1. Critic Loss (with optional regularization)
                value_loss, l2_loss = self.critic_loss(
                    self.ll_critic, mb_states, mb_returns
                )
                # Track value loss for logging
                value_losses.append(value_loss.item())
                l2_losses.append(l2_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = (
                    self.actor_loss(
                        self.ll_actor,
                        mb_states,
                        mb_actions,
                        mb_old_logprobs,
                        mb_advantages,
                    )
                )

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                target_kl.append(kl_div.item())

                if kl_div.item() > self.target_kl:
                    break

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss + l2_loss
                losses.append(loss.item())

                # Update critic parameters
                self.ll_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.ll_actor, self.ll_critic],
                    ["ll_actor", "ll_critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.ll_optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        # Logging
        loss_dict = {
            f"{self.name}/ll_loss/loss": np.mean(losses),
            f"{self.name}/ll_loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/ll_loss/value_loss": np.mean(value_losses),
            f"{self.name}/ll_loss/l2_loss": np.mean(l2_losses),
            f"{self.name}/ll_loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/ll_analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/ll_analytics/klDivergence": target_kl[-1],
            f"{self.name}/ll_analytics/K-epoch": k + 1,
            f"{self.name}/ll_analytics/avg_rewards": torch.mean(rewards).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.ll_actor, self.ll_critic],
            ["ll_actor", "ll_critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        timesteps = self.num_minibatch * self.minibatch_size
        update_time = time.time() - t0

        return loss_dict, timesteps, update_time

    def actor_loss(
        self,
        actor: nn.Module,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        _, metaData = actor(mb_states)
        logprobs = actor.log_prob(metaData["dist"], mb_actions)
        entropy = actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
            * mb_advantages
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self._entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def critic_loss(
        self,
        critic: nn.Module,
        mb_states: torch.Tensor,
        mb_returns: torch.Tensor,
    ):
        mb_values = critic(mb_states)
        value_loss = self.mse_loss(mb_values, mb_returns)
        l2_loss = (
            sum(param.pow(2).sum() for param in critic.parameters())
            * self.l2_reg
        )

        return value_loss, l2_loss

    def hl_rewards(self, rewards: torch.Tensor, masks: torch.Tensor):
        """
        It computes the return by the low-level policy given eachg subgoal
        generated by the high-level policy.
        """
        r = 0
        j = 0
        reward_list = []
        for i in range(len(masks) - 1):
            if masks[i]:
                j = 0
                r = rewards[i]
            else:
                j += 1
                r += self.gamma**j * rewards[i]

            if masks[i + 1] == 1:
                reward_list.append(r)

        if rewards.shape[0] != torch.sum(masks):
            reward_list.append(rewards[i + 1])

        rewards = (
            torch.tensor(reward_list)
            .unsqueeze(-1)
            .to(self.device)
            .to(self.dtype)
        )

        assert rewards.shape[0] == torch.sum(masks)
        return rewards

    def ll_rewards(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        next_states: torch.Tensor,
    ):
        # process reward
        rewards = -torch.linalg.norm(
            states + goals - next_states, dim=1, keepdim=True
        )
        return rewards
