import gymnasium as gym
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from hrl_tl.baseline.log.wandb_logger import WandbLogger
from hrl_tl.baseline.trainer.base_trainer import Trainer
from hrl_tl.baseline.utils.sampler import Sampler
from hrl_tl.baseline.utils.utils import print_model_summary


class PPO:
    def __init__(
        self, env: gym.Env, logger: WandbLogger, writer: SummaryWriter, args
    ):
        """
        This is a naive PPO wrapper that includes all necessary training pipelines for HRL.
        This trains SF network and train PPO according to the extracted features by SF network
        """
        self.env = env

        self.sampler = Sampler(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            episode_len=args.episode_len,
            batch_size=int(args.num_minibatch * args.minibatch_size),
            cpu_preserve_rate=args.cpu_preserve_rate,
            num_cores=args.num_cores,
            verbose=False,
        )

        # object initialization
        self.logger = logger
        self.writer = writer
        self.args = args

        ### Call network param and run
        self.define_ppo_policy()

    def begin_training(self):
        print_model_summary(self.policy, model_name="PPO model")
        ppo_trainer = Trainer(
            env=self.env,
            policy=self.policy,
            sampler=self.sampler,
            logger=self.logger,
            writer=self.writer,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_episodes,
            seed=self.args.seed,
        )
        ppo_trainer.train()

    def define_ppo_policy(self):
        from ..models.layers.ppo_networks import PPO_Actor, PPO_Critic
        from ..models.ppo import PPO_Learner

        actor = PPO_Actor(
            input_dim=np.prod(self.args.state_dim),
            hidden_dim=self.args.actor_dim,
            action_dim=self.args.action_dim,
            activation=nn.Tanh(),
            is_discrete=self.args.is_discrete,
        )
        critic = PPO_Critic(
            input_dim=np.prod(self.args.state_dim),
            hidden_dim=self.args.critic_dim,
            activation=nn.Tanh(),
        )

        nupdates = int(
            self.args.timesteps
            / (self.args.num_minibatch * self.args.minibatch_size)
        )
        self.policy = PPO_Learner(
            actor=actor,
            critic=critic,
            nupdates=nupdates,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )
