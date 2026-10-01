import gymnasium as gym
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from hrl_tl.baseline.log.wandb_logger import WandbLogger
from hrl_tl.baseline.trainer.hc_trainer import HCTrainer
from hrl_tl.baseline.utils.hc_sampler import HCSampler
from hrl_tl.baseline.utils.utils import print_model_summary


class TL_HRL:
    def __init__(
        self,
        env: gym.Env,
        logger: WandbLogger,
        writer: SummaryWriter,
        args,
    ):
        """
        TL_HRL
        """
        self.env = env

        self.sampler = HCSampler(
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
        self.define_policy()

    def begin_training(self):
        print_model_summary(self.policy, model_name="TL_HRL model")
        tlhrl_trainer = HCTrainer(
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
        tlhrl_trainer.train()

    def define_policy(self):
        from ..models.layers.ppo_networks import PPO_Actor, PPO_Critic

        # high-level-actor
        # note that it outputs the state (goal-state)
        hl_actor = PPO_Actor(
            input_dim=np.prod(self.args.state_dim),
            hidden_dim=self.args.actor_dim,
            action_dim=np.prod(self.args.state_dim),
            activation=nn.Tanh(),
            is_discrete=True,
        )
        hl_critic = PPO_Critic(
            input_dim=np.prod(self.args.state_dim),
            hidden_dim=self.args.critic_dim,
            activation=nn.Tanh(),
        )
        # low-level actor
        ll_actor = PPO_Actor(
            input_dim=int(2 * np.prod(self.args.state_dim)),
            hidden_dim=self.args.actor_dim,
            action_dim=self.args.action_dim,
            activation=nn.Tanh(),
            is_discrete=self.args.is_discrete,
        )
        ll_critic = PPO_Critic(
            input_dim=int(2 * np.prod(self.args.state_dim)),
            hidden_dim=self.args.critic_dim,
            activation=nn.Tanh(),
        )

        nupdates = int(
            self.args.timesteps
            / (self.args.num_minibatch * self.args.minibatch_size)
        )
        self.policy = TL_HRL_Learner(
            hl_actor=hl_actor,
            hl_critic=hl_critic,
            ll_actor=ll_actor,
            ll_critic=ll_critic,
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
            frequency=self.args.frequency,
            device=self.args.device,
        )
