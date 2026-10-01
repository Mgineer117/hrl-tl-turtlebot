import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from hrl_tl.baseline.log.wandb_logger import WandbLogger
from hrl_tl.baseline.utils.utils import print_model_summary
from hrl_tl.baseline.utils.wrappers import HIROWrapper


class HIRO:
    def __init__(
        self,
        env: gym.Env,
        logger: WandbLogger,
        writer: SummaryWriter,
        args,
    ):
        """
        HIRO
        """
        self.env = HIROWrapper(env)

        # object initialization
        self.logger = logger
        self.writer = writer
        self.args = args

        ### Call network param and run
        self.define_policy()

    def begin_training(self):
        from hrl_tl.baseline.trainer.hiro_trainer import HiroTrainer

        print_model_summary(self.policy, model_name="HIRO model")
        hiro_trainer = HiroTrainer(
            env=self.env,
            policy=self.policy,
            logger=self.logger,
            writer=self.writer,
            epochs=int(self.args.timesteps // self.env._max_episode_steps),
            eval_num=1,
            seed=self.args.seed,
        )
        hiro_trainer.train()

    def define_policy(self):
        from ..models.hiro import HIRO_Learner

        self.policy = HIRO_Learner(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            goal_dim=self.args.state_dim,
            subgoal_dim=self.args.state_dim,
            actor_fc_dim=self.args.actor_fc_dim,
            critic_fc_dim=self.args.critic_fc_dim,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            is_discrete=self.args.is_discrete,
            # batch_size=self.args.batch_size,
            batch_size=256,
            gamma=self.args.gamma,
            device=self.args.device,
        )
