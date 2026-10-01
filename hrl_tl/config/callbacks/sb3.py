from gymnasium import Env
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3.common.vec_env import VecEnv


class EvalCallbackConfig(BaseModel):
    """
    Configuration for the Stable Baselines3 evaluation callback.
    """

    eval_env: Env | VecEnv | None = None
    eval_freq: int = Field(
        ge=0,
        description="""Frequency of evaluation in timesteps.
        When using multiple environments, each call to env.step() will effectively correspond to n_envs steps.
        To account for that, you can use eval_freq = max(eval_freq // n_envs, 1)""",
    )
    n_eval_episodes: int = Field(
        ge=1, description="Number of episodes to evaluate the model."
    )
    log_path: str = Field(
        default="eval",
        description="Path to the directory where evaluation logs will be saved.",
    )
    best_model_save_path: str = Field(
        default="best_model",
        description="Path to save the best model during evaluation.",
    )
    deterministic: bool = Field(
        default=False,
        description="Whether to use deterministic actions during evaluation.",
    )
    render: bool = Field(
        default=False,
        description="Whether to render the environment during evaluation.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointCallbackConfig(BaseModel):
    """
    Configuration for the Stable Baselines3 checkpoint callback.
    """

    save_freq: int = Field(
        ge=1, description="Frequency of saving the model in timesteps."
    )
    save_path: str = Field(
        default="ckpts", description="Path to save the checkpoints."
    )
    name_prefix: str = Field(
        default="ckpt", description="Prefix for the checkpoint filenames."
    )
    save_replay_buffer: bool = Field(
        default=True, description="Whether to save replay files."
    )
    verbose: int = Field(
        default=0, description="Verbosity level of the callback."
    )
