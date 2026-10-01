import os
from pathlib import Path
import subprocess
import sys
from typing import Any, TypedDict

from filelock import FileLock
from gymnasium.core import ActType, Env, ObsType
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field
from rl_pipeline.core import ConfigReader, YAMLReaderMixin
from rl_pipeline.sb3 import SB3Pipeline
from stable_baselines3.common.base_class import BaseAlgorithm
import yaml

from hrl_tl.config.train import SB3LowLevelTrainingConfig
from hrl_tl.config.wrapper import TLSB3PipelineConfigReader
from hrl_tl.utils.io import get_class
from hrl_tl.utils.sb3.replay import record_replay

from .base import LowLevelPolicy, LowLevelPolicyBuffer, TLObs


class SB3PolicyArgsDict(TypedDict):
    """A dictionary to hold the arguments for the low-level policy."""

    algorithm: type[BaseAlgorithm]
    algo_config: dict[str, Any]
    model_save_dir: str
    model_prefix: str
    model_name: str
    training_config: dict[str, Any]
    device: str


class TrainingConfig(BaseModel):
    total_timesteps: int = 50_000
    n_envs: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SB3PolicyArgs(BaseModel):
    """A Pydantic model to hold the arguments for the low-level policy."""

    algo_config_file: str = "configs/fourroom/rl/ppo_tl.yaml"
    model_save_dir: str = "out/maze/ltl_ll/ll_policies"
    model_prefix: str = "maze_tl_ppo_stay_"
    model_name: str = "final_model"
    training_config: TrainingConfig = TrainingConfig()
    device: str = "cuda:0"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def algo_config_dict(self) -> dict[str, Any]:
        with open(self.algo_config_file, "r") as f:
            return yaml.safe_load(f)

    @computed_field
    @property
    def algorithm(self) -> type[BaseAlgorithm]:
        algo_name: str = self.algo_config_dict["algo_name"]
        module_name = f"stable_baselines3.{algo_name.lower()}"
        algorithm: type[BaseAlgorithm] | None = get_class(module_name)
        if not algorithm:
            raise ValueError(
                f"Algorithm {algo_name} not found in {module_name}."
            )
        return algorithm

    @computed_field
    @property
    def algo_config(self) -> dict[str, Any]:
        algo_config = self.algo_config_dict["rl_config"]
        return algo_config


class SB3LowLevelPolicy(
    LowLevelPolicy[BaseAlgorithm, SB3LowLevelTrainingConfig, ObsType, NDArray]
):
    policy_args_validator = SB3LowLevelTrainingConfig
    """Abstract base class for Stable Baselines3 low-level policies."""

    def define_policy(
        self, policy_args: SB3LowLevelTrainingConfig
    ) -> BaseAlgorithm:
        if isinstance(policy_args, dict):
            policy_args = self.policy_args_validator.model_validate(policy_args)
        self.policy_args = policy_args
        self.policy_args.tl_spec = self.tl_spec

        if os.path.exists(self.policy_args.model_save_path):
            if self.policy_args.verbose:
                print(f"Loading model from {self.policy_args.model_save_path}")
            policy = self.policy_args.load_model(load_best=True)
        else:
            if self.policy_args.verbose:
                print(
                    f"Model not found at {self.policy_args.model_save_path}, training a new model."
                )
            vec_env = self.policy_args.train_vec_env()
            policy = self.policy_args.model(vec_env)
            policy.learn(**self.policy_args.model_learn_config)
            os.makedirs(self.policy_args.model_save_dir, exist_ok=True)
            policy.save(self.policy_args.model_save_path)

            record_replay(
                self.policy_args.eval_env(),
                policy,
                self.policy_args.animation_save_path,
            )

        return policy

    def act(
        self,
        obs: TLObs[ObsType],
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, ActType] | None = None,
        tl_wrapper_args: dict[str, Any] = {},
    ) -> NDArray:
        """Predict the action using the low-level policy."""
        action, _ = self.policy.predict(obs)
        return action


class SB3SubpolicyConfig(BaseModel):
    config_path: str = ""
    device: str = "cuda:0"
    available_devices: list[str] = ["cuda:0"]
    retrain_model: bool = False
    verbose: bool = False
    auto_train: bool = True


class SB3SubpolicyConfigReader(
    BaseModel, ConfigReader[SB3SubpolicyConfig], YAMLReaderMixin
):
    config_path: str = ""
    device: str = "cuda:0"
    available_devices: list[str] = ["cuda:0"]
    retrain_model: bool = False
    verbose: bool = False
    auto_train: bool = True

    def to_config(self) -> SB3SubpolicyConfig:
        return SB3SubpolicyConfig(
            config_path=self.config_path,
            device=self.device,
            available_devices=self.available_devices,
            retrain_model=self.retrain_model,
            verbose=self.verbose,
            auto_train=self.auto_train,
        )


class SB3SubpolicyBuffer(LowLevelPolicyBuffer[SB3SubpolicyConfig]):
    """Buffer that caches loaded subpolicies across environment resets."""

    def __init__(self, policy_args: SB3SubpolicyConfig) -> None:
        super().__init__(policy_args)
        self.policy_cache: dict[str, BaseAlgorithm] = {}

    def at_reset(self) -> None:
        """Preserves policy_cache across environment resets."""
        pass


class SB3Subpolicy(
    LowLevelPolicy[BaseAlgorithm, SB3SubpolicyConfig, ObsType, NDArray]
):
    policy_args_validator = SB3SubpolicyConfig
    policy_args_reader = SB3SubpolicyConfigReader
    buffer_class = SB3SubpolicyBuffer
    """Stable Baselines3 low-level subpolicy with on-demand training and multi-GPU caching."""

    def define_policy(self, policy_args: SB3SubpolicyConfig) -> BaseAlgorithm:
        if isinstance(policy_args, dict):
            policy_args = self.policy_args_validator.model_validate(policy_args)

        # 1. Check in-memory buffer cache
        if self.buffer is not None and isinstance(
            self.buffer, SB3SubpolicyBuffer
        ):
            if (
                self.tl_spec in self.buffer.policy_cache
                and not policy_args.retrain_model
            ):
                return self.buffer.policy_cache[self.tl_spec]

        # 2. Build subpolicy pipeline config to resolve model save paths
        config_reader = TLSB3PipelineConfigReader.from_yaml(
            policy_args.config_path
        )
        config_reader.tl_spec = self.tl_spec
        config_reader.pipeline_config.device = policy_args.device
        config_reader.pipeline_config.retrain_model = policy_args.retrain_model
        subpolicy_pipeline_config = config_reader.to_config()
        pipeline = SB3Pipeline(
            config=subpolicy_pipeline_config, verbose=policy_args.verbose
        )

        model_save_path = Path(pipeline.save_config.best_model_save_path)
        lock_dir = model_save_path.parent
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".train_{model_save_path.stem}.lock"

        # 3. Synchronize with FileLock across parallel workers
        with FileLock(str(lock_path)):
            if not model_save_path.exists() or policy_args.retrain_model:
                if not policy_args.auto_train:
                    raise FileNotFoundError(
                        f"Model not found at {model_save_path} and auto_train is False."
                    )
                if policy_args.verbose:
                    print(
                        f"Subpolicy for '{self.tl_spec}' not found at {model_save_path}. "
                        f"Training with SDSAC on {policy_args.device}..."
                    )

                cmd = [
                    sys.executable,
                    "scripts/baseline/train_single_subpolicy.py",
                    f"--tl_spec={self.tl_spec}",
                    f"--training_config_path={policy_args.config_path}",
                    f"--device={policy_args.device}",
                    f"--retrain_model={policy_args.retrain_model}",
                ]
                subprocess.run(cmd, check=True)

            policy = pipeline.load_model("best", device=policy_args.device)

        # 4. Store in buffer cache
        if self.buffer is not None and isinstance(
            self.buffer, SB3SubpolicyBuffer
        ):
            self.buffer.policy_cache[self.tl_spec] = policy

        return policy

    def act(
        self,
        obs: TLObs[ObsType],
        info: dict[str, Any] | None = None,
        current_env: Env[ObsType, ActType] | None = None,
        tl_wrapper_args: dict[str, Any] = {},
    ) -> NDArray:
        """Predict the action using the low-level policy."""
        action, _ = self.policy.predict(obs)
        return action
