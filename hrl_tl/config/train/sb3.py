import os
from typing import Any, Literal

import gymnasium as gym
import yaml
from gym_tl_tools import (
    TLObservationReward,
    TLObservationRewardConfig,
    replace_special_characters,
)
from gymnasium.core import Env, Wrapper
from pydantic import computed_field
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from hrl_tl.config.algo import SB3AlgorithmConfig
from hrl_tl.config.callbacks import CheckpointCallbackConfig, EvalCallbackConfig
from hrl_tl.config.env import EnvMakeConfig, VecEnvMakeConfig
from hrl_tl.eval.callback import SuccessEvalCallback, VideoRecorderCallback
from hrl_tl.utils.io import format_large_number, get_file_with_largest_number
from hrl_tl.utils.sb3.env import make_vec_env
from hrl_tl.wrappers.tl_high_level import (
    TLHighLevelWrapper,
    TLHighLevelWrapperConfig,
)

from .base import BaseTrainingConfig
from .utils.save import SaveConfig


class SB3BaseTrainingConfig(BaseTrainingConfig):
    gpu_id: int = 0
    experiment_id: str = ""
    retrain_model: bool = True
    save_config: SaveConfig = SaveConfig()
    config_dir: str = "configs/fourroom"
    env_config_file: str = "env/full_random.yaml"
    eval_env_config_file: str | None = None
    rl_config_file: str = "rl/ppo.yaml"

    @property
    def model_name_suffix(self) -> str:
        total_timesteps: int = self.learn_config_dict["total_timesteps"]
        return f"_{format_large_number(total_timesteps)}"

    @property
    def model_full_name(self) -> str:
        return self.save_config.model_name + self.model_name_suffix

    @computed_field
    @property
    def model_save_dir(self) -> str:
        return os.path.join(
            self.save_config.models_dir,
            self.experiment_id,
            self.model_full_name,
            self.save_config.replicate_dir,
        )

    @property
    def _model_filename(self) -> str:
        return self.save_config.model_filename.replace(
            ".zip",
            f"_{self.experiment_id}" + self.model_name_suffix + ".zip",
        )

    @computed_field
    @property
    def model_save_path(self) -> str:
        return os.path.join(self.model_save_dir, self._model_filename)

    @property
    def animation_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, "animation")

    @computed_field
    @property
    def animation_save_path(self) -> str:
        return os.path.join(
            self.animation_save_dir, self._model_filename
        ).replace(".zip", ".gif")

    @computed_field
    @property
    def monitor_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, self.save_config.monitor_dir)

    @computed_field
    @property
    def tb_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, self.save_config.tb_dir)

    @computed_field
    @property
    def eval_save_dir(self) -> str:
        return os.path.join(self.model_save_dir, self.save_config.eval_dir)

    @computed_field
    @property
    def eval_metrics_save_path(self) -> str:
        filename: str = (
            self.save_config.eval_metrics_filename.replace(
                ".yaml", f"{self.model_name_suffix}.yaml"
            )
            if self.save_config.include_extension
            else self.save_config.eval_metrics_filename
        )
        return os.path.join(self.eval_save_dir, filename)

    @property
    def env_config_dict(self) -> dict[str, Any]:
        with open(
            os.path.join(self.config_dir, self.env_config_file), "r"
        ) as f:
            return yaml.safe_load(f)

    @computed_field
    @property
    def env_config(self) -> EnvMakeConfig:
        return EnvMakeConfig(**self.env_config_dict)

    @property
    def eval_env_config_dict(self) -> dict[str, Any]:
        if self.eval_env_config_file is None:
            return self.env_config_dict
        else:
            with open(
                os.path.join(self.config_dir, self.eval_env_config_file), "r"
            ) as f:
                return yaml.safe_load(f)

    @computed_field
    @property
    def eval_env_config(self) -> EnvMakeConfig:
        return EnvMakeConfig(**self.eval_env_config_dict)

    def train_env(self) -> Env:
        return gym.make(**self.env_config.model_dump(context={"flatten": True}))

    def eval_env(self) -> Env:
        return gym.make(
            **self.eval_env_config.model_dump(context={"flatten": True})
        )

    @property
    def vec_config_dict(self) -> dict[str, Any]:
        with open(os.path.join(self.config_dir, self.rl_config_file), "r") as f:
            return yaml.safe_load(f)["vec_config"]

    @computed_field
    @property
    def vec_config(
        self,
    ) -> VecEnvMakeConfig:
        return self._get_vec_config("train")

    @computed_field
    @property
    def eval_vec_config(
        self,
    ) -> VecEnvMakeConfig:
        return self._get_vec_config("eval")

    def _get_vec_config(
        self, mode: Literal["train", "eval"]
    ) -> VecEnvMakeConfig:
        match mode:
            case "train":
                env_kwargs = self.env_config.env_kwargs
            case "eval":
                env_kwargs = self.eval_env_config.env_kwargs

        return VecEnvMakeConfig(
            **self.vec_config_dict,
            env_id=self.env_config.id,
            env_kwargs=env_kwargs,
            monitor_dir=self.monitor_save_dir,
        )

    def train_vec_env(self) -> VecEnv:
        return make_vec_env(**self.vec_config.model_dump())

    def eval_vec_env(self) -> VecEnv:
        return make_vec_env(**self.eval_vec_config.model_dump())

    @property
    def rl_config_dict(self) -> dict[str, Any]:
        with open(os.path.join(self.config_dir, self.rl_config_file), "r") as f:
            return yaml.safe_load(f)["rl_config"]

    @computed_field
    @property
    def algorithm_config(self) -> SB3AlgorithmConfig:
        with open(os.path.join(self.config_dir, self.rl_config_file), "r") as f:
            algo_name: str = yaml.safe_load(f)["algo_name"]
        return SB3AlgorithmConfig(
            algorithm_name=algo_name, algo_kwargs=self.rl_config_dict
        )

    def model(self, train_env: Env | Wrapper | VecEnv) -> BaseAlgorithm:
        algo_class = self.algorithm_config.algo_class
        return algo_class(
            **self.algorithm_config.algo_kwargs,
            env=train_env,
            tensorboard_log=self.tb_save_dir,
            device=f"cuda:{self.gpu_id}",
        )

    def load_model(
        self, env: Env | Wrapper | VecEnv | None = None, load_best: bool = False
    ) -> BaseAlgorithm:
        """Load the model from the saved path."""

        if load_best:
            dir_name, model_full_filename = os.path.split(self.model_save_path)
            model_filename_wo_ext = os.path.splitext(model_full_filename)[0]
            best_model_filename_wo_ext = os.path.splitext(
                self.save_config.best_model_filename
            )[0]
            filename = os.path.join(
                dir_name,
                model_full_filename.replace(
                    model_filename_wo_ext,
                    best_model_filename_wo_ext,
                ),
            )
        else:
            filename = self.model_save_path

        filename: str = get_file_with_largest_number(filename)

        return self.algorithm_config.algo_class.load(
            filename, env=env, device=f"cuda:{self.gpu_id}"
        )

    def load_checkpoint(
        self, timestep: int | Literal["latest"] = "latest"
    ) -> BaseAlgorithm:
        """Load the model checkpoint from the saved path."""
        print(f"Loading checkpoint at timestep {timestep}...")
        ckpt_callback_config: CheckpointCallbackConfig = (
            self.checkpoint_callback_config
        )
        # Search the available checkpoints
        ckpt_files: list[str] = [
            f
            for f in os.listdir(ckpt_callback_config.save_path)
            if f.startswith(ckpt_callback_config.name_prefix)
        ]

        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {ckpt_callback_config.save_path}"
            )
        if timestep == "latest":
            # Find the latest checkpoint
            ckpt_files.sort()
            ckpt_file = ckpt_files[-1]
        else:
            # Find the checkpoint with the specified timestep
            ckpt_file = None
            for f in ckpt_files:
                if f.endswith(f"{timestep}.zip"):
                    ckpt_file = f
                    break
            if ckpt_file is None:
                raise FileNotFoundError(
                    f"No checkpoint file found for timestep {timestep} in {ckpt_callback_config.save_path}",
                    f"Available files: {ckpt_files}",
                )
        print(f"- Found checkpoint file: {ckpt_file}")
        return self.algorithm_config.algo_class.load(
            os.path.join(ckpt_callback_config.save_path, ckpt_file),
            # env=self.train_env(),
            device=f"cuda:{self.gpu_id}",
        )

    @property
    def model_learn_config(self) -> dict[str, Any]:
        eval_dict = self.eval_callback_config.model_dump()
        eval_dict.update({"eval_env": self.eval_vec_env()})
        eval_callback = SuccessEvalCallback(**eval_dict)
        ckpt_callback = CheckpointCallback(
            **self.checkpoint_callback_config.model_dump()
        )
        video_callback = VideoRecorderCallback(
            eval_env=self.eval_env(),
            render_freq=self.checkpoint_callback_config.save_freq,
            save_dir=self.checkpoint_callback_config.save_path,
            name_prefix=self.checkpoint_callback_config.name_prefix,
        )
        config = self.learn_config_dict
        config.update(
            {"callback": [eval_callback, ckpt_callback, video_callback]}
        )
        return config

    @property
    def learn_config_dict(self) -> dict[str, Any]:
        with open(os.path.join(self.config_dir, self.rl_config_file), "r") as f:
            learn_config_dict: dict[str, Any] = yaml.safe_load(f)[
                "learn_config"
            ]
            return learn_config_dict

    @property
    def eval_callback_config(self) -> EvalCallbackConfig:
        with open(os.path.join(self.config_dir, self.rl_config_file), "r") as f:
            eval_callback_config_dict: dict[str, Any] = yaml.safe_load(f)[
                "eval_callback_config"
            ]

        eval_callback_config_dict.update(
            {
                "log_path": os.path.join(
                    self.model_save_dir, eval_callback_config_dict["log_path"]
                ),
                "best_model_save_path": self.model_save_dir,
                "deterministic": False,
                "render": False,
            }
        )
        return EvalCallbackConfig(**eval_callback_config_dict)

    @property
    def checkpoint_config_dict(self) -> dict[str, Any]:
        with open(os.path.join(self.config_dir, self.rl_config_file), "r") as f:
            return yaml.safe_load(f)["ckpt_callback_config"]

    @property
    def checkpoint_callback_config(self) -> CheckpointCallbackConfig:
        ckpt_callback_config_dict: dict[str, Any] = self.checkpoint_config_dict

        ckpt_callback_config_dict.update(
            {
                "save_path": os.path.join(
                    self.model_save_dir, ckpt_callback_config_dict["save_path"]
                )
            }
        )
        return CheckpointCallbackConfig(**ckpt_callback_config_dict)


class SB3LowLevelTrainingConfig(SB3BaseTrainingConfig):
    tl_spec: str = ""
    # gpu_id: int = 0
    # retrain_model: bool = True
    # models_dir: str = "out/fourroom/ltl_ll/ll_policies"
    # model_name: str = "fourroom_tl_ppo_test_rand"
    # model_filename: str = "final_model.zip"
    # monitor_dir: str = "monitor"
    # tb_dir: str = "tb"
    # config_dir: str = "configs/fourroom"
    # env_config_file: str = "env/full_random.yaml"
    # eval_env_config_file: str | None = None
    # rl_config_file: str = "rl/ppo_tl.yaml"
    tl_wrapper_config_file: str = "rl/tl_wrapper.yaml"

    @property
    def model_name_suffix(self) -> str:
        return f"_{replace_special_characters(self.tl_spec)}"

    @property
    def _model_filename(self) -> str:
        return self.save_config.model_filename

    @computed_field
    @property
    def animation_save_path(self) -> str:
        return os.path.join(
            self.animation_save_dir,
            self.save_config.model_filename.replace(
                ".zip", self.model_name_suffix + ".gif"
            ),
        )

    @property
    def tl_wrapper_config_dict(self) -> dict[str, Any]:
        with open(
            os.path.join(self.config_dir, self.tl_wrapper_config_file), "r"
        ) as f:
            return yaml.safe_load(f)

    @computed_field
    @property
    def tl_wrapper_config(self) -> TLObservationRewardConfig:
        config = self.tl_wrapper_config_dict
        config.update({"tl_spec": self.tl_spec})
        return TLObservationRewardConfig(**config)

    def train_env(self) -> Wrapper:
        return TLObservationReward(
            gym.make(**self.env_config.model_dump(context={"flatten": True})),
            **self.tl_wrapper_config.model_dump(),
        )

    def eval_env(self) -> Wrapper:
        return TLObservationReward(
            gym.make(
                **self.eval_env_config.model_dump(context={"flatten": True})
            ),
            **self.tl_wrapper_config.model_dump(),
        )

    def _get_vec_config(
        self, mode: Literal["train", "eval"]
    ) -> VecEnvMakeConfig:
        match mode:
            case "train":
                env_kwargs = self.env_config.env_kwargs
            case "eval":
                env_kwargs = self.eval_env_config.env_kwargs

        return VecEnvMakeConfig(
            **self.vec_config_dict,
            env_id=self.env_config.id,
            env_kwargs=env_kwargs,
            monitor_dir=self.monitor_save_dir,
            wrapper_class=TLObservationReward,  # type: ignore
            wrapper_kwargs=self.tl_wrapper_config.model_dump(),
        )


class SB3TLHRLTrainingConfig(SB3BaseTrainingConfig):
    experiment_id: str = ""
    all_formulae_file_path: str = ""
    hrl_wrapper_config_file: str = "hrl/tl_hrl_ppo_pretrained.yaml"

    @property
    def hrl_wrapper_config_dict(self) -> dict[str, Any]:
        with open(
            os.path.join(self.config_dir, self.hrl_wrapper_config_file), "r"
        ) as f:
            return yaml.safe_load(f)

    @computed_field
    @property
    def hrl_wrapper_config(
        self,
    ) -> TLHighLevelWrapperConfig[
        BaseAlgorithm,
        SB3LowLevelTrainingConfig,
    ]:
        return TLHighLevelWrapperConfig(**self.hrl_wrapper_config_dict)

    def train_env(self) -> Wrapper:
        return TLHighLevelWrapper(
            gym.make(**self.env_config.model_dump(context={"flatten": True})),
            spec_rep_class=self.hrl_wrapper_config.spec_rep_class,
            spec_rep_args=self.hrl_wrapper_config.spec_rep_args,
            low_level_policy_class=self.hrl_wrapper_config.low_level_policy_class,
            low_level_policy_args=self.hrl_wrapper_config.low_level_policy_args,
            max_low_level_policy_steps=self.hrl_wrapper_config.max_low_level_policy_steps,
            all_formulae_file_path=self.hrl_wrapper_config.all_formulae_file_path,
            invalid_tl_action=self.hrl_wrapper_config.invalid_tl_action,
            tl_wrapper_args=self.hrl_wrapper_config.tl_wrapper_config.model_dump(),
            verbose=self.hrl_wrapper_config.verbose,
        )

    def eval_env(self) -> Wrapper:
        return TLHighLevelWrapper(
            gym.make(
                **self.eval_env_config.model_dump(context={"flatten": True})
            ),
            spec_rep_class=self.hrl_wrapper_config.spec_rep_class,
            spec_rep_args=self.hrl_wrapper_config.spec_rep_args,
            low_level_policy_class=self.hrl_wrapper_config.low_level_policy_class,
            low_level_policy_args=self.hrl_wrapper_config.low_level_policy_args,
            max_low_level_policy_steps=self.hrl_wrapper_config.max_low_level_policy_steps,
            all_formulae_file_path=self.hrl_wrapper_config.all_formulae_file_path,
            invalid_tl_action=self.hrl_wrapper_config.invalid_tl_action,
            tl_wrapper_args=self.hrl_wrapper_config.tl_wrapper_config.model_dump(),
            verbose=self.hrl_wrapper_config.verbose,
        )

    def _get_vec_config(
        self, mode: Literal["train", "eval"]
    ) -> VecEnvMakeConfig:
        match mode:
            case "train":
                env_kwargs = self.env_config.env_kwargs
            case "eval":
                env_kwargs = self.eval_env_config.env_kwargs

        return VecEnvMakeConfig(
            **self.vec_config_dict,
            env_id=self.env_config.id,
            env_kwargs=env_kwargs,
            monitor_dir=self.monitor_save_dir,
            wrapper_class=TLHighLevelWrapper,  # type: ignore
            wrapper_kwargs=self.hrl_wrapper_config.model_dump(),
        )
