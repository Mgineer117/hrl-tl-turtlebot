from collections.abc import Mapping
import os
from typing import Any, Literal

import numpy as np
from gym_tl_tools import (
    RewardConfig,
    TLObservationReward,
    replace_special_characters,
)
from pydantic import BaseModel, Field
from rl_pipeline.core import ConfigReader, SaveConfig, YAMLReaderMixin
from rl_pipeline.core.utils.io import get_class, read_config_dict_from_yaml
from rl_pipeline.gymnasium import WrapperConfig
from rl_pipeline.sb3 import SB3PipelineConfig, SB3PipelineConfigReader
from sb3_hrl.option import (
    BaseOption,
    MetaControllerPrimitiveStepTimeLimitWrapper,
    SubpolicyTrainingWrapper,
)
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.wrappers.gc_ltl import FixedGCLTLWrapper, GCLTLWrapper, Predicate
from hrl_tl.wrappers.low_level_policies.base import LowLevelPolicy
from hrl_tl.wrappers.tl_high_level import TLHighLevelWrapper
from hrl_tl.wrappers.utils.spec_rep import SpecRep


class TLObservationRewardConfigReader(
    BaseModel, ConfigReader[WrapperConfig], YAMLReaderMixin
):
    tl_spec: str = ""
    atomic_predicates: list[Predicate]
    var_value_info_generator_cls: str
    var_value_info_generator_args: dict[str, Any] = {}
    reward_config: RewardConfig = RewardConfig()
    early_termination: bool = True
    dict_aut_state_key: str = "aut_state"

    def to_config(self) -> WrapperConfig:
        var_value_info_generator_cls: type = get_class(
            self.var_value_info_generator_cls
        )
        var_value_info_generator = var_value_info_generator_cls(
            **self.var_value_info_generator_args
        )
        wrapper_config = WrapperConfig(
            wrapper_class=TLObservationReward,
            wrapper_kwargs={
                "tl_spec": self.tl_spec,
                "atomic_predicates": self.atomic_predicates,
                "var_value_info_generator": var_value_info_generator,
                "reward_config": self.reward_config.model_dump(),
                "early_termination": self.early_termination,
                "dict_aut_state_key": self.dict_aut_state_key,
            },
        )
        return wrapper_config


class TLHighLevelWrapperConfigReader(BaseModel, ConfigReader[WrapperConfig]):
    spec_rep_class: str
    spec_rep_args: dict[str, Any]
    low_level_policy_class: str
    low_level_policy_args: dict[str, Any]
    max_low_level_policy_steps: int = 10
    all_formulae_file_path: str = "out/maze/all_formulae_2_cla_2_max_pred.json"
    invalid_tl_action: Literal["stay", "random"] = "random"
    tl_wrapper_args: str = "configs/fourroom/rl/tl_wrapper.yaml"
    excluded_obs_keys: list[str] = []
    verbose: bool = False

    def to_config(self) -> WrapperConfig:
        spec_rep_class: type[SpecRep] = get_class(self.spec_rep_class)
        low_level_policy_class: type[LowLevelPolicy] = get_class(
            self.low_level_policy_class
        )
        low_level_policy_args: dict[str, Any] = (
            low_level_policy_class.policy_args_reader(
                **self.low_level_policy_args
            )
            .to_config()
            .model_dump()
        )
        tl_wrapper_args: dict[str, Any] = (
            read_config_dict_from_yaml(
                os.path.dirname(self.tl_wrapper_args),
                os.path.basename(self.tl_wrapper_args),
                TLObservationRewardConfigReader,
            )
            .to_config()
            .wrapper_kwargs
        )

        wrapper_config = WrapperConfig(
            wrapper_class=TLHighLevelWrapper,
            wrapper_kwargs={
                "spec_rep_class": spec_rep_class,
                "spec_rep_args": self.spec_rep_args,
                "low_level_policy_class": low_level_policy_class,
                "low_level_policy_args": low_level_policy_args,
                "max_low_level_policy_steps": self.max_low_level_policy_steps,
                "all_formulae_file_path": self.all_formulae_file_path,
                "invalid_tl_action": self.invalid_tl_action,
                "excluded_obs_keys": self.excluded_obs_keys,
                "tl_wrapper_args": tl_wrapper_args,
                "verbose": self.verbose,
            },
        )
        return wrapper_config


class TLHighLevelPipelineConfigReader(SB3PipelineConfigReader):
    def _to_wrapper_config(
        self, replicate_signature: str = ""
    ) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_file = self.wrapper_config_file
            if "{replicate_signature}" in wrapper_config_file:
                sig = replicate_signature if replicate_signature else "rep0"
                candidate = os.path.join(
                    self.config_dir,
                    wrapper_config_file.format(replicate_signature=sig),
                )
                if not os.path.exists(candidate):
                    alt_sig = "rep_0" if sig == "rep0" else "rep0"
                    alt_candidate = os.path.join(
                        self.config_dir,
                        wrapper_config_file.format(replicate_signature=alt_sig),
                    )
                    if os.path.exists(alt_candidate):
                        sig = alt_sig
                wrapper_config_file = wrapper_config_file.format(
                    replicate_signature=sig
                )
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir,
                wrapper_config_file,
                TLHighLevelWrapperConfigReader,
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None


class TLSB3PipelineConfigReader(
    BaseModel, ConfigReader[SB3PipelineConfig], YAMLReaderMixin
):
    tl_spec: str = ""
    pipeline_config: SB3PipelineConfigReader

    def to_config(self) -> SB3PipelineConfig:
        device: str = (
            self.pipeline_config.device
            if isinstance(self.pipeline_config.device, str)
            else f"cuda:{self.pipeline_config.device}"
        )

        model_config_reader = self.pipeline_config._to_model_config_reader()

        save_config: SaveConfig = self._to_save_config()

        model_config = model_config_reader.to_config(save_config=save_config)

        pipeline_config = SB3PipelineConfig(
            device=device,
            experiment_id=self.pipeline_config.experiment_id,
            retrain_model=self.pipeline_config.retrain_model,
            save_config=save_config,
            env_config=self.pipeline_config._to_env_config(),
            wrapper_config=self._to_wrapper_config(),
            vec_config=model_config.vec_config,
            algo_config=model_config.algo_config,
            learn_config=model_config.learn_config,
            callback_config=model_config.callback_config,
            experiment_manager_config=self.pipeline_config._to_manager_config(),
        )

        return pipeline_config

    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.pipeline_config.wrapper_config_file:
            wrapper_config_reader: TLObservationRewardConfigReader = (
                read_config_dict_from_yaml(
                    self.pipeline_config.config_dir,
                    self.pipeline_config.wrapper_config_file,
                    TLObservationRewardConfigReader,
                )
            )
            wrapper_config_reader.tl_spec = self.tl_spec
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None

    def _to_save_config(self, replicate_signature: str = "") -> SaveConfig:
        wrapper_config: WrapperConfig | None = self._to_wrapper_config()
        assert wrapper_config is not None
        tl_spec: str = replace_special_characters(
            wrapper_config.wrapper_kwargs["tl_spec"]
        )
        return self.pipeline_config.save_config.to_config(
            experiment_id=self.pipeline_config.experiment_id,
            model_name_suffix=tl_spec,
        )


class GCLTLWrapperConfigReader(
    BaseModel, ConfigReader[WrapperConfig], YAMLReaderMixin
):
    predicates: list[Predicate]
    var_value_info_generator_cls: str
    var_value_info_generator_args: dict[str, Any] = {}
    early_termination: bool = True
    step_penalty: float = Field(
        0.05, ge=0.0, description="Penalty applied at each step."
    )
    reward_type: Literal["sparse", "dense"] = "sparse"
    dense_reward_scale: float = 12.0
    goal_rep: Literal["index", "one_hot"] = "index"

    def to_config(self) -> WrapperConfig:
        var_value_info_generator_cls: type = get_class(
            self.var_value_info_generator_cls
        )
        var_value_info_generator = var_value_info_generator_cls(
            **self.var_value_info_generator_args
        )
        wrapper_config = WrapperConfig(
            wrapper_class=GCLTLWrapper,
            wrapper_kwargs={
                "predicates": self.predicates,
                "var_value_info_generator": var_value_info_generator,
                "early_termination": self.early_termination,
                "step_penalty": self.step_penalty,
                "reward_type": self.reward_type,
                "dense_reward_scale": self.dense_reward_scale,
                "goal_rep": self.goal_rep,
            },
        )
        return wrapper_config


class FixedGoalGCLTLWrapperConfigReader(GCLTLWrapperConfigReader):
    goal_pred_name: str

    def to_config(self) -> WrapperConfig:
        var_value_info_generator_cls: type = get_class(
            self.var_value_info_generator_cls
        )
        var_value_info_generator = var_value_info_generator_cls(
            **self.var_value_info_generator_args
        )
        wrapper_config = WrapperConfig(
            wrapper_class=FixedGCLTLWrapper,
            wrapper_kwargs={
                "predicates": self.predicates,
                "var_value_info_generator": var_value_info_generator,
                "early_termination": self.early_termination,
                "step_penalty": self.step_penalty,
                "reward_type": self.reward_type,
                "dense_reward_scale": self.dense_reward_scale,
                "goal_pred_name": self.goal_pred_name,
            },
        )
        return wrapper_config


class GCLTLPipelineConfigReader(SB3PipelineConfigReader):
    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir,
                self.wrapper_config_file,
                GCLTLWrapperConfigReader,
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None


class FixedGCLTLPipelineConfigReader(SB3PipelineConfigReader):
    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir,
                self.wrapper_config_file,
                FixedGoalGCLTLWrapperConfigReader,
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None


class SubpolicyTrainingWrapperConfigReader(
    BaseModel, ConfigReader[WrapperConfig], YAMLReaderMixin
):
    intrinsic_reward_cls: str
    intrinsic_reward_args: dict[str, Any] = {}
    termination_condition: None = None

    def to_config(self) -> WrapperConfig:
        wrapper_config = WrapperConfig(
            wrapper_class=SubpolicyTrainingWrapper,
            wrapper_kwargs={
                "intrinsic_reward_cls": get_class(self.intrinsic_reward_cls),
                "intrinsic_reward_args": self.intrinsic_reward_args,
                "termination_condition": self.termination_condition,
            },
        )
        return wrapper_config


class AlloSB3PipelineConfigReader(
    BaseModel, ConfigReader[SB3PipelineConfig], YAMLReaderMixin
):
    eig_idx: int = 0
    rep_idx: int = 0
    reverse_reward: bool = False
    device: str = "cpu"
    pipeline_config: SB3PipelineConfigReader
    algo_kwargs: dict[str, Any] = Field(default_factory=dict)

    def to_config(self) -> SB3PipelineConfig:
        device: str = (
            self.pipeline_config.device
            if isinstance(self.pipeline_config.device, str)
            else f"cuda:{self.pipeline_config.device}"
        )

        model_config_reader = self.pipeline_config._to_model_config_reader()

        save_config: SaveConfig = self._to_save_config()

        model_config = model_config_reader.to_config(save_config=save_config)
        if self.algo_kwargs:
            model_config.algo_config.algo_kwargs.update(self.algo_kwargs)

        pipeline_config = SB3PipelineConfig(
            device=device,
            experiment_id=self.pipeline_config.experiment_id,
            retrain_model=self.pipeline_config.retrain_model,
            save_config=save_config,
            env_config=self.pipeline_config._to_env_config(),
            wrapper_config=self._to_wrapper_config(),
            vec_config=model_config.vec_config,
            algo_config=model_config.algo_config,
            learn_config=model_config.learn_config,
            callback_config=model_config.callback_config,
            experiment_manager_config=self.pipeline_config._to_manager_config(),
        )

        return pipeline_config

    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.pipeline_config.wrapper_config_file:
            wrapper_config_reader: SubpolicyTrainingWrapperConfigReader = (
                read_config_dict_from_yaml(
                    self.pipeline_config.config_dir,
                    self.pipeline_config.wrapper_config_file,
                    SubpolicyTrainingWrapperConfigReader,
                )
            )
            wrapper_config_reader.intrinsic_reward_args.update(
                {
                    "eig_idx": self.eig_idx,
                    "reverse_reward": self.reverse_reward,
                    "rep_idx": self.rep_idx,
                    "device": self.device,
                }
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None

    def _to_save_config(self, replicate_signature: str = "") -> SaveConfig:
        return self.pipeline_config.save_config.to_config(
            experiment_id=self.pipeline_config.experiment_id,
            model_name_suffix=f"eig{self.eig_idx}"
            + ("_rev" if self.reverse_reward else ""),
        )


class TLOption(BaseOption[dict[str, Any], Any]):
    """Option wrapper that injects an initial automaton state into dictionary observations.

    Used specifically for temporal logic subpolicies that were trained with
    TLObservationReward and require an `aut_state` key in observation dictionaries.
    """

    def __init__(
        self,
        default_aut_state: int = 0,
        policy: Any | None = None,
        policy_cls: type | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            policy=policy, policy_cls=policy_cls, policy_kwargs=policy_kwargs
        )
        self.default_aut_state: int = default_aut_state

    def predict(
        self,
        obs: dict[str, Any],
        deterministic: bool = True,
    ) -> Any:
        """Injects default aut_state into obs dictionary if absent and predicts primitive action."""
        if "aut_state" not in obs:
            obs = {**obs, "aut_state": np.int64(self.default_aut_state)}
        return super().predict(obs, deterministic=deterministic)


class AlloMetaTrainingWrapperConfigReader(
    BaseModel, ConfigReader[WrapperConfig], YAMLReaderMixin
):
    max_episode_steps: int = 250
    option_policy_class: str = "stable_baselines3.PPO"
    option_policy_dir: str = "out/fr_cont/allo/subpolicies/7.b.a"
    option_policy_name: str = "final_model"
    device: str = "cuda:0"
    reward_type: Literal["smdp", "intra_option"] = "smdp"
    gamma: float = 0.99
    invalid_option_penalty: float = -1.0
    include_random_option: bool = True
    random_option_termination_steps: int = 1
    capture_primitive_transitions: bool | None = None
    max_option_steps: int = 50
    include_step_count_in_obs: bool = False

    def to_config(self) -> WrapperConfig:

        policy_class: BaseAlgorithm = get_class(self.option_policy_class)  # type: ignore
        # Search all the zip file paths in the `option_policy_dir` for the one that starts with `option_policy_name`
        all_policy_paths: list[str] = []
        for root, dirs, files in os.walk(self.option_policy_dir):
            for file in files:
                if file.startswith(self.option_policy_name) and file.endswith(
                    ".zip"
                ):
                    all_policy_paths.append(os.path.join(root, file))

        if len(all_policy_paths) == 0:
            raise FileNotFoundError(
                f"No policy file found in {self.option_policy_dir} starting with {self.option_policy_name}"
            )

        # Sort the file paths
        all_policy_paths.sort()

        options: list[BaseOption] = [
            BaseOption(
                policy_cls=policy_class,
                policy_kwargs={"path": policy_path, "device": self.device},
            )
            for policy_path in all_policy_paths
        ]

        wrapper_config = WrapperConfig(
            wrapper_class=MetaControllerPrimitiveStepTimeLimitWrapper,
            wrapper_kwargs={
                "max_episode_steps": self.max_episode_steps,
                "options": options,
                "reward_type": self.reward_type,
                "gamma": self.gamma,
                "invalid_option_penalty": self.invalid_option_penalty,
                "include_random_option": self.include_random_option,
                "random_option_termination_steps": self.random_option_termination_steps,
                "capture_primitive_transitions": self.capture_primitive_transitions,
                "max_option_steps": self.max_option_steps,
                "include_step_count_in_obs": self.include_step_count_in_obs,
            },
        )
        return wrapper_config


class AlloMetaPipelineConfigReader(SB3PipelineConfigReader):
    def _to_wrapper_config(
        self, replicate_signature: str = ""
    ) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_file = self.wrapper_config_file
            if "{replicate_signature}" in wrapper_config_file:
                wrapper_config_file = wrapper_config_file.format(
                    replicate_signature=replicate_signature
                )
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir,
                wrapper_config_file,
                AlloMetaTrainingWrapperConfigReader,
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None


class TLPretrainedMetaTrainingWrapperConfigReader(
    BaseModel, ConfigReader[WrapperConfig], YAMLReaderMixin
):
    max_episode_steps: int = 250
    option_policy_class: str = "sb3_soft.SDSAC"
    option_policy_dir: str = "out/fr_cont/baseline/subpolicies/rep_0"
    option_policy_name: str = "best_model"
    device: str = "cuda:0"
    reward_type: Literal["smdp", "intra_option"] = "smdp"
    gamma: float = 0.99
    invalid_option_penalty: float = -1.0
    include_random_option: bool = True
    random_option_termination_steps: int = 1
    capture_primitive_transitions: bool | None = None
    max_option_steps: int = 50
    include_step_count_in_obs: bool = False
    default_aut_state: int = 0

    def to_config(self) -> WrapperConfig:

        policy_class: BaseAlgorithm = get_class(self.option_policy_class)  # type: ignore
        # Search all the zip file paths in the `option_policy_dir` for the one that starts with `option_policy_name`
        all_policy_paths: list[str] = []
        for root, dirs, files in os.walk(self.option_policy_dir):
            for file in files:
                if file.startswith(self.option_policy_name) and file.endswith(
                    ".zip"
                ):
                    all_policy_paths.append(os.path.join(root, file))

        if len(all_policy_paths) == 0:
            raise FileNotFoundError(
                f"No policy file found in {self.option_policy_dir} starting with {self.option_policy_name}"
            )

        # Sort the file paths
        all_policy_paths.sort()

        options: list[BaseOption] = [
            TLOption(
                default_aut_state=self.default_aut_state,
                policy_cls=policy_class,
                policy_kwargs={"path": policy_path, "device": self.device},
            )
            for policy_path in all_policy_paths
        ]

        wrapper_config = WrapperConfig(
            wrapper_class=MetaControllerPrimitiveStepTimeLimitWrapper,
            wrapper_kwargs={
                "max_episode_steps": self.max_episode_steps,
                "options": options,
                "reward_type": self.reward_type,
                "gamma": self.gamma,
                "invalid_option_penalty": self.invalid_option_penalty,
                "include_random_option": self.include_random_option,
                "random_option_termination_steps": self.random_option_termination_steps,
                "capture_primitive_transitions": self.capture_primitive_transitions,
                "max_option_steps": self.max_option_steps,
                "include_step_count_in_obs": self.include_step_count_in_obs,
            },
        )
        return wrapper_config


class TLPretrainedMetaPipelineConfigReader(SB3PipelineConfigReader):
    def _to_wrapper_config(self) -> WrapperConfig | None:
        if self.wrapper_config_file:
            wrapper_config_reader = read_config_dict_from_yaml(
                self.config_dir,
                self.wrapper_config_file,
                TLPretrainedMetaTrainingWrapperConfigReader,
            )
            wrapper_config = wrapper_config_reader.to_config()
            return wrapper_config
        else:
            return None
