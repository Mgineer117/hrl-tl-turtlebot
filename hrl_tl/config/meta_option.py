from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from rl_pipeline.core import ConfigReader
from rl_pipeline.core.utils.io import get_class, read_config_dict_from_yaml
from rl_pipeline.gymnasium import WrapperConfig
from rl_pipeline.sb3 import (
    SB3ModelConfigReader,
    SB3PipelineConfig,
    SB3PipelineConfigReader,
)
from sb3_hrl.option.policies.primitive_step_ppo import PrimitiveStepPPO
from stable_baselines3 import PPO
from typing_extensions import override

from hrl_tl.config.wrapper import TLObservationRewardConfigReader
from hrl_tl.wrappers.low_level_policies import LowLevelPolicy
from hrl_tl.wrappers.tl_meta_option import (
    TLMetaOptionPrimitiveStepTimeLimitWrapper,
    TLMetaOptionWrapper,
)
from hrl_tl.wrappers.utils.spec_rep import SpecRep


class TLMetaOptionWrapperConfigReader(BaseModel, ConfigReader[WrapperConfig]):
    """A configuration reader for TLMetaOptionWrapper and PrimitiveStepTimeLimit."""

    spec_rep_class: str
    spec_rep_args: dict[str, Any]
    low_level_policy_class: str
    low_level_policy_args: dict[str, Any]
    max_low_level_policy_steps: int = 10
    all_formulae_file_path: str = "out/maze/all_formulae_2_cla_2_max_pred.json"
    invalid_tl_action: Literal["stay", "random"] = "random"
    tl_wrapper_args: str | dict[str, Any] = (
        "configs/fourroom/rl/tl_wrapper.yaml"
    )
    excluded_obs_keys: list[str] = Field(default_factory=list)
    verbose: bool = False
    max_episode_steps: int | None = None
    reward_type: Literal["smdp", "intra_option"] = "smdp"
    gamma: float = 0.99

    @override
    def to_config(self) -> WrapperConfig:
        """Constructs the WrapperConfig with resolved low-level policy and wrapper kwargs."""
        spec_rep_class: type[SpecRep[Any]] = get_class(self.spec_rep_class)
        low_level_policy_class: type[LowLevelPolicy[Any, Any, Any, Any]] = (
            get_class(self.low_level_policy_class)
        )
        low_level_policy_args: dict[str, Any] = (
            low_level_policy_class.policy_args_reader(
                **self.low_level_policy_args
            )
            .to_config()
            .model_dump()
        )
        if isinstance(self.tl_wrapper_args, str):
            tl_path = Path(self.tl_wrapper_args)
            tl_wrapper_args: dict[str, Any] = (
                read_config_dict_from_yaml(
                    str(tl_path.parent),
                    tl_path.name,
                    TLObservationRewardConfigReader,
                )
                .to_config()
                .wrapper_kwargs
            )
        else:
            tl_wrapper_args = self.tl_wrapper_args

        wrapper_kwargs: dict[str, Any] = {
            "spec_rep_class": spec_rep_class,
            "spec_rep_args": self.spec_rep_args,
            "low_level_policy_class": low_level_policy_class,
            "low_level_policy_args": low_level_policy_args,
            "max_low_level_policy_steps": self.max_low_level_policy_steps,
            "all_formulae_file_path": self.all_formulae_file_path,
            "excluded_obs_keys": self.excluded_obs_keys,
            "tl_wrapper_args": tl_wrapper_args,
            "reward_type": self.reward_type,
            "gamma": self.gamma,
            "verbose": self.verbose,
        }

        if self.max_episode_steps is not None:
            wrapper_kwargs["max_episode_steps"] = self.max_episode_steps
            wrapper_cls = TLMetaOptionPrimitiveStepTimeLimitWrapper
        else:
            wrapper_cls = TLMetaOptionWrapper

        return WrapperConfig(
            wrapper_class=wrapper_cls,
            wrapper_kwargs=wrapper_kwargs,
        )


class TLMetaOptionPipelineConfigReader(SB3PipelineConfigReader):
    """A pipeline configuration reader configuring option wrappers and PrimitiveStepPPO."""

    wrapper_kwargs: dict[str, Any] | None = None

    @override
    def _to_wrapper_config(
        self, replicate_signature: str = ""
    ) -> WrapperConfig | None:
        """Loads and resolves wrapper configuration with automatic max_episode_steps injection."""
        if not self.wrapper_config_file:
            return None

        wrapper_config_file = self.wrapper_config_file
        if "{replicate_signature}" in wrapper_config_file:
            sig = replicate_signature if replicate_signature else "rep0"
            candidate = Path(self.config_dir) / wrapper_config_file.format(
                replicate_signature=sig
            )
            if not candidate.exists():
                alt_sig = "rep_0" if sig == "rep0" else "rep0"
                alt_candidate = Path(
                    self.config_dir
                ) / wrapper_config_file.format(replicate_signature=alt_sig)
                if alt_candidate.exists():
                    sig = alt_sig
            wrapper_config_file = wrapper_config_file.format(
                replicate_signature=sig
            )

        env_config = self._to_env_config()
        max_episode_steps = env_config.max_episode_steps or 250

        wrapper_config_reader: TLMetaOptionWrapperConfigReader = (
            read_config_dict_from_yaml(
                self.config_dir,
                wrapper_config_file,
                TLMetaOptionWrapperConfigReader,
            )
        )
        if wrapper_config_reader.max_episode_steps is None:
            wrapper_config_reader.max_episode_steps = max_episode_steps

        wrapper_config = wrapper_config_reader.to_config()
        if self.wrapper_kwargs:
            from rl_pipeline.sb3.experiment.optuna import deep_update

            deep_update(wrapper_config.wrapper_kwargs, self.wrapper_kwargs)

        return wrapper_config

    @override
    def _to_model_config_reader(self) -> SB3ModelConfigReader:
        """Loads model config reader and ensures PrimitiveStepPPO is configured for PPO."""
        model_config_reader: SB3ModelConfigReader = (
            super()._to_model_config_reader()
        )
        if model_config_reader.algo_config.algorithm in (
            "PPO",
            "stable_baselines3.PPO",
        ):
            model_config_reader.algo_config.algorithm = (
                "sb3_hrl.option.policies.primitive_step_ppo.PrimitiveStepPPO"
            )
        return model_config_reader

    @override
    def to_config(self) -> SB3PipelineConfig:
        """Builds SB3PipelineConfig and ensures PrimitiveStepPPO is used for PPO models."""
        pipeline_config = super().to_config()
        if pipeline_config.algo_config.algorithm is PPO:
            pipeline_config.algo_config.algorithm = PrimitiveStepPPO
        return pipeline_config
