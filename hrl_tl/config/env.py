import importlib
from typing import Any, Callable, Generic, TypeVar

from gymnasium import Env
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    computed_field,
    model_serializer,
)
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from hrl_tl.utils.io import get_class

EnvKwargsType = TypeVar("EnvKwargsType", bound=BaseModel)
VecEnvKwargsType = TypeVar("VecEnvKwargsType")
MonitorKwargsType = TypeVar("MonitorKwargsType")
WrapperKwargsType = TypeVar("WrapperKwargsType", bound=BaseModel)


class EnvMakeConfig(BaseModel):
    id: str = "multigrid-rooms-v0"
    max_episode_steps: int | None
    disable_env_checker: bool | None = None
    env_kwargs: dict[str, Any] = {}

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_serializer
    def serialize(self, info: SerializationInfo) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        context = info.context
        if context:
            if context.get("flatten", False):
                return {
                    "id": self.id,
                    "max_episode_steps": self.max_episode_steps,
                    "disable_env_checker": self.disable_env_checker,
                    **self.env_kwargs,
                }

        return {
            "id": self.id,
            "max_episode_steps": self.max_episode_steps,
            "disable_env_checker": self.disable_env_checker,
            "env_kwargs": self.env_kwargs,
        }


class VecEnvMakeConfig(BaseModel):
    env_id: str = ""
    n_envs: int = 1
    seed: int | None = None
    start_index: int = 0
    monitor_dir: str | None = None
    wrapper_class: Callable[[Env], Env] | None = None
    env_kwargs: dict[str, Any] | None = None
    vec_env_cls_name: str | None = (
        "stable_baselines3.common.vec_env.SubprocVecEnv"
    )
    vec_env_kwargs: dict[str, Any] | None = None
    monitor_kwargs: dict[str, Any] | None = None
    wrapper_kwargs: dict[str, Any] | None = None

    @computed_field
    @property
    def vec_env_cls(self) -> type | None:
        # `self.vec_env_cls_name` a class name with the module prefix, e.g., "stable_baselines3.common.vec_env.SubprocVecEnv"
        return get_class(self.vec_env_cls_name)

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self, serializer, info: SerializationInfo
    ) -> dict[str, Any]:
        # Use the default serializer first
        output: dict[str, Any] = serializer(self)
        # Remove "cls_name" and "cls_map" fields from the output
        output.pop("wrapper_cls_name", None)
        output.pop("vec_env_cls_name", None)
        return output

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)
