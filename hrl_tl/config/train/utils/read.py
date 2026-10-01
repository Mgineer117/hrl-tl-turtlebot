import os

from pydantic import BaseModel, computed_field


class ConfigPathConfig(BaseModel):
    config_dir: str
    env_config_file: str
    eval_env_config_file: str | None = None
    rl_config_file: str
    wrapper_config_file: str | None = None
