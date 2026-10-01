"""Fetch Reach-Avoid environment package."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

from hrl_tl.envs.fetch.config import (
    FetchActionConfig,
    FetchReachAvoidConfig,
    FetchRewardConfig,
    FetchSubtaskConfig,
    FetchZoneConfig,
)
from hrl_tl.envs.fetch.env import FetchReachAvoidEnv
from hrl_tl.envs.fetch.patch import apply_patch
from hrl_tl.envs.fetch.var_value import FetchVarValueInfoGenerator

apply_patch()

__all__ = [
    "FetchActionConfig",
    "FetchReachAvoidConfig",
    "FetchReachAvoidEnv",
    "FetchRewardConfig",
    "FetchSubtaskConfig",
    "FetchVarValueInfoGenerator",
    "FetchZoneConfig",
]

gym.register(
    id="FetchReachAvoid-v0",
    entry_point="hrl_tl.envs.fetch.env:FetchReachAvoidEnv",
    max_episode_steps=250,
)

gym.register(
    id="hrl_tl/FetchReachAvoid-v0",
    entry_point="hrl_tl.envs.fetch.env:FetchReachAvoidEnv",
    max_episode_steps=250,
)
