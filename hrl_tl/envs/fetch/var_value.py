"""Variable value generator for temporal logic evaluation on Fetch Reach-Avoid."""

from __future__ import annotations

from typing import Any

import numpy as np
from gym_tl_tools import BaseVarValueInfoGenerator
from gymnasium import Env, Wrapper


class FetchVarValueInfoGenerator(
    BaseVarValueInfoGenerator[dict[str, np.ndarray], np.ndarray]
):
    """A variable value generator calculating Euclidean distances to Fetch zones."""

    def get_var_values(
        self,
        env: Env[dict[str, np.ndarray], np.ndarray]
        | Wrapper[
            dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray
        ]
        | None,
        obs: dict[str, np.ndarray],
        info: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute minimum distances to yellow, red, and white zones.

        Args:
            env: The environment or wrapped environment instance.
            obs: Observation dictionary containing zone displacement vectors.
            info: Optional auxiliary environment info dictionary.

        Returns:
            Dictionary containing 'd_y', 'd_r', and 'd_w' scalar distance values.
        """
        del env, info

        d_y: float = float(np.min(np.linalg.norm(obs["yellow_dist"], axis=1)))
        d_r: float = float(np.min(np.linalg.norm(obs["red_dist"], axis=1)))
        d_w: float = float(np.min(np.linalg.norm(obs["white_dist"], axis=1)))

        return {"d_y": d_y, "d_r": d_r, "d_w": d_w}
