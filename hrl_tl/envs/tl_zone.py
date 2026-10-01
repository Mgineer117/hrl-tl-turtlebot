from typing import Any

import numpy as np
from gym_tl_tools import BaseVarValueInfoGenerator
from gymnasium import Env, Wrapper
from numpy.typing import NDArray


class ZoneVarValueInfoGenerator(
    BaseVarValueInfoGenerator[dict[str, NDArray[np.int64]], np.int64]
):
    """
    For ContGrid's RoomsEnv,
    """

    def get_var_values(
        self,
        env: Env[dict[str, NDArray[np.int64]], np.int64]
        | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64]
        | None,
        obs: dict[str, NDArray[np.int64]],
        info: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        env : Env or Wrapper
            The environment or wrapped environment.
        obs : NDArray[np.int64]
            The current observation.

        info : dict[str, Any] = {}
            Additional information from the environment.
            Assume it's always empty.

        Returns
        -------
        var_values : dict[str, Any]
            A dictionary containing the variable values.
            The dictionary contains the following keys:
            - 'd_y': Distance to the closest yellow zone.
            - 'd_r': Distance to the closest red zone.
            - 'd_w': Distance to the closest white zone.
            - 'd_b': Distance to the closest black zone.
        """

        d_y: float = float(np.min(np.linalg.norm(obs["yellow_dist"], axis=1)))
        d_r: float = float(np.min(np.linalg.norm(obs["red_dist"], axis=1)))
        d_w: float = float(np.min(np.linalg.norm(obs["white_dist"], axis=1)))
        d_b: float = float(np.min(np.linalg.norm(obs["black_dist"], axis=1)))

        return {"d_y": d_y, "d_r": d_r, "d_w": d_w, "d_b": d_b}
