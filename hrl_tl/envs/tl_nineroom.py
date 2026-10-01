from typing import Any

import numpy as np
from gym_tl_tools import BaseVarValueInfoGenerator
from gymnasium import Env, Wrapper
from numpy.typing import NDArray


class NineRoomsVarValueInfoGenerator(
    BaseVarValueInfoGenerator[dict[str, NDArray[np.int64]], np.int64]
):
    """
    For ContGrid's NineRoomsEnv

    The default doorway positions are:
        "tl_tc": (6, 16),
        "tc_tr": (12, 14),
        "ml_mc": (6, 10),
        "mc_mr": (12, 8),
        "bl_bc": (6, 4),
        "bc_br": (12, 2),
        # Vertical doorways (gaps in horizontal walls at row 6 and row 12)
        "tl_ml": (2, 12),
        "tc_mc": (8, 12),
        "tr_mr": (16, 12),
        "ml_bl": (4, 6),
        "mc_bc": (10, 6),
        "mr_br": (14, 6),

    The grid is 19x19.

    """

    def get_var_values(
        self,
        env: Env[dict[str, NDArray[np.int64]], np.int64]
        | Wrapper[NDArray[np.int64], np.int64, NDArray[np.int64], np.int64]
        | None,
        obs: dict[str, NDArray[np.int64]],
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        room_scale: float = 1

        d_tl_tc: float = float(
            np.linalg.norm(obs["doorway_pos"][0] * room_scale)
        )
        d_tc_tr: float = float(
            np.linalg.norm(obs["doorway_pos"][1] * room_scale)
        )
        d_ml_mc: float = float(
            np.linalg.norm(obs["doorway_pos"][2] * room_scale)
        )
        d_mc_mr: float = float(
            np.linalg.norm(obs["doorway_pos"][3] * room_scale)
        )
        d_bl_bc: float = float(
            np.linalg.norm(obs["doorway_pos"][4] * room_scale)
        )
        d_bc_br: float = float(
            np.linalg.norm(obs["doorway_pos"][5] * room_scale)
        )
        d_tl_ml: float = float(
            np.linalg.norm(obs["doorway_pos"][6] * room_scale)
        )
        d_tc_mc: float = float(
            np.linalg.norm(obs["doorway_pos"][7] * room_scale)
        )
        d_tr_mr: float = float(
            np.linalg.norm(obs["doorway_pos"][8] * room_scale)
        )
        d_ml_bl: float = float(
            np.linalg.norm(obs["doorway_pos"][9] * room_scale)
        )
        d_mc_bc: float = float(
            np.linalg.norm(obs["doorway_pos"][10] * room_scale)
        )
        d_mr_br: float = float(
            np.linalg.norm(obs["doorway_pos"][11] * room_scale)
        )

        d_gl: float = float(np.linalg.norm(obs["goal_pos"] * room_scale))
        d_lv: float = float(
            np.min(np.linalg.norm(obs["lava_pos"] * room_scale, axis=1))
        )
        d_hl: float = float(
            np.min(np.linalg.norm(obs["hole_pos"] * room_scale, axis=1))
        )

        return {
            "d_tl_tc": d_tl_tc,
            "d_tc_tr": d_tc_tr,
            "d_ml_mc": d_ml_mc,
            "d_mc_mr": d_mc_mr,
            "d_bl_bc": d_bl_bc,
            "d_bc_br": d_bc_br,
            "d_tl_ml": d_tl_ml,
            "d_tc_mc": d_tc_mc,
            "d_tr_mr": d_tr_mr,
            "d_ml_bl": d_ml_bl,
            "d_mc_bc": d_mc_bc,
            "d_mr_br": d_mr_br,
            "d_gl": d_gl,
            "d_lv": d_lv,
            "d_hl": d_hl,
        }
