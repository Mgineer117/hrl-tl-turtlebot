"""MuJoCo and NumPy 2.0 compatibility patch for Gymnasium Robotics."""

from __future__ import annotations

import mujoco
import numpy as np
from gymnasium_robotics.utils import mujoco_utils

_PATCHED = False


def _safe_get_joint_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> np.ndarray:
    """Return the joint position values (qpos) for the given joint name.

    Args:
        model: MuJoCo model descriptor.
        data: MuJoCo dynamic simulation state.
        name: Name of the joint.

    Returns:
        A NumPy slice of the joint position values.

    Raises:
        AssertionError: If the joint is not found in the model.
    """
    joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model."
    joint_type: int = int(model.jnt_type[joint_id])
    joint_addr: int = int(model.jnt_qposadr[joint_id])

    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 7
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        ndim = 4
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        ndim = 1

    return data.qpos[joint_addr : joint_addr + ndim]


def _safe_get_joint_qvel(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> np.ndarray:
    """Return the joint velocity values (qvel) for the given joint name.

    Args:
        model: MuJoCo model descriptor.
        data: MuJoCo dynamic simulation state.
        name: Name of the joint.

    Returns:
        A NumPy slice of the joint velocity values.

    Raises:
        AssertionError: If the joint is not found in the model.
    """
    joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model."
    joint_type: int = int(model.jnt_type[joint_id])
    joint_addr: int = int(model.jnt_dofadr[joint_id])

    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 6
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        ndim = 3
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        ndim = 1

    return data.qvel[joint_addr : joint_addr + ndim]


def _safe_set_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    value: np.ndarray | float,
) -> None:
    """Set the joint positions (qpos) of the model safely with NumPy 2.0.

    Args:
        model: MuJoCo model descriptor.
        data: MuJoCo dynamic simulation state.
        name: Name of the joint.
        value: Joint position array or scalar to set.

    Raises:
        AssertionError: If the joint is not found in the model.
    """
    joint_id: int = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id != -1, f"Joint with name '{name}' is not part of the model."
    joint_type: int = int(model.jnt_type[joint_id])
    joint_addr: int = int(model.jnt_qposadr[joint_id])

    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        ndim = 7
    elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        ndim = 4
    else:
        assert joint_type in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    arr_val = np.asarray(value)
    if ndim > 1:
        assert arr_val.shape == (end_idx - start_idx,)
    data.qpos[start_idx:end_idx] = arr_val


def apply_patch() -> None:
    """Apply monkey patches to gymnasium_robotics mujoco_utils if not already applied."""
    global _PATCHED
    if _PATCHED:
        return
    mujoco_utils.get_joint_qpos = _safe_get_joint_qpos
    mujoco_utils.get_joint_qvel = _safe_get_joint_qvel
    mujoco_utils.set_joint_qpos = _safe_set_joint_qpos
    _PATCHED = True
