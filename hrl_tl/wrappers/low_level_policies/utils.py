import numpy as np
from numpy.typing import NDArray

from hrl_tl.wrappers.gc_ltl import GoalRep


def parse_tl_spec(
    tl_spec: str, predicates: list[str]
) -> tuple[list[str], list[str]]:
    """
    Parse a temporal logic specification string to extract goals and constraints.

    Parameters
    ----------
    tl_spec : str
        The temporal logic specification string.
    predicates : list[str]
        The list of valid predicates.

    Returns
    -------
    goals : list[str]
        The list of goal predicates extracted from the TL spec.
    constraints : list[str]
        The list of constraint predicates extracted from the TL spec.
    """
    goals: list[str] = []
    constraints: list[str] = []
    if "F" in tl_spec:
        f_part = tl_spec.split("F")[1].split(" & ")[0].split(" G")[0]
        if "|" in f_part:
            goals = [
                g.strip()
                for g in f_part.replace("(", "").replace(")", "").split("|")
            ]
        elif "&" in f_part:
            goals = [
                g.strip()
                for g in f_part.replace("(", "").replace(")", "").split("&")
            ]
        else:
            goals = [f_part.strip()]
    if "G" in tl_spec:
        g_part = tl_spec.split("G")[1].split(" & ")[0].split(" F")[0]
        if "|" in g_part:
            constraints = [
                c.strip().replace("!", "")
                for c in g_part.replace("(", "").replace(")", "").split("|")
            ]
        elif "&" in g_part:
            constraints = [
                c.strip().replace("!", "")
                for c in g_part.replace("(", "").replace(")", "").split("&")
            ]
        else:
            constraints = [g_part.strip().replace("!", "")]

    for goal in goals:
        assert goal in predicates, (
            f"Goal {goal} not in predicates {predicates} for TL spec {tl_spec}"
        )
    for constraint in constraints:
        assert constraint in predicates, (
            f"Constraint {constraint} not in predicates {predicates} for TL spec {tl_spec}"
        )

    goals: list[str] = sorted(list(set(goals)))
    constraints: list[str] = sorted(list(set(constraints)))

    return goals, constraints


def create_goal_conditioning_obs(
    obs: np.ndarray | dict[str, np.ndarray],
    goal_pred: str | int,
    goal_rep: GoalRep,
) -> dict[str, np.ndarray]:
    """Create a goal-conditioned observation by adding goal predicates."""
    goal_obs: dict[str, np.ndarray]

    if isinstance(goal_pred, str):
        predicates: list[str] = goal_rep.pred_names
        goal_pred_index: int = predicates.index(goal_pred)
    else:
        goal_pred_index = goal_pred

    goal_encoding = goal_rep.pred2rep(goal_pred_index)

    if isinstance(obs, dict):
        goal_obs = obs.copy()
        # Assert "aut_state" in obs
        assert "aut_state" in obs, f"'aut_state' not in observation {obs}"
        len_aut_state: int = np.array(obs["aut_state"]).size
        num_repeats: int = (
            (len_aut_state // goal_encoding.size)
            if len_aut_state > goal_encoding.size
            else 1
        )
        # Repeat goal_pred_index to match the size of aut_state
        if num_repeats > 1:
            goal_obs["goal_pred"] = np.repeat(goal_encoding, num_repeats)
        else:
            goal_obs["goal_pred"] = goal_encoding
        # Remove "aut_state" from goal_obs
        goal_obs.pop("aut_state")
    else:
        goal_obs = {
            "obs": obs.copy(),
            "goal_pred": goal_encoding,
        }

    return goal_obs
