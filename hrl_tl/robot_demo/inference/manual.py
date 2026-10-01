"""One calibrated movement through the same controller as policy actions."""

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.control import action
from hrl_tl.robot_demo.inference import policy


class SingleActionProvider(policy.ActionProvider):
    def __init__(
        self, movement: action.MovementAction, *, angle_deg: float, distance_m: float
    ) -> None:
        self._movement = movement
        self._issued = False
        self._decision = {
            "source": "manual_single_action",
            "zone_direction_deg": angle_deg,
            "distance_m": distance_m,
        }

    def next_action(self, measured: pose.Pose2D) -> action.MovementAction | None:
        if self._issued:
            return None
        self._issued = True
        return self._movement

    def feedback(self) -> policy.Feedback:
        return policy.Feedback(decision=self._decision)
