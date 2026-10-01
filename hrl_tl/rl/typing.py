from typing import Protocol

import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs


class Actor(Protocol):
    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        Parameters
        ----------
        obs: PyTorchObs
            The observation tensor for which to compute the action distribution parameters.

        Returns
        -------
        mean_actions: th.Tensor
            The mean actions of the action distribution.
        log_std: th.Tensor
            The log standard deviation of the action distribution.
        extra_params: dict[str, th.Tensor]
            Any extra parameters needed for the action distribution (e.g., latent_sde).
        """
        ...
