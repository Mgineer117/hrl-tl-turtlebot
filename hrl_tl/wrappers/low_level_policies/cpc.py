from typing import Generic, Literal

import numpy as np
import torch
from gym_tl_tools import BaseVarValueInfoGenerator, Parser
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from rl_pipeline.core import ConfigReader
from sb3_soft import SDSAC
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch import Tensor

from hrl_tl.rl.typing import Actor
from hrl_tl.utils.io import get_class
from hrl_tl.wrappers.gc_ltl import (
    GoalRep,
    IndexGoalRep,
    OneHotGoalRep,
    Predicate,
)

from .base import LowLevelPolicy, TLObs
from .utils import create_goal_conditioning_obs, parse_tl_spec


class ProhibitedActionGenerator:
    """Generate prohibited directional actions from wall distances.

    Expected wall distance observation format:
    ``observation["wall_dist"] == [top_dist, right_dist, bottom_dist, left_dist]``.
    """

    def __init__(
        self,
        num_direction_actions: int,
        direction_action_dim: int = 0,
        wall_threshold: float = 0.2,
        clockwise: bool = False,
    ) -> None:
        self.num_direction_actions = num_direction_actions
        self.direction_action_dim = direction_action_dim
        self.wall_threshold = wall_threshold
        self.clockwise = clockwise

        # Angular convention follows renderer:
        # angle = (direction_idx / num_directions) * 2*pi
        # idx=0 -> right, idx=n/4 -> top, idx=n/2 -> left, idx=3n/4 -> bottom
        self.direction_angles: dict[str, float] = {
            "right": 0.0,
            "top": np.pi / 2,
            "left": np.pi,
            "bottom": 3 * np.pi / 2,
        }

    def _angle_to_index(self, angle: float) -> int:
        raw = (
            (-angle if self.clockwise else angle)
            / (2 * np.pi)
            * self.num_direction_actions
        )
        return int(round(raw)) % self.num_direction_actions

    def get_prohibited_direction_indices(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
    ) -> set[int]:
        if not isinstance(observation, dict):
            return set()
        if "wall_dist" not in observation:
            return set()

        wall_dist_raw = np.asarray(observation["wall_dist"]).reshape(-1)
        if wall_dist_raw.size < 4:
            return set()

        top_dist, right_dist, bottom_dist, left_dist = wall_dist_raw[:4]

        prohibited: set[int] = set()
        if float(top_dist) < self.wall_threshold:
            prohibited.add(self._angle_to_index(self.direction_angles["top"]))
        if float(right_dist) < self.wall_threshold:
            prohibited.add(self._angle_to_index(self.direction_angles["right"]))
        if float(bottom_dist) < self.wall_threshold:
            prohibited.add(
                self._angle_to_index(self.direction_angles["bottom"])
            )
        if float(left_dist) < self.wall_threshold:
            prohibited.add(self._angle_to_index(self.direction_angles["left"]))

        return prohibited


class LambdaConfig(BaseModel):
    """
    Configuration for the sigmoid weight for the constraints in CPC policy.
    """

    L_gain: float = 3.0
    k_steepness: float = 2.0
    eps_margin: float = 0.5
    max_constraint_std_ratio: float = 3.0

    """Maximum ratio of constraint std to goal std before clamping.

    When a constraint policy is much more uncertain than the goal policy
    (large std ratio), its precision becomes negligible in the CPC
    composition, making it unable to push the action away from the
    obstacle.  Clamping ensures the constraint always contributes
    meaningfully.
    """
    enable_min_angular_deflection: bool = True
    """Enforce minimum angular deflection from each constraint direction.

    When the goal and constraint policies point in nearly the same
    direction, vector subtraction in CPC cannot produce meaningful
    angular deflection — it only reduces the resultant magnitude.
    This option enforces a minimum angular separation of
    arctan(lambda * kappa_c / kappa_g) from each constraint direction,
    ensuring the agent steers around obstacles even when they lie
    directly on the path to the goal.
    """


def lambda_weight(
    robustness: NDArray[np.floating],
    L_gain: float,
    k_steepness: float,
    eps_margin: float,
) -> NDArray[np.floating]:
    """
    Compute the sigmoid weight for the constraints in CPC policy.

    Parameters
    ----------
    robustness : NDArray[np.float64]
        The robustness value of the constraint of psi_o (not !psi_o).
    L_gain : float
        The gain of the sigmoid function.
    k_stepness : float
        The steepness of the sigmoid function.
    eps_margin : float
        The margin for the sigmoid function.

    Returns
    -------
    weight : float
        The computed weight for the constraint.
    """

    weight: NDArray[np.float64] = L_gain / (
        1.0 + np.exp(-k_steepness * (robustness + eps_margin))
    )
    return weight


class CPCCompositePolicyConfig(BaseModel):
    tl_spec: str = ""
    predicates: list[Predicate] = []
    model: BaseAlgorithm
    var_value_info_generator: BaseVarValueInfoGenerator
    lambda_config: LambdaConfig = LambdaConfig()
    normalize_lambdas: bool = False
    parser: Parser = Parser()
    goal_rep: Literal["index", "one_hot"] = "one_hot"
    prohibited_action_generator: ProhibitedActionGenerator | None = None
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CPCCompositePolicyConfigReader(
    BaseModel, ConfigReader[CPCCompositePolicyConfig]
):
    tl_spec: str = ""
    predicates: list[Predicate] = []
    algo: str = "PPO"
    model_path: str
    var_value_info_generator_cls: str
    lambda_config: LambdaConfig = LambdaConfig()
    normalize_lambdas: bool = False
    device: str = "cuda:0"
    goal_rep: Literal["index", "one_hot"] = "one_hot"
    verbose: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_config(self) -> CPCCompositePolicyConfig:
        if self.algo == "PPO":
            model = PPO.load(self.model_path, device=self.device)
        elif self.algo == "SDSAC":
            model = SDSAC.load(self.model_path, device=self.device)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algo}")

        var_value_info_generator: BaseVarValueInfoGenerator
        gen_cls = get_class(self.var_value_info_generator_cls)
        assert gen_cls is not None, (
            f"Could not find class {self.var_value_info_generator_cls}"
        )
        var_value_info_generator = gen_cls()

        return CPCCompositePolicyConfig(
            tl_spec=self.tl_spec,
            predicates=self.predicates,
            model=model,
            var_value_info_generator=var_value_info_generator,
            lambda_config=self.lambda_config,
            normalize_lambdas=self.normalize_lambdas,
            goal_rep=self.goal_rep,
            verbose=self.verbose,
        )


class CPCCompositePolicy(Generic[ObsType]):
    """
    A composite low-level policy that combines multiple goal-reaching policies using Contrasitive Policy Composition


    """

    def __init__(
        self,
        tl_spec: str,
        predicates: list[Predicate],
        model: OnPolicyAlgorithm | OffPolicyAlgorithm,
        var_value_info_generator: BaseVarValueInfoGenerator,
        lambda_config: LambdaConfig = LambdaConfig(),
        normalize_lambdas: bool = False,
        parser: Parser = Parser(),
        for_eval: bool = False,
        goal_rep: Literal["index", "one_hot"] = "index",
        action_space_type: Literal["discrete", "continuous"] = "discrete",
        prohibited_action_generator: ProhibitedActionGenerator | None = None,
        verbose: bool = False,
    ) -> None:
        self.atomic_predicates: list[Predicate] = predicates

        self.goal_rep: GoalRep
        match goal_rep:
            case "index":
                self.goal_rep = IndexGoalRep(predicates)
            case "one_hot":
                self.goal_rep = OneHotGoalRep(predicates)
            case _:
                raise ValueError(
                    f"Invalid goal_rep: {goal_rep}. Must be 'index' or 'one_hot'."
                )
        self.predicates: list[str] = self.goal_rep.pred_names

        self.tl_spec: str = tl_spec

        self.for_eval: bool = for_eval
        self.goals: list[str]
        self.constraints: list[str]
        self.goals, self.constraints = parse_tl_spec(tl_spec, self.predicates)

        self.parser: Parser = parser
        self.model: OnPolicyAlgorithm | OffPolicyAlgorithm = model

        self.var_value_info_generator: BaseVarValueInfoGenerator = (
            var_value_info_generator
        )
        self.lambda_config: LambdaConfig = lambda_config
        self.verbose: bool = verbose
        self.normalize_lambdas: bool = normalize_lambdas

        self.constraint_idx_to_pred_idx: list[int] = [
            self.predicates.index(constraint) for constraint in self.constraints
        ]

        self.action_space_type: Literal["discrete", "continuous"] = (
            action_space_type
        )

        self.prohibited_action_generator: ProhibitedActionGenerator | None = (
            prohibited_action_generator
        )
        if self.action_space_type == "discrete":
            if isinstance(self.model.action_space, spaces.Discrete):
                self.prohibited_action_generator = ProhibitedActionGenerator(
                    num_direction_actions=int(self.model.action_space.n),
                    direction_action_dim=0,
                )
            elif isinstance(self.model.action_space, spaces.MultiDiscrete):
                self.prohibited_action_generator = ProhibitedActionGenerator(
                    num_direction_actions=int(self.model.action_space.nvec[0]),
                    direction_action_dim=0,
                )

        self.last_goal_prob: NDArray[np.floating] | None = None
        self.last_constraint_prob: NDArray[np.floating] | None = None
        self.last_joint_prob: NDArray[np.floating] | None = None
        self.action_combinations: NDArray[np.integer] | None = None
        self.last_goal_mean: NDArray[np.floating] | None = None
        self.last_goal_std: NDArray[np.floating] | None = None
        self.last_constraint_mean: NDArray[np.floating] | None = None
        self.last_constraint_std: NDArray[np.floating] | None = None
        self.last_joint_mean: NDArray[np.floating] | None = None
        self.last_joint_std: NDArray[np.floating] | None = None

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Predict action using Contrastive Policy Composition.

        Handles batched observations and dispatches to discrete or continuous
        CPC based on ``action_space_type``.
        """
        if (
            episode_start is not None
            and isinstance(episode_start, np.ndarray)
            and episode_start.ndim >= 1
        ):
            return self._predict_batched(
                observation, state, episode_start.shape[0], deterministic
            )
        return self._predict_single(observation, deterministic)

    # ------------------------------------------------------------------ #
    #  Batch handling                                                      #
    # ------------------------------------------------------------------ #

    def _predict_batched(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None,
        batch_size: int,
        deterministic: bool,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Process batched observations by predicting each element individually."""
        actions = []
        for i in range(batch_size):
            if isinstance(observation, dict):
                single_obs = {
                    key: value[i] if isinstance(value, np.ndarray) else value
                    for key, value in observation.items()
                }
            else:
                single_obs = observation[i]
            action, _ = self.predict(single_obs, state, None, deterministic)
            actions.append(action)
        return np.array(actions), state

    # ------------------------------------------------------------------ #
    #  Observation / lambda preparation                                    #
    # ------------------------------------------------------------------ #

    def _prepare_observations(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
    ) -> tuple[
        list[np.ndarray | dict[str, np.ndarray]],
        list[np.ndarray | dict[str, np.ndarray]],
        list[Tensor | dict[str, Tensor]],
        list[Tensor | dict[str, Tensor]],
    ]:
        """Create goal-conditioned observations as numpy arrays and tensors."""
        goal_obs_np: list[np.ndarray | dict[str, np.ndarray]] = [
            create_goal_conditioning_obs(observation, goal, self.goal_rep)
            for goal in self.goals
        ]
        constraint_obs_np: list[np.ndarray | dict[str, np.ndarray]] = [
            create_goal_conditioning_obs(observation, constraint, self.goal_rep)
            for constraint in self.constraints
        ]
        goal_obs: list[Tensor | dict[str, Tensor]] = [
            self.model.policy.obs_to_tensor(obs)[0] for obs in goal_obs_np
        ]
        constraint_obs: list[Tensor | dict[str, Tensor]] = [
            self.model.policy.obs_to_tensor(obs)[0] for obs in constraint_obs_np
        ]
        return goal_obs_np, constraint_obs_np, goal_obs, constraint_obs

    def _prepare_observations_np(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
    ) -> tuple[
        list[np.ndarray | dict[str, np.ndarray]],
        list[np.ndarray | dict[str, np.ndarray]],
    ]:
        """Create goal-conditioned observations as numpy arrays only.

        Used by the discrete path which handles batched tensor conversion
        internally for better performance.
        """
        goal_obs_np: list[np.ndarray | dict[str, np.ndarray]] = [
            create_goal_conditioning_obs(observation, goal, self.goal_rep)
            for goal in self.goals
        ]
        constraint_obs_np: list[np.ndarray | dict[str, np.ndarray]] = [
            create_goal_conditioning_obs(observation, constraint, self.goal_rep)
            for constraint in self.constraints
        ]
        return goal_obs_np, constraint_obs_np

    @staticmethod
    def _batch_dict_obs(
        obs_list: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Stack a list of single-observation dicts into one batched dict."""
        return {
            key: np.stack([obs[key] for obs in obs_list]) for key in obs_list[0]
        }

    def _compute_lambdas(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
    ) -> NDArray[np.floating]:
        """Compute robustness-based sigmoid weights for constraint policies."""
        var_values: dict[str, float] = (
            self.var_value_info_generator.get_var_values(
                None,  # type: ignore[arg-type]
                observation,
                {},
            )
        )
        ap_rob_dict: dict[str, float] = {
            atom_pred.name: self.parser.tl2rob(atom_pred.formula, var_values)
            for atom_pred in self.atomic_predicates
        }
        rob_values: NDArray[np.floating] = np.array(
            [[ap_rob_dict[constraint] for constraint in self.constraints]]
        )
        return lambda_weight(
            robustness=rob_values,
            L_gain=self.lambda_config.L_gain,
            k_steepness=self.lambda_config.k_steepness,
            eps_margin=self.lambda_config.eps_margin,
        )

    # ------------------------------------------------------------------ #
    #  Single-observation dispatcher                                       #
    # ------------------------------------------------------------------ #

    def _predict_single(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        deterministic: bool,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Predict action for a single (non-batched) observation."""
        with torch.no_grad():
            lambdas = self._compute_lambdas(observation)

            match (self.action_space_type, self.model):
                case ("discrete", OnPolicyAlgorithm()) | (
                    "discrete",
                    OffPolicyAlgorithm(),
                ):
                    # Discrete path uses batched obs preparation for
                    # better performance (single obs_to_tensor +
                    # single get_distribution forward pass).
                    goal_obs_np, constraint_obs_np = (
                        self._prepare_observations_np(observation)
                    )
                    return self._predict_discrete(
                        observation,
                        goal_obs_np,
                        constraint_obs_np,
                        lambdas,
                        deterministic,
                    )
                case ("continuous", OffPolicyAlgorithm()):
                    goal_obs_np, constraint_obs_np, goal_obs, constraint_obs = (
                        self._prepare_observations(observation)
                    )
                    return self._predict_continuous(
                        goal_obs_np,
                        goal_obs,
                        constraint_obs,
                        lambdas,
                        deterministic,
                    )

        raise ValueError(
            f"Unsupported action_space_type/model combination: "
            f"{self.action_space_type}, {type(self.model)}"
        )

    # ------------------------------------------------------------------ #
    #  Discrete action space CPC                                           #
    # ------------------------------------------------------------------ #

    def _extract_discrete_log_probs(
        self,
        observations: list[Tensor | dict[str, Tensor]],
    ) -> list[list[Tensor]]:
        """Extract per-dimension log-probabilities from on-policy discrete distributions.

        Returns a list (per observation) of lists (per action dimension) of
        log-prob tensors.
        """
        all_log_probs: list[list[Tensor]] = []
        for obs in observations:
            dist: CategoricalDistribution | MultiCategoricalDistribution = (
                self.model.policy.get_distribution(obs)
            )
            if isinstance(dist, MultiCategoricalDistribution):
                all_log_probs.append(
                    [
                        torch.log_softmax(d.logits, dim=-1)
                        for d in dist.distribution
                    ]
                )
            else:
                all_log_probs.append(
                    [torch.log_softmax(dist.distribution.logits, dim=-1)]
                )
        return all_log_probs

    def _extract_discrete_log_probs_batched(
        self,
        observations_np: list[np.ndarray | dict[str, np.ndarray]],
    ) -> list[list[Tensor]]:
        """Extract per-dimension log-probabilities via a single batched forward pass.

        All observations are stacked into one batch, converted to tensors once,
        and forwarded through the network in a single call. The resulting logits
        are then sliced back into per-observation log-prob lists.

        Returns the same format as ``_extract_discrete_log_probs``:
        a list (per observation) of lists (per action dimension) of
        log-prob tensors with shape ``[1, n_actions_i]``.
        """
        if not observations_np:
            return []

        n_obs = len(observations_np)

        # Batch all observations into a single dict / array
        if isinstance(observations_np[0], dict):
            batched_obs_np = self._batch_dict_obs(observations_np)  # type: ignore[arg-type]
        else:
            batched_obs_np = np.stack(observations_np)  # type: ignore[arg-type]

        batched_tensor, _ = self.model.policy.obs_to_tensor(batched_obs_np)
        dist: CategoricalDistribution | MultiCategoricalDistribution = (
            self.model.policy.get_distribution(batched_tensor)
        )

        # Split batched logits back into per-observation log-probs
        all_log_probs: list[list[Tensor]] = []
        if isinstance(dist, MultiCategoricalDistribution):
            # dist.distribution is a list of Categorical, each with
            # logits shape [n_obs, n_actions_i]
            per_dim_log_probs: list[Tensor] = [
                torch.log_softmax(d.logits, dim=-1) for d in dist.distribution
            ]
            for obs_idx in range(n_obs):
                all_log_probs.append(
                    [lp[obs_idx : obs_idx + 1] for lp in per_dim_log_probs]
                )
        else:
            # Single Categorical with logits shape [n_obs, n_actions]
            log_probs = torch.log_softmax(dist.distribution.logits, dim=-1)
            for obs_idx in range(n_obs):
                all_log_probs.append([log_probs[obs_idx : obs_idx + 1]])

        return all_log_probs

    def _compose_discrete_single_dim(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        goal_log_probs: list[list[Tensor]],
        constraint_log_probs: list[list[Tensor]],
        lambdas: NDArray[np.floating],
        deterministic: bool,
    ) -> Tensor:
        """CPC composition for single-dimension discrete action space."""
        self.action_combinations = np.arange(
            goal_log_probs[0][0].shape[-1]
        ).reshape(-1, 1)

        goal_log_probs_tensor: Tensor = torch.stack(
            [lp[0] for lp in goal_log_probs], dim=0
        ).squeeze(1)
        goal_log_probs_sum: Tensor = torch.sum(goal_log_probs_tensor, dim=0)

        has_constraints = len(constraint_log_probs) > 0

        if has_constraints:
            constraint_log_probs_tensor: Tensor = (
                torch.stack([lp[0] for lp in constraint_log_probs], dim=0)
                .squeeze(1)
                .to(goal_log_probs_tensor.device)
            )
            lambda_weights: Tensor = torch.tensor(
                lambdas[0, :, None],
                dtype=constraint_log_probs_tensor.dtype,
                device=constraint_log_probs_tensor.device,
            )

            # Normalize lambda weights if normalize_lambdas is True
            if self.normalize_lambdas:
                # Normalize: lambda_weights -> lambda / (1 + lambda)
                normalized_lambda = lambda_weights / (1.0 + lambda_weights)
                # Scale goal log-probs by 1 / (1 + lambda) element-wise
                goal_scaling = 1.0 / (1.0 + lambda_weights.squeeze(-1))
                goal_log_probs_sum = goal_log_probs_sum * goal_scaling
                lambda_weights = normalized_lambda

            weighted_constraint_log_probs: Tensor = torch.sum(
                constraint_log_probs_tensor * lambda_weights,
                dim=0,
            )
            combined_log_probs: Tensor = (
                goal_log_probs_sum - weighted_constraint_log_probs
            )
            self.last_constraint_prob = (
                torch.exp(
                    torch.log_softmax(weighted_constraint_log_probs, dim=-1)
                )
                .cpu()
                .numpy()
            )
        else:
            combined_log_probs = goal_log_probs_sum
            self.last_constraint_prob = None

        self.last_goal_prob = (
            torch.exp(torch.log_softmax(goal_log_probs_sum, dim=-1))
            .cpu()
            .numpy()
        )

        joint_probs: Tensor = torch.exp(
            torch.log_softmax(combined_log_probs, dim=-1)
        )
        joint_probs = self._apply_prohibited_action_mask(
            observation=observation,
            joint_probs=joint_probs,
        )
        self.last_joint_prob = joint_probs.cpu().numpy()

        if deterministic:
            return torch.argmax(joint_probs, dim=-1)
        else:
            return torch.distributions.Categorical(probs=joint_probs).sample()

    def _compute_per_policy_joint_log_probs(
        self,
        all_log_probs: list[list[Tensor]],
        action_combinations: Tensor,
        n_action_dims: int,
    ) -> Tensor:
        """Compute joint log-prob for each policy across all action combinations.

        Returns tensor of shape ``[n_policies, total_combinations]``.
        """
        total_combinations = action_combinations.shape[0]
        device = all_log_probs[0][0].device
        dtype = all_log_probs[0][0].dtype

        joint_log_probs = torch.zeros(
            len(all_log_probs),
            total_combinations,
            device=device,
            dtype=dtype,
        )

        for policy_idx, policy_lp in enumerate(all_log_probs):
            for dim_idx in range(n_action_dims):
                action_dim_values = action_combinations[:, dim_idx]
                lp_for_dim = policy_lp[dim_idx].squeeze()
                joint_log_probs[policy_idx] += lp_for_dim[action_dim_values]

        return joint_log_probs

    def _compose_discrete_multi_dim(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        goal_log_probs: list[list[Tensor]],
        constraint_log_probs: list[list[Tensor]],
        lambdas: NDArray[np.floating],
        deterministic: bool,
    ) -> Tensor:
        """CPC composition for multi-dimension discrete action space."""
        n_action_dims = len(goal_log_probs[0])
        device = goal_log_probs[0][0].device
        dtype = goal_log_probs[0][0].dtype

        n_actions_per_dim = [
            goal_log_probs[0][dim_idx].shape[-1]
            for dim_idx in range(n_action_dims)
        ]

        # Create meshgrid for all action combinations
        action_indices = torch.meshgrid(
            *[torch.arange(n, device=device) for n in n_actions_per_dim],
            indexing="ij",
        )
        action_combinations = torch.stack(
            [idx.flatten() for idx in action_indices], dim=-1
        )
        self.action_combinations = action_combinations.cpu().numpy()

        # Compute per-policy joint log-probs
        goal_joint_log_probs = self._compute_per_policy_joint_log_probs(
            goal_log_probs, action_combinations, n_action_dims
        )
        composed_goal_joint = torch.sum(goal_joint_log_probs, dim=0)

        has_constraints = len(constraint_log_probs) > 0

        if has_constraints:
            constraint_joint_log_probs = (
                self._compute_per_policy_joint_log_probs(
                    constraint_log_probs, action_combinations, n_action_dims
                )
            )
            lambda_weights_tensor = torch.tensor(
                lambdas[0, :],
                dtype=dtype,
                device=device,
            )

            # Normalize lambda weights if normalize_lambdas is True
            if self.normalize_lambdas:
                # Normalize: lambda_weights -> lambda / (1 + lambda)
                normalized_lambda = lambda_weights_tensor / (
                    1.0 + lambda_weights_tensor
                )
                # Scale goal log-probs by 1 / (1 + lambda) element-wise
                goal_scaling = 1.0 / (1.0 + lambda_weights_tensor)
                composed_goal_joint = composed_goal_joint * goal_scaling[None]
                lambda_weights_tensor = normalized_lambda

            composed_constraint_joint = torch.sum(
                constraint_joint_log_probs * lambda_weights_tensor[:, None],
                dim=0,
            )
            joint_log_probs = composed_goal_joint - composed_constraint_joint
            self.last_constraint_prob = (
                torch.exp(composed_constraint_joint).cpu().numpy()
            )
        else:
            joint_log_probs = composed_goal_joint
            self.last_constraint_prob = None

        self.last_goal_prob = torch.exp(composed_goal_joint).cpu().numpy()

        joint_probs = torch.exp(torch.log_softmax(joint_log_probs, dim=-1))
        joint_probs = self._apply_prohibited_action_mask(
            observation=observation,
            joint_probs=joint_probs,
            action_combinations=action_combinations,
        )
        self.last_joint_prob = joint_probs.cpu().numpy()

        if deterministic:
            best_idx = torch.argmax(joint_probs)
            return action_combinations[best_idx]
        else:
            sampled_idx = torch.distributions.Categorical(
                probs=joint_probs
            ).sample()
            return action_combinations[sampled_idx]

    def _apply_prohibited_action_mask(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        joint_probs: Tensor,
        action_combinations: Tensor | None = None,
    ) -> Tensor:
        """Mask prohibited actions (based on wall distances) and renormalize."""
        if self.prohibited_action_generator is None:
            return joint_probs

        prohibited_indices = (
            self.prohibited_action_generator.get_prohibited_direction_indices(
                observation
            )
        )
        if not prohibited_indices:
            return joint_probs

        masked_probs = joint_probs.clone()
        if action_combinations is None:
            # Single discrete action: action index is direction index
            prohibited_tensor = torch.tensor(
                sorted(prohibited_indices),
                device=joint_probs.device,
                dtype=torch.long,
            )
            masked_probs[prohibited_tensor] = 0.0
        else:
            # Multi-discrete: mask combinations where directional dimension is prohibited
            direction_dim = (
                self.prohibited_action_generator.direction_action_dim
            )
            prohibited_tensor = torch.tensor(
                sorted(prohibited_indices),
                device=action_combinations.device,
                dtype=action_combinations.dtype,
            )
            direction_values = action_combinations[:, direction_dim]
            prohibit_mask = (
                direction_values[:, None] == prohibited_tensor[None, :]
            ).any(dim=1)
            masked_probs[prohibit_mask] = 0.0

        total_prob = torch.sum(masked_probs)
        if total_prob <= 0:
            # Fallback to original distribution when all actions are masked
            return joint_probs

        return masked_probs / total_prob

    def _predict_discrete(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        goal_observations_np: list[np.ndarray | dict[str, np.ndarray]],
        constraint_observations_np: list[np.ndarray | dict[str, np.ndarray]],
        lambdas: NDArray[np.floating],
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        """Predict using CPC in discrete action space (log-space composition).

        All goal and constraint observations are combined into a single batch
        and processed with one ``obs_to_tensor`` + ``get_distribution`` call.

        log pi_tilde(a|s,phi) = sum_i log pi(a|s,psi_g,i) - sum_j lambda_j(s) log pi(a|s,psi_o,j)
        """
        n_goals = len(goal_observations_np)

        # Combine all observations into a single batch for one forward pass
        all_obs_np = goal_observations_np + constraint_observations_np
        all_log_probs = self._extract_discrete_log_probs_batched(all_obs_np)

        # Split back into goal / constraint groups
        goal_log_probs = all_log_probs[:n_goals]
        constraint_log_probs = all_log_probs[n_goals:]

        n_action_dims = len(goal_log_probs[0])

        if n_action_dims == 1:
            action = self._compose_discrete_single_dim(
                observation,
                goal_log_probs,
                constraint_log_probs,
                lambdas,
                deterministic,
            )
        else:
            action = self._compose_discrete_multi_dim(
                observation,
                goal_log_probs,
                constraint_log_probs,
                lambdas,
                deterministic,
            )

        if self.verbose:
            self._log_discrete_debug(
                lambdas, n_action_dims, constraint_log_probs, action
            )

        return (
            np.array([action.cpu().numpy()])
            if self.for_eval
            else action.cpu().numpy(),
            None,
        )

    def _log_discrete_debug(
        self,
        lambdas: NDArray[np.floating],
        n_action_dims: int,
        constraint_log_probs: list[list[Tensor]],
        action: Tensor,
    ) -> None:
        """Print debug information for discrete CPC prediction."""
        has_constraints = len(constraint_log_probs) > 0
        print(f"Lambdas: {lambdas}")

        if n_action_dims == 1:
            print(f"Goal probs: {self.last_goal_prob}")
            if has_constraints:
                print(f"Weighted constraint probs: {self.last_constraint_prob}")
            print(f"Combined probs: {self.last_joint_prob}")
        else:
            assert self.last_joint_prob is not None
            assert self.action_combinations is not None
            joint_probs_t = torch.tensor(self.last_joint_prob)
            print(f"Joint log-probs shape: {joint_probs_t.shape}")
            print("Top 5 action combinations and their probs:")
            top_k = min(5, joint_probs_t.shape[0])
            top_probs, top_indices = torch.topk(joint_probs_t, top_k)
            for i in range(top_k):
                combo = self.action_combinations[top_indices[i]]
                print(f"  {combo}: {top_probs[i].item():.4f}")

        print(f"Action: {action.cpu().numpy()}")

    # ------------------------------------------------------------------ #
    #  Continuous action space CPC                                         #
    # ------------------------------------------------------------------ #

    def _is_single_dim_angle_action_space(self) -> bool:
        """Check whether action space is a 1D angle in [0, 2*pi]."""
        if not isinstance(self.model.action_space, spaces.Box):
            return False
        if self.model.action_space.shape != (1,):
            return False

        low = float(np.asarray(self.model.action_space.low).reshape(-1)[0])
        high = float(np.asarray(self.model.action_space.high).reshape(-1)[0])
        return (abs(low - 0.0) <= 1e-6) and (abs(high - 2 * np.pi) <= 1e-5)

    def _compose_angular_distributions(
        self,
        means: NDArray[np.floating],
        stds: NDArray[np.floating],
        weights: NDArray[np.floating],
    ) -> tuple[float, float]:
        """Compose angle distributions using circular concentration vectors.

        Each component contributes ``w * kappa * [cos(mu), sin(mu)]`` where
        ``kappa ~= 1/sigma^2``.
        """
        safe_stds = np.clip(stds.astype(np.float64), 1e-6, None)
        kappas = 1.0 / np.square(safe_stds)
        weighted = weights.astype(np.float64) * kappas

        x = float(np.sum(weighted * np.cos(means)))
        y = float(np.sum(weighted * np.sin(means)))

        norm = float(np.hypot(x, y))
        if norm < 1e-12:
            mu = float(np.mod(means[0], 2 * np.pi))
            sigma = float(np.mean(safe_stds))
            return mu, sigma

        mu = float(np.mod(np.arctan2(y, x), 2 * np.pi))
        sigma = float(np.sqrt(1.0 / max(norm, 1e-6)))
        return mu, sigma

    def _fuse_gaussians(
        self,
        goal_dist_params: list[tuple[Tensor, Tensor, dict[str, Tensor]]],
        constraint_dist_params: list[tuple[Tensor, Tensor, dict[str, Tensor]]],
        lambdas: NDArray[np.floating],
    ) -> tuple[Tensor, Tensor]:
        """Fuse Gaussian distributions via CPC precision arithmetic.

        Applies Gaussian product for goals and Gaussian division for constraints::

            1/sigma_new^2 = sum_i 1/sigma_g,i^2  -  sum_j lambda_j / sigma_o,j^2
            mu_new = sigma_new^2 * (sum_i mu_g,i/sigma_g,i^2  -  sum_j lambda_j * mu_o,j / sigma_o,j^2)

        Returns ``(new_mean, new_std)`` for the composed distribution.
        """
        goal_means: list[Tensor] = [p[0] for p in goal_dist_params]
        goal_vars: list[Tensor] = [
            torch.exp(2 * p[1]) for p in goal_dist_params
        ]

        # Accumulate goal precisions: sum_i 1/sigma_g,i^2
        goal_precision_sum: Tensor = torch.zeros_like(goal_means[0])
        goal_weighted_mean_sum: Tensor = torch.zeros_like(goal_means[0])
        for mean, var in zip(goal_means, goal_vars):
            precision = 1.0 / var
            goal_precision_sum += precision
            goal_weighted_mean_sum += mean * precision

        if constraint_dist_params:
            constraint_means: list[Tensor] = [
                p[0] for p in constraint_dist_params
            ]
            constraint_vars: list[Tensor] = [
                torch.exp(2 * p[1]) for p in constraint_dist_params
            ]

            # Clamp constraint variance so its std <= ratio * goal std
            max_ratio = self.lambda_config.max_constraint_std_ratio
            goal_var_mean: Tensor = torch.stack(goal_vars).mean(dim=0)
            max_var = max_ratio**2 * goal_var_mean
            constraint_vars = [
                torch.clamp(v, max=max_var) for v in constraint_vars
            ]

            lambda_tensor = torch.tensor(
                lambdas[0, :],
                dtype=goal_means[0].dtype,
                device=goal_means[0].device,
            )

            constraint_precision_sum: Tensor = torch.zeros_like(goal_means[0])
            constraint_weighted_mean_sum: Tensor = torch.zeros_like(
                goal_means[0]
            )
            for j, (mean, var) in enumerate(
                zip(constraint_means, constraint_vars)
            ):
                precision = 1.0 / var
                constraint_precision_sum += lambda_tensor[j] * precision
                constraint_weighted_mean_sum += (
                    lambda_tensor[j] * mean * precision
                )

            new_precision = goal_precision_sum - constraint_precision_sum
            new_weighted_mean = (
                goal_weighted_mean_sum - constraint_weighted_mean_sum
            )
        else:
            new_precision = goal_precision_sum
            new_weighted_mean = goal_weighted_mean_sum

        # Clip precision to ensure valid distribution
        new_precision = torch.clamp(new_precision, min=1e-6)

        new_var: Tensor = 1.0 / new_precision
        new_mean: Tensor = new_var * new_weighted_mean
        new_std: Tensor = torch.sqrt(new_var)

        return new_mean, new_std

    def _aggregate_weighted_gaussians(
        self,
        dist_params: list[tuple[Tensor, Tensor, dict[str, Tensor]]],
        weights: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        """Aggregate diagonal Gaussians with positive precision weights.

        Returns ``(mean, std)`` of the weighted product distribution.
        """
        if not dist_params:
            return None

        means: list[Tensor] = [p[0] for p in dist_params]
        vars_: list[Tensor] = [torch.exp(2 * p[1]) for p in dist_params]

        precision_sum = torch.zeros_like(means[0])
        weighted_mean_sum = torch.zeros_like(means[0])

        for idx, (mean, var) in enumerate(zip(means, vars_)):
            precision = 1.0 / var
            w = weights[idx]
            precision_sum += w * precision
            weighted_mean_sum += w * mean * precision

        precision_sum = torch.clamp(precision_sum, min=1e-6)
        var_new = 1.0 / precision_sum
        mean_new = var_new * weighted_mean_sum
        std_new = torch.sqrt(var_new)
        return mean_new, std_new

    def _to_action_space_gaussian(
        self,
        mean: Tensor,
        std: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Approximate Gaussian parameters in env action space.

        For squashed policies, applies first-order uncertainty propagation through
        ``tanh`` and SB3's affine unscale mapping.
        """
        if not isinstance(self.model.action_space, spaces.Box):
            return mean, std

        squash_output = bool(getattr(self.model.policy, "squash_output", False))
        if not squash_output:
            return mean, std

        low = torch.as_tensor(
            self.model.action_space.low,
            dtype=mean.dtype,
            device=mean.device,
        )
        high = torch.as_tensor(
            self.model.action_space.high,
            dtype=mean.dtype,
            device=mean.device,
        )
        scale = (high - low) / 2.0

        squashed_mean = torch.tanh(mean)
        env_mean = low + (squashed_mean + 1.0) * scale

        tanh_jacobian = 1.0 - torch.tanh(mean).pow(2)
        env_std = torch.abs(scale * tanh_jacobian) * std
        env_std = torch.clamp(env_std, min=1e-6)
        return env_mean, env_std

    def _predict_continuous(
        self,
        goal_obs_np: list[np.ndarray | dict[str, np.ndarray]],
        goal_observations: list[Tensor | dict[str, Tensor]],
        constraint_observations: list[Tensor | dict[str, Tensor]],
        lambdas: NDArray[np.floating],
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        """Predict using CPC for squashed diagonal Gaussian policies.

        pi(a) ~ tanh(N(mu_new, sigma_new^2))
        """
        actor: Actor = self.model.actor  # type: ignore

        self.model.predict(goal_obs_np[0])

        goal_dist_params: list[tuple[Tensor, Tensor, dict[str, Tensor]]] = [
            actor.get_action_dist_params(obs) for obs in goal_observations
        ]
        constraint_dist_params: list[
            tuple[Tensor, Tensor, dict[str, Tensor]]
        ] = [
            actor.get_action_dist_params(obs) for obs in constraint_observations
        ]

        if self._is_single_dim_angle_action_space():
            goal_means: list[float] = []
            goal_stds: list[float] = []
            for mean_t, std_t, _ in goal_dist_params:
                env_mean_t, env_std_t = self._to_action_space_gaussian(
                    mean_t,
                    torch.exp(std_t),
                )
                goal_means.append(
                    float(env_mean_t.detach().cpu().numpy().reshape(-1)[0])
                )
                goal_stds.append(
                    float(env_std_t.detach().cpu().numpy().reshape(-1)[0])
                )

            constraint_means: list[float] = []
            constraint_stds: list[float] = []
            for mean_t, std_t, _ in constraint_dist_params:
                env_mean_t, env_std_t = self._to_action_space_gaussian(
                    mean_t,
                    torch.exp(std_t),
                )
                constraint_means.append(
                    float(env_mean_t.detach().cpu().numpy().reshape(-1)[0])
                )
                constraint_stds.append(
                    float(env_std_t.detach().cpu().numpy().reshape(-1)[0])
                )

            goal_means_arr = np.asarray(goal_means, dtype=np.float64)
            goal_stds_arr = np.asarray(goal_stds, dtype=np.float64)

            # Clamp constraint stds so concentration is comparable to goal
            max_ratio = self.lambda_config.max_constraint_std_ratio
            max_std = float(np.max(goal_stds_arr)) * max_ratio
            original_constraint_stds = list(constraint_stds)
            constraint_stds = [min(s, max_std) for s in constraint_stds]
            goal_weights = np.ones_like(goal_means_arr, dtype=np.float64)

            goal_mu, goal_sigma = self._compose_angular_distributions(
                goal_means_arr,
                goal_stds_arr,
                goal_weights,
            )
            self.last_goal_mean = np.array([[goal_mu]], dtype=np.float64)
            self.last_goal_std = np.array([[goal_sigma]], dtype=np.float64)

            _deflection_info: str | None = None
            if len(constraint_means) > 0:
                constraint_means_arr = np.asarray(
                    constraint_means, dtype=np.float64
                )
                constraint_stds_arr = np.asarray(
                    constraint_stds, dtype=np.float64
                )
                lambda_arr = np.asarray(lambdas[0], dtype=np.float64)
                constraint_mu, constraint_sigma = (
                    self._compose_angular_distributions(
                        constraint_means_arr,
                        constraint_stds_arr,
                        lambda_arr,
                    )
                )
                self.last_constraint_mean = np.array(
                    [[constraint_mu]], dtype=np.float64
                )
                self.last_constraint_std = np.array(
                    [[constraint_sigma]], dtype=np.float64
                )

                goal_kappa = goal_weights / np.square(
                    np.clip(goal_stds_arr, 1e-6, None)
                )
                constraint_kappa = lambda_arr / np.square(
                    np.clip(constraint_stds_arr, 1e-6, None)
                )
                gx = float(np.sum(goal_kappa * np.cos(goal_means_arr)))
                gy = float(np.sum(goal_kappa * np.sin(goal_means_arr)))
                cx = float(
                    np.sum(constraint_kappa * np.cos(constraint_means_arr))
                )
                cy = float(
                    np.sum(constraint_kappa * np.sin(constraint_means_arr))
                )
                joint_x = gx - cx
                joint_y = gy - cy
                joint_norm = float(np.hypot(joint_x, joint_y))
                if joint_norm < 1e-12:
                    joint_mu = goal_mu
                    joint_sigma = goal_sigma
                else:
                    joint_mu = float(
                        np.mod(np.arctan2(joint_y, joint_x), 2 * np.pi)
                    )
                    joint_sigma = float(np.sqrt(1.0 / max(joint_norm, 1e-6)))

                # --- Minimum angular deflection enforcement ---
                # When goal and constraint means are nearly aligned, vector
                # subtraction barely changes direction.  Enforce a minimum
                # angular separation of arctan(λ·κ_c / κ_g) from each
                # constraint direction so the agent steers around obstacles.
                if self.lambda_config.enable_min_angular_deflection:
                    g_kappa_total = float(np.sum(goal_kappa))
                    for i, c_mean in enumerate(constraint_means_arr):
                        c_kappa_i = float(constraint_kappa[i])
                        min_deflection = float(
                            np.arctan2(c_kappa_i, g_kappa_total)
                        )
                        # Signed angular difference: joint_mu - c_mean in [-π, π]
                        delta = float(
                            (joint_mu - c_mean + np.pi) % (2 * np.pi) - np.pi
                        )
                        if abs(delta) < min_deflection:
                            pre_deflection_mu = joint_mu
                            # Choose deflection direction: same side as goal
                            goal_delta = float(
                                (goal_mu - c_mean + np.pi) % (2 * np.pi) - np.pi
                            )
                            direction = (
                                np.sign(goal_delta)
                                if abs(goal_delta) > 1e-6
                                else 1.0
                            )
                            joint_mu = float(
                                np.mod(
                                    c_mean + direction * min_deflection,
                                    2 * np.pi,
                                )
                            )
                            _deflection_info = (
                                f"Min deflection applied: {np.degrees(pre_deflection_mu):.1f}° → "
                                f"{np.degrees(joint_mu):.1f}° "
                                f"(min_defl={np.degrees(min_deflection):.1f}°, "
                                f"|δ|={np.degrees(abs(delta)):.1f}°, "
                                f"constraint[{i}]={np.degrees(c_mean):.1f}°)"
                            )
            else:
                self.last_constraint_mean = None
                self.last_constraint_std = None
                joint_mu = goal_mu
                joint_sigma = goal_sigma

            self.last_joint_mean = np.array([[joint_mu]], dtype=np.float64)
            self.last_joint_std = np.array([[joint_sigma]], dtype=np.float64)

            if deterministic:
                action_angle = joint_mu
            else:
                action_angle = float(np.random.normal(joint_mu, joint_sigma))
            action_angle = float(np.mod(action_angle, 2 * np.pi))

            if self.verbose:
                print(f"Lambdas: {lambdas}")
                print(f"Goal angle means: {goal_means_arr}")
                print(f"Goal angle stds: {goal_stds_arr}")
                if len(constraint_means) > 0:
                    print(
                        f"Constraint angle means: {np.asarray(constraint_means, dtype=np.float64)}"
                    )
                    orig_arr = np.asarray(
                        original_constraint_stds, dtype=np.float64
                    )
                    clamp_arr = np.asarray(constraint_stds, dtype=np.float64)
                    if np.any(clamp_arr < orig_arr - 1e-8):
                        print(f"Constraint angle stds (raw): {orig_arr}")
                        print(
                            f"Constraint angle stds (clamped, ratio={max_ratio}): {clamp_arr}"
                        )
                    else:
                        print(f"Constraint angle stds: {clamp_arr}")
                print(f"Composed angle mean: {joint_mu}")
                print(f"Composed angle std: {joint_sigma}")
                if _deflection_info is not None:
                    print(f"  {_deflection_info}")
                print(f"Action angle: {action_angle}")

            action_np = np.array([action_angle], dtype=np.float32)
            return (
                np.array([action_np]) if self.for_eval else action_np,
                None,
            )

        new_mean, new_std = self._fuse_gaussians(
            goal_dist_params, constraint_dist_params, lambdas
        )

        goal_weights = torch.ones(
            len(goal_dist_params), dtype=new_mean.dtype, device=new_mean.device
        )
        goal_agg = self._aggregate_weighted_gaussians(
            goal_dist_params, goal_weights
        )

        constraint_agg: tuple[Tensor, Tensor] | None = None
        if constraint_dist_params:
            lambda_tensor = torch.tensor(
                lambdas[0, :],
                dtype=new_mean.dtype,
                device=new_mean.device,
            )
            constraint_agg = self._aggregate_weighted_gaussians(
                constraint_dist_params, lambda_tensor
            )

        joint_env_mean, joint_env_std = self._to_action_space_gaussian(
            new_mean, new_std
        )
        self.last_joint_mean = joint_env_mean.detach().cpu().numpy()
        self.last_joint_std = joint_env_std.detach().cpu().numpy()

        if goal_agg is not None:
            goal_env_mean, goal_env_std = self._to_action_space_gaussian(
                goal_agg[0], goal_agg[1]
            )
            self.last_goal_mean = goal_env_mean.detach().cpu().numpy()
            self.last_goal_std = goal_env_std.detach().cpu().numpy()
        else:
            self.last_goal_mean = None
            self.last_goal_std = None
        if constraint_agg is not None:
            constraint_env_mean, constraint_env_std = (
                self._to_action_space_gaussian(
                    constraint_agg[0], constraint_agg[1]
                )
            )
            self.last_constraint_mean = (
                constraint_env_mean.detach().cpu().numpy()
            )
            self.last_constraint_std = constraint_env_std.detach().cpu().numpy()
        else:
            self.last_constraint_mean = None
            self.last_constraint_std = None

        # Sample from squashed Gaussian: pi(a) ~ tanh(N(mu_new, sigma_new^2))
        if deterministic:
            action = torch.tanh(new_mean)
        else:
            composed_dist = torch.distributions.Normal(new_mean, new_std)
            action = torch.tanh(composed_dist.rsample())

        if self.verbose:
            self._log_continuous_debug(
                lambdas,
                goal_dist_params,
                constraint_dist_params,
                new_mean,
                new_std,
                action,
            )

        # Convert to numpy and reshape to action space shape
        action_np = (
            action.cpu().numpy().reshape((-1, *self.model.action_space.shape))
        )

        # Unscale/clip actions as in SB3's BasePolicy.predict
        if isinstance(self.model.action_space, spaces.Box):
            if self.model.policy.squash_output:
                action_np = self.model.policy.unscale_action(action_np)
            else:
                action_np = np.clip(
                    action_np,
                    self.model.action_space.low,
                    self.model.action_space.high,
                )

        # Remove batch dimension
        action_np = action_np.squeeze(axis=0)

        return (
            np.array([action_np]) if self.for_eval else action_np,
            None,
        )

    def _log_continuous_debug(
        self,
        lambdas: NDArray[np.floating],
        goal_dist_params: list[tuple[Tensor, Tensor, dict[str, Tensor]]],
        constraint_dist_params: list[tuple[Tensor, Tensor, dict[str, Tensor]]],
        new_mean: Tensor,
        new_std: Tensor,
        action: Tensor,
    ) -> None:
        """Print debug information for continuous CPC prediction."""
        print(f"Lambdas: {lambdas}")
        print(f"Goal means: {[p[0].cpu().numpy() for p in goal_dist_params]}")
        print(
            f"Goal stds: {[torch.exp(p[1]).cpu().numpy() for p in goal_dist_params]}"
        )
        if constraint_dist_params:
            print(
                f"Constraint means: "
                f"{[p[0].cpu().numpy() for p in constraint_dist_params]}"
            )
            print(
                f"Constraint stds: "
                f"{[torch.exp(p[1]).cpu().numpy() for p in constraint_dist_params]}"
            )
        print(f"Composed mean: {new_mean.cpu().numpy()}")
        print(f"Composed std: {new_std.cpu().numpy()}")
        print(f"Action: {action.cpu().numpy()}")


class CPCLowLevelPolicy(
    LowLevelPolicy[
        CPCCompositePolicy, CPCCompositePolicyConfig, ObsType, NDArray
    ]
):
    """
    Low-level policy wrapper for CPCCompositePolicy.
    """

    policy_args_validator = CPCCompositePolicyConfig
    policy_args_reader = CPCCompositePolicyConfigReader

    def define_policy(
        self, policy_args: CPCCompositePolicyConfig
    ) -> CPCCompositePolicy:
        if isinstance(policy_args, dict):
            policy_args = self.policy_args_validator(**policy_args)
        policy = CPCCompositePolicy(
            tl_spec=self.tl_spec,
            predicates=self.tl_wrapper_args["atomic_predicates"],
            model=policy_args.model,
            var_value_info_generator=policy_args.var_value_info_generator,
            lambda_config=policy_args.lambda_config,
            normalize_lambdas=policy_args.normalize_lambdas,
            parser=policy_args.parser,
            goal_rep=policy_args.goal_rep,
            prohibited_action_generator=policy_args.prohibited_action_generator,
            verbose=policy_args.verbose,
        )
        return policy

    def act(
        self,
        obs: TLObs[ObsType],
        info=None,
        current_env=None,
        tl_wrapper_args={},
    ) -> NDArray:
        action, _ = self.policy.predict(obs)

        return action
