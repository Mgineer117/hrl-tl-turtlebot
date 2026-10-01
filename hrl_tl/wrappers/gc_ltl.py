from typing import Any, Callable, Generic, Literal, TypeVar

import numpy as np
from gym_tl_tools import BaseVarValueInfoGenerator, Parser
from gymnasium import Env, Wrapper
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Space
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from typing_extensions import SupportsFloat


class Predicate(BaseModel):
    name: str
    formula: str
    num_entities: int = 1

    model_config = ConfigDict(arbitrary_types_allowed=True)


class InfoUpdates(BaseModel):
    is_success: bool
    goal: Predicate
    goal_name: str
    goal_index: int
    goal_rob: float
    pred_robustness: dict[str, float]
    var_value_info: dict[str, float]

    model_config = ConfigDict(arbitrary_types_allowed=True)


RepT = TypeVar("RepT")


class GoalRep(Generic[RepT]):
    def __init__(self, predicates: list[Predicate]):
        self.predicates: list[Predicate] = sorted(
            predicates, key=lambda p: p.name
        )
        self.pred_names: list[str] = [p.name for p in self.predicates]
        self.encoding = self.init_encoding()

    def init_encoding(self) -> dict[int, RepT]:
        raise NotImplementedError

    def pred2rep(self, pred_index: int) -> RepT:
        return self.encoding[pred_index]

    def goal_space(self) -> Space:
        raise NotImplementedError


class IndexGoalRep(GoalRep[int]):
    def init_encoding(self) -> dict[int, int]:
        return {i: i for i in range(len(self.predicates))}

    def goal_space(self):
        return Discrete(len(self.predicates))


class OneHotGoalRep(GoalRep[NDArray[np.number]]):
    def init_encoding(self) -> dict[int, NDArray[np.number]]:
        encoding: dict[int, NDArray[np.number]] = {}
        for i in range(len(self.predicates)):
            one_hot = np.zeros(len(self.predicates), dtype=np.int64)
            one_hot[i] = 1
            encoding[i] = one_hot
        return encoding

    def goal_space(self):
        return MultiDiscrete([2] * len(self.predicates))


class GCLTLWrapper(
    Wrapper[
        dict[str, ObsType | np.int64 | NDArray[np.number]],
        ActType,
        ObsType,
        ActType,
    ],
    RecordConstructorArgs,
):
    """
    A wrapper that converts a Gymnasium environment into one that supports goal-conditioned LTL tasks from GCRL-LTL paper.

    Observation
    -----------
    The observation space is modified to include the original observation and the current goal (predicate),
    randomly sampled from a provided list of predicates at each reset.
    More specifically, the observation is a dictionary with two keys:
        - "obs": the original observation from the environment
        - "goal_pred": the index of the current goal predicate (the key can be changed via `dict_goal_pred_key` argument)

    Reward
    -------
    The reward is modified to be 1 if the current observation satisfies the goal predicate, and 0 otherwise
    (i.e., satisfaction is defined as the strictly positive robustness of the current observation with respect to the goal predicate).

    Info
    ----
    The info dictionary is augmented with the following keys:
        - "success": bool
          whether the goal predicate is satisfied
        - "goal": str
          the current goal predicate
        - "goal_index": int
          the index of the current goal predicate
        - "pred_robustness": dict[str, float]
          the robustness of the current observation with respect to the predicates
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        predicates: list[Predicate],
        var_value_info_generator: BaseVarValueInfoGenerator[ObsType, ActType],
        early_termination: bool = True,
        dict_goal_pred_key: str = "goal_pred",
        reward_type: Literal["sparse", "dense"] = "sparse",
        step_penalty: float = 0.05,
        dense_reward_scale: float = 12.0,
        goal_rep: Literal["index", "one_hot"] = "index",
        closest_goal_pos: bool = False,
        parser: Parser = Parser(),
    ):
        RecordConstructorArgs.__init__(
            self,
            predicates=predicates,
            var_value_info_generator=var_value_info_generator,
            early_termination=early_termination,
            parser=parser,
            step_penalty=step_penalty,
            reward_type=reward_type,
            dense_reward_scale=dense_reward_scale,
            goal_rep=goal_rep,
            closest_goal_pos=closest_goal_pos,
        )
        Wrapper.__init__(self, env)

        self.reward_type: Literal["sparse", "dense"] = reward_type
        self.step_penalty: float = step_penalty
        self.dense_reward_scale: float = dense_reward_scale
        self.parser: Parser = parser
        self.predicates: list[Predicate] = sorted(
            predicates, key=lambda p: p.name
        )
        self.var_value_info_generator: BaseVarValueInfoGenerator[
            ObsType, ActType
        ] = var_value_info_generator
        self.early_termination: bool = early_termination
        self.closest_goal_pos: bool = closest_goal_pos
        self.num_entities: list[int] = [p.num_entities for p in self.predicates]

        self.goal_rep: GoalRep
        match goal_rep:
            case "index":
                self.goal_rep = IndexGoalRep(self.predicates)
            case "one_hot":
                self.goal_rep = OneHotGoalRep(self.predicates)
            case _:
                raise ValueError(
                    f"Invalid goal_rep: {goal_rep}. Must be 'index' or 'one_hot'."
                )

        self._append_data_func: Callable[
            [ObsType, int | NDArray],
            dict[str, ObsType | np.int64 | NDArray | int],
        ]
        # Find the observation space
        match env.observation_space:
            case Dict():
                assert dict_goal_pred_key not in env.observation_space.spaces, (
                    f"Key '{dict_goal_pred_key}' already exists in the observation space. "
                    "Please choose a different key."
                )
                obs_space_dict = {
                    **env.observation_space.spaces,
                    dict_goal_pred_key: self.goal_rep.goal_space(),
                }

                self._append_data_func = lambda obs, goal_pred: {
                    **obs,
                    dict_goal_pred_key: goal_pred,
                }
            # case Tuple():
            #     observation_space = Tuple(
            #         env.observation_space.spaces + (aut_state_space,)
            #     )
            #     self._append_data_func = lambda obs, aut_state: obs + (aut_state,)
            case _:
                obs_space_dict = {
                    "obs": env.observation_space,
                    dict_goal_pred_key: self.goal_rep.goal_space(),
                }
                self._append_data_func = lambda obs, goal_pred: {
                    "obs": obs,
                    dict_goal_pred_key: goal_pred,
                }

        # obs_space_dict |= (
        #     {"closest_goal_pos": obs_space_dict["agent_pos"]}
        #     if self.closest_goal_pos
        #     else {}
        # )

        self.observation_space = Dict(obs_space_dict)
        self._obs_postprocess_func = lambda obs: obs

    def observation(
        self, observation: ObsType
    ) -> dict[str, ObsType | np.int64 | int | NDArray]:
        """
        Process the observation to include the automaton state.

        Parameters
        ----------
        observation : ObsType
            The original observation from the environment.

        Returns
        -------
        new_obs: dict[str,ObsType|int]
            The processed observation with the automaton state appended.
        """
        new_obs: dict[str, ObsType | np.int64 | NDArray | int] = (
            self._append_data_func(
                observation, self.goal_rep.pred2rep(self.goal_pred_index)
            )
        )
        return new_obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, ObsType | np.int64 | NDArray], dict[str, Any]]:
        """
        Reset the environment and sample a new goal predicate.

        Parameters
        ----------
        seed : int | None
            The seed for the environment's random number generator.
        options : dict[str, Any] | None
            Additional options for resetting the environment.

        Returns
        -------
        observation : dict[str, ObsType | np.int64 | NDArray]
            The initial observation after reset, including the new goal predicate.
        info : dict[str, Any]
            Additional information from the environment reset.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # The sampling probabilitiy depends on the number of entities for each predicate
        probabilities = np.array(self.num_entities) / sum(self.num_entities)
        self.goal_pred_index: int = self.np_random.choice(
            len(self.predicates), p=probabilities
        )
        self.current_goal = self.predicates[self.goal_pred_index]

        info_updates: InfoUpdates = self._info_updates(obs, info)
        info.update(info_updates.model_dump())

        return self.observation(obs), info

    def _info_updates(self, obs: ObsType, info: dict[str, Any]) -> InfoUpdates:
        """
        Update the info dictionary with goal and robustness information.

        This method is used to update the info dictionary with the current goal predicate,
        its index, and the robustness of the current observation with respect to all predicates.
        It is called during the step and reset methods.

        Parameters
        ----------
        obs : ObsType
            The current observation from the environment.
        info : dict[str, Any]
            The info dictionary to be updated.

        Returns
        -------
        info_updates : InfoUpdates
            The updates to be added to the info dictionary.
        """
        var_value_info_updates: dict[str, float] = self._var_value_info_updates(
            obs, info
        )
        pred_robustness: dict[str, float] = self._pred_rob_updates(
            var_value_info_updates
        )
        success: bool = pred_robustness[self.current_goal.name] > 0
        goal_rob: float = pred_robustness[self.current_goal.name]
        info_updates: InfoUpdates = InfoUpdates(
            is_success=success,
            goal=self.current_goal,
            goal_name=self.current_goal.name,
            goal_index=self.goal_pred_index,
            goal_rob=goal_rob,
            pred_robustness=pred_robustness,
            var_value_info=var_value_info_updates,
        )
        return info_updates

    def _var_value_info_updates(
        self, obs: ObsType, info: dict[str, Any]
    ) -> dict[str, float]:
        """
        Update the info dictionary with variable values based on the observation.

        This method is used to update the info dictionary with the values of the atomic predicates
        based on the current observation. It is called during the step and reset methods.

        Parameters
        ----------
        obs : ObsType
            The current observation from the environment.

        Returns
        -------
        var_value_info_updates : dict[str, Any]
            Updates to be added to the info dictionary containing variable values.
        """
        var_value_info_updates: dict[str, float] = (
            self.var_value_info_generator.get_var_values(self.env, obs, info)
        )

        return var_value_info_updates

    def _pred_rob_updates(
        self, var_value_info_updates: dict[str, float]
    ) -> dict[str, float]:
        """
        Compute the robustness of each predicate based on variable values.

        Parameters
        ----------
        var_value_info_updates : dict[str, float]
            A dictionary containing variable values needed to evaluate the predicates.

        Returns
        -------
        pred_robustness : dict[str, float]
            A dictionary mapping predicate names to their robustness values.
        """
        pred_robustness: dict[str, float] = {
            atom_pred.name: self.parser.tl2rob(
                atom_pred.formula, var_value_info_updates
            )
            for atom_pred in self.predicates
        }
        return pred_robustness

    def step(
        self, action: ActType
    ) -> tuple[
        dict[str, ObsType | np.int64 | NDArray],
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        """
        Take a step in the environment with the given action.

        Parameters
        ----------
        action : ActType
            The action to take in the environment.

        Returns
        -------
        new_obs: dict[str,ObsType|int]
            The new observation after taking the action.
        reward: SupportsFloat
            The reward received from the environment.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated.
        info: dict[str, Any]
            Additional information from the step.
            Should contain the variable keys and values that define the atomic predicates.
        """
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        info_updates: InfoUpdates = self._info_updates(obs, info)
        info.update(
            info_updates.model_dump() | {"original_reward": orig_reward}
        )

        reward: float = 0

        goal_reached: bool = info_updates.is_success
        match self.reward_type:
            case "sparse":
                # Reward is 1 if the current observation satisfies the goal predicate, 0 otherwise

                if goal_reached:
                    reward = 1.0
                else:
                    reward -= self.step_penalty

            case "dense":
                reward += info_updates.goal_rob / self.dense_reward_scale
            case _:
                raise ValueError(
                    f"Invalid reward_type: {self.reward_type}. Must be 'sparse' or 'dense'."
                )

        if goal_reached and self.early_termination:
            terminated: bool = True

        new_obs = self.observation(obs)
        return new_obs, reward, terminated, truncated, info


class FixedGCLTLWrapper(
    GCLTLWrapper[ObsType, ActType],
    Generic[ObsType, ActType],
):
    """
    A wrapper that converts a Gymnasium environment into one that supports goal-conditioned LTL tasks from GCRL-LTL paper.

    Observation
    -----------
    The observation space is modified to include the original observation and the current goal (predicate),
    randomly sampled from a provided list of predicates at each reset.
    More specifically, the observation is a dictionary with two keys:
        - "obs": the original observation from the environment
        - "goal_pred": the index of the current goal predicate (the key can be changed via `dict_goal_pred_key` argument)

    Reward
    -------
    The reward is modified to be 1 if the current observation satisfies the goal predicate, and 0 otherwise
    (i.e., satisfaction is defined as the strictly positive robustness of the current observation with respect to the goal predicate).

    Info
    ----
    The info dictionary is augmented with the following keys:
        - "success": bool
          whether the goal predicate is satisfied
        - "goal": str
          the current goal predicate
        - "goal_index": int
          the index of the current goal predicate
        - "pred_robustness": dict[str, float]
          the robustness of the current observation with respect to the predicates
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        goal_pred_name: str,
        predicates: list[Predicate],
        var_value_info_generator: BaseVarValueInfoGenerator[ObsType, ActType],
        early_termination: bool = True,
        dict_goal_pred_key: str = "goal_pred",
        reward_type: Literal["sparse", "dense"] = "sparse",
        step_penalty: float = 0.05,
        dense_reward_scale: float = 12.0,
        goal_rep: Literal["index", "one_hot"] = "index",
        parser: Parser = Parser(),
    ):
        RecordConstructorArgs.__init__(
            self,
            predicates=predicates,
            goal_pred_name=goal_pred_name,
            var_value_info_generator=var_value_info_generator,
            early_termination=early_termination,
            parser=parser,
            step_penalty=step_penalty,
            reward_type=reward_type,
            dense_reward_scale=dense_reward_scale,
            goal_rep=goal_rep,
        )
        GCLTLWrapper.__init__(
            self,
            env,
            predicates,
            var_value_info_generator,
            early_termination,
            dict_goal_pred_key,
            reward_type,
            step_penalty,
            dense_reward_scale,
            goal_rep,
            parser,
        )

        self.goal_pred_name: str = goal_pred_name
        self.goal_pred_index: int = next(
            i for i, p in enumerate(self.predicates) if p.name == goal_pred_name
        )
        self.current_goal = self.predicates[self.goal_pred_index]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, ObsType | np.int64 | NDArray], dict[str, Any]]:
        """
        Reset the environment and sample a new goal predicate.

        Parameters
        ----------
        seed : int | None
            The seed for the environment's random number generator.
        options : dict[str, Any] | None
            Additional options for resetting the environment.

        Returns
        -------
        observation : dict[str, ObsType | np.int64 | NDArray]
            The initial observation after reset, including the new goal predicate.
        info : dict[str, Any]
            Additional information from the environment reset.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        # The sampling probabilitiy depends on the number of entities for each predicate

        info_updates: InfoUpdates = self._info_updates(obs, info)
        info.update(info_updates.model_dump())

        return self.observation(obs), info
