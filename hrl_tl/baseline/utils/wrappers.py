import gymnasium as gym
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class HIROWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(HIROWrapper, self).__init__(env)

    def reset(self, **kwargs):
        state_dict, info = self.env.reset(**kwargs)
        # state_dict = {"observation": state, "desired_goal": self.desired_goal}

        return state_dict, info

    def step(self, action):
        # Call the original step method
        state_dict, reward, termination, truncation, info = self.env.step(
            action
        )
        # state_dict = {"observation": state, "desired_goal": self.desired_goal}

        done = termination or truncation

        return state_dict, reward, done, info

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)

    @property
    def max_episode_steps(self):
        """Return the maximum number of steps per episode."""
        if self.env.spec and hasattr(self.env.spec, "max_episode_steps"):
            max_ep_steps: int | None = self.env.spec.max_episode_steps
            if max_ep_steps is not None:
                return max_ep_steps
            else:
                raise ValueError(
                    "`None` value for `max_episode_steps` in the environment spec."
                )
        else:
            raise AttributeError(
                "The environment spec does not have a 'max_episode_steps' attribute."
            )
