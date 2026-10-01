import os
from typing import Any, Generic, TypeVar

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, computed_field
from rl_pipeline.core.eval import scaled_same_convolve

FileReaderConfig = TypeVar("FileReaderConfig", bound=BaseModel)


class EvalFileReader(Generic[FileReaderConfig]):
    @staticmethod
    def read_eval_files(
        eval_data_config: FileReaderConfig, num_reps: int
    ) -> tuple[list[NDArray], list[NDArray]]: ...

    @staticmethod
    def format_data_length(
        reward_data: list[NDArray],
        timestep_data: list[NDArray],
        max_timesteps: int,
        data_points: int,
    ) -> tuple[list[NDArray], list[NDArray]]:
        """
        Interpolate the datapoints between 0 to max_timesteps with designated number of data_points.
        If timestep_data is shorter than data_points, pad it with the last value.
        If timestep_data is longer than data_points, truncate it.
        """
        formatted_reward_data = []
        formatted_timestep_data = []

        # Create target timesteps array for interpolation
        target_timesteps = np.linspace(0, max_timesteps, data_points)

        for rewards, timesteps in zip(reward_data, timestep_data):
            # Ensure timesteps start from 0 and don't exceed max_timesteps
            timesteps = np.clip(timesteps, 0, max_timesteps)

            if len(timesteps) == 0:
                # Handle empty data case
                interpolated_rewards = np.zeros(data_points)
            else:
                # Ensure we have at least the start and end points for interpolation
                if timesteps[0] > 0:
                    # Prepend 0 timestep with first reward value
                    timesteps = np.concatenate([[0], timesteps])
                    rewards = np.concatenate([[rewards[0]], rewards])

                if timesteps[-1] < max_timesteps:
                    # Append max_timesteps with last reward value
                    timesteps = np.concatenate([timesteps, [max_timesteps]])
                    rewards = np.concatenate([rewards, [rewards[-1]]])

                # Interpolate to get desired number of points
                interpolated_rewards = np.interp(
                    target_timesteps, timesteps, rewards
                )

            formatted_reward_data.append(interpolated_rewards)
            formatted_timestep_data.append(target_timesteps.copy())

        return formatted_reward_data, formatted_timestep_data

    @staticmethod
    def get_plot_data(
        reward_data: list[NDArray],
        timestep_data: list[NDArray],
        smoothing_window: int,
        method_name: str,
        start_timestep: int = 0,
    ) -> list[dict[str, Any]]:
        plot_data = []
        # Add data from this configuration to the plot data
        for rep_id, (rewards, steps) in enumerate(
            zip(reward_data, timestep_data)
        ):
            convolved_rewards = scaled_same_convolve(
                rewards, size=smoothing_window
            )
            for step, reward in zip(steps, convolved_rewards):
                plot_data.append(
                    {
                        "timestep": step + start_timestep,
                        "reward": reward,
                        "replicate": rep_id,
                        "method": method_name,  # Use config name for grouping/legend
                    }
                )
        return plot_data


class EvalDataConfig(BaseModel, Generic[FileReaderConfig]):
    name: str = "Ours"
    num_replicates: int = 10
    max_timesteps: int = 500_000
    data_points: int = 200
    smooth_window: int = 20
    file_reader_class: type[EvalFileReader]
    eval_file_config: FileReaderConfig
    start_timestep: int = 0


class SB3FileReaderConfig(BaseModel):
    models_dir: str = "out/poc/test/2.b/2.b.q/fourroom_ppo_test_5.0M"
    ind_eval_file: str = "rep_{rep_id}/eval/evaluations.npz"

    @computed_field
    @property
    def ind_eval_file_path(self) -> str:
        return os.path.join(self.models_dir, self.ind_eval_file)


class SB3EvalFileReader(EvalFileReader[SB3FileReaderConfig]):
    @staticmethod
    def read_eval_files(
        eval_data_config: SB3FileReaderConfig, num_reps: int
    ) -> tuple[list[NDArray], list[NDArray]]:
        # Load evaluation data from multiple repetitions

        reward_data: list[NDArray] = []
        timestep_data: list[NDArray] = []
        print(f"Loading evaluation data from {num_reps} repetitions...")
        successful_loads = 0

        for rep_id in range(num_reps):
            file_path = eval_data_config.ind_eval_file_path.format(
                rep_id=rep_id
            )
            try:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue

                eval_data = np.load(file_path)

                # Check if required keys exist
                if "results" not in eval_data or "timesteps" not in eval_data:
                    print(f"Warning: Required keys not found in {file_path}")
                    continue

                reward_data.append(eval_data["results"].mean(axis=1))
                timestep_data.append(eval_data["timesteps"])
                successful_loads += 1
                print(f"✓ Loaded rep_{rep_id}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if successful_loads == 0:
            print("Error: No evaluation data could be loaded!")
            print(
                f"Expected files in format: {eval_data_config.ind_eval_file_path}"
            )
            exit(1)

        print(f"Successfully loaded {successful_loads}/{num_reps} repetitions")

        return reward_data, timestep_data


class WandbCSVReaderConfig(BaseModel):
    file_path: str


class WandbCSVEvalFileReader(EvalFileReader[WandbCSVReaderConfig]):
    @staticmethod
    def read_eval_files(
        eval_data_config: WandbCSVReaderConfig, num_reps: int
    ) -> tuple[list[NDArray], list[NDArray]]:
        csv_path = eval_data_config.file_path

        # Read CSV using polars
        df = pl.read_csv(csv_path)

        # The first columns is timesteps
        # The following columns correspond to runs (replicates), and each replicate has 3 identical reward columns,
        # so only the 3*<replicate_id> + 1 columns are needed for each replicate.
        # The reward columns have missing values as each run has different timesteps that collected the rewards,
        # so each runs timesteps_data should be aligned with its existing reward data.
        reward_data = []
        timestep_data = []

        print(f"Loading evaluation data from CSV: {csv_path}")
        print(f"CSV shape: {df.shape}")

        successful_loads = 0

        for rep_id in range(num_reps):
            # Calculate the column index for this replicate
            # Each replicate has 3 columns, we want the middle one (3*rep_id + 1)
            # But we need to account for the step column being first, so add 1
            reward_col_idx = 3 * rep_id + 1 + 1  # +1 for step column offset

            # Check if this column exists
            if reward_col_idx >= df.shape[1]:
                print(
                    f"Warning: Column index {reward_col_idx} for rep_{rep_id} exceeds CSV columns ({df.shape[1]})"
                )
                continue

            try:
                # Get the reward column name
                reward_col_name = df.columns[reward_col_idx]

                # Extract rewards and filter out null values
                reward_series = df[reward_col_name]
                step_series = df["Step"]

                # Convert string rewards to float, treating empty strings as null
                # Replace empty strings with None first, then convert to float
                reward_series = reward_series.map_elements(
                    lambda x: None if x == "" else x, return_dtype=pl.String
                ).cast(pl.Float64, strict=False)

                # Create mask for non-null values
                mask = reward_series.is_not_null()

                # Filter both rewards and timesteps to remove null entries
                valid_rewards = reward_series.filter(mask).to_numpy()
                valid_timesteps = step_series.filter(mask).to_numpy()

                if len(valid_rewards) == 0:
                    print(
                        f"Warning: No valid data for rep_{rep_id} (column: {reward_col_name})"
                    )
                    continue

                reward_data.append(valid_rewards)
                timestep_data.append(valid_timesteps)
                successful_loads += 1
                print(
                    f"✓ Loaded rep_{rep_id} from column '{reward_col_name}' ({len(valid_rewards)} data points)"
                )

            except Exception as e:
                print(f"Error loading rep_{rep_id}: {e}")
                continue

        if successful_loads == 0:
            print("Error: No evaluation data could be loaded!")
            print(
                f"Expected reward columns at indices: {[3 * i + 2 for i in range(num_reps)]}"
            )  # +2 for step column + 0-based indexing
            exit(1)

        print(f"Successfully loaded {successful_loads}/{num_reps} repetitions")

        return reward_data, timestep_data
