import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from hrl_tl.baseline.log.wandb_logger import WandbLogger
from hrl_tl.baseline.utils.get_args import get_args


def concat_csv_columnwise_and_delete(folder_path, output_file="output.csv"):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate column-wise (axis=1)
    combined_df = pd.concat(dataframes, axis=1)

    # Save to output file
    output_file = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

    # Delete original CSV files
    for file in csv_files:
        os.remove(os.path.join(folder_path, file))

    print("Original CSV files deleted.")


def override_args(env_name: str | None = None):
    args = get_args(verbose=False)
    if env_name is not None:
        args.env_name = env_name
    file_path = "assets/env_params.json"
    current_params = load_hyperparams(
        file_path=file_path, env_name=args.env_name
    )

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


def load_hyperparams(file_path, env_name):
    """Load hyperparameters for a specific environment from a JSON file."""
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
            return hyperparams.get(env_name, {})
    except FileNotFoundError:
        print(
            f"No file found at {file_path}. Returning default empty dictionary for {env_name}."
        )
        return {}


def seed_all(seed=0):
    # Set the seed for hash-based operations in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Ensure reproducibility of PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_summary(model, model_name="Model"):
    # Header with model name
    print("=" * 50)
    print(f"{model_name:<30} {'Param # (K)':>15}")
    print("=" * 50)

    total_params = 0
    total_trainable_params = 0
    total_non_trainable_params = 0

    # Iterate through model layers
    for name, param in model.named_parameters():
        num_params = np.prod(param.size())
        total_params += num_params

        if param.requires_grad:
            total_trainable_params += num_params
        else:
            total_non_trainable_params += num_params

        # # Layer name and number of parameters (in thousands)
        # print(f"{name:<30} {num_params / 1e3:>15,.2f} K")

    # Footer with totals
    # print("=" * 50)
    print(f"Total Parameters: {total_params / 1e3:,.2f} K")
    print(f"Trainable Parameters: {total_trainable_params / 1e3:,.2f} K")
    print(
        f"Non-trainable Parameters: {total_non_trainable_params / 1e3:,.2f} K"
    )
    print("=" * 50)


def setup_logger(args, unique_id: str, exp_time: str, seed: int):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Get the current date and time
    now = datetime.now()
    args.seed = seed

    if args.group is None:
        args.group = "-".join((exp_time, unique_id))

    if args.name is None:
        args.name = "-".join(
            (args.algo_name, args.env_name, unique_id, "seed:" + str(seed))
        )

    if args.project is None:
        args.project = args.env_name

    args.logdir = os.path.join(args.logdir, args.group)

    default_cfg = vars(args)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
        fps=args.render_fps,
    )
    logger.save_config(default_cfg, verbose=args.verbose)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, writer
