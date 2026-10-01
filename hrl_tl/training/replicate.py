"""Generic replicate training runner for SB3 pipelines."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rl_pipeline.sb3 import (
    SB3PipelineConfigReader,
    SB3ReplicatePipeline,
    SB3ReplicatePipelineConfig,
    SB3ReplicatePipelineConfigReader,
)
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.training.types import ReplayModel
from hrl_tl.utils.device import normalize_device


def run_replicate_training(
    config_path: str | Path,
    config_reader_cls: type[SB3PipelineConfigReader],
    retrain_model: bool = False,
    record_replays: bool = True,
    replay_model: ReplayModel = "best",
    device: str | None = None,
    num_replicates: int | None = None,
    replicate_start_id: int | None = None,
    verbose: bool = True,
    post_config_hook: Callable[
        [SB3ReplicatePipelineConfigReader, SB3ReplicatePipelineConfig], None
    ]
    | None = None,
) -> tuple[SB3ReplicatePipeline, list[BaseAlgorithm]]:
    """Runs replicate training for an SB3 pipeline.

    Args:
        config_path: Path to the replicate experiment configuration YAML file.
        config_reader_cls: The SB3PipelineConfigReader subclass to
            parameterize the replicate reader with.
        retrain_model: Whether to force retraining even if model files exist.
        record_replays: Whether to record rollout replays after training.
        replay_model: Which model checkpoint to use for replays ('best',
            'final', or 'latest').
        device: Optional compute device override (e.g., '0', '1', 'cuda:0',
            'cpu').
        num_replicates: Optional override for number of replicates to train.
        replicate_start_id: Optional override for starting replicate ID.
        verbose: Whether to print verbose progress information.
        post_config_hook: Optional callback invoked after config construction,
            receiving the reader and the built config. Use for
            algorithm-specific post-processing (e.g., injecting HIRO goal
            spaces).

    Returns:
        A tuple of (pipeline, models), where pipeline is the executed
        SB3ReplicatePipeline and models is the list of best loaded models.

    Raises:
        FileNotFoundError: If the configuration YAML file does not exist.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")

    reader = SB3ReplicatePipelineConfigReader[config_reader_cls].from_yaml(  # ty: ignore[invalid-type-form]
        str(path)
    )

    if num_replicates is not None:
        reader.replicate_config.num_replicates = num_replicates
    if replicate_start_id is not None:
        reader.replicate_config.replicate_start_id = replicate_start_id

    exp_config: SB3ReplicatePipelineConfig = reader.to_config()

    normalized_device = normalize_device(device) if device is not None else None
    for ind_config in exp_config.ind_pipeline_configs:
        ind_config.retrain_model = retrain_model
        if normalized_device is not None:
            ind_config.device = normalized_device

    if post_config_hook is not None:
        post_config_hook(reader, exp_config)

    pipeline = SB3ReplicatePipeline(config=exp_config, verbose=verbose)
    pipeline.train_on_unsaved_model()

    models = pipeline.load_models(replay_model)
    if record_replays:
        pipeline.record_replays(models, verbose=verbose)

    return pipeline, models
