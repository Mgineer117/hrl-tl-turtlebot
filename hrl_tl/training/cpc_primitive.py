"""CPC / GC-LTL primitive policy replicate training execution module.

Provides a thin wrapper around the generic replicate training runner,
parameterized with the GC-LTL pipeline configuration reader.
"""

from __future__ import annotations

from pathlib import Path

from rl_pipeline.sb3 import SB3ReplicatePipeline
from stable_baselines3.common.base_class import BaseAlgorithm

from hrl_tl.config.wrapper import GCLTLPipelineConfigReader
from hrl_tl.training.replicate import run_replicate_training
from hrl_tl.training.types import ReplayModel


def run_cpc_primitive_training(
    config_path: str | Path,
    retrain_model: bool = False,
    record_replays: bool = True,
    replay_model: ReplayModel = "best",
    device: str | None = None,
    num_replicates: int | None = None,
    replicate_start_id: int | None = None,
    verbose: bool = True,
) -> tuple[SB3ReplicatePipeline, list[BaseAlgorithm]]:
    """Executes replicate training for a CPC / GC-LTL primitive policy.

    Args:
        config_path: Path to the replicate experiment configuration YAML file.
        retrain_model: Whether to force retraining even if model files exist.
        record_replays: Whether to record rollout replays after training.
        replay_model: Which model checkpoint to use for replays ('best',
          'final', or 'latest').
        device: Optional compute device override (e.g., '0', '1', 'cuda:0',
          'cpu').
        num_replicates: Optional override for number of replicates to train.
        replicate_start_id: Optional override for starting replicate ID.
        verbose: Whether to print verbose progress information.

    Returns:
        A tuple of (pipeline, models), where pipeline is the executed
        SB3ReplicatePipeline and models is the list of best loaded models.

    Raises:
        FileNotFoundError: If the configuration YAML file does not exist.
    """
    return run_replicate_training(
        config_path=config_path,
        config_reader_cls=GCLTLPipelineConfigReader,
        retrain_model=retrain_model,
        record_replays=record_replays,
        replay_model=replay_model,
        device=device,
        num_replicates=num_replicates,
        replicate_start_id=replicate_start_id,
        verbose=verbose,
    )
