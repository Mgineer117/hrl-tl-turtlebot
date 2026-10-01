"""Two-stage hyperparameter tuning workflow utilities for HRL and baselines.

Provides configuration definitions, Optuna study creation, tuning execution,
and automated export of best parameters into replicate training configurations.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import optuna
import yaml
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from rl_pipeline.sb3 import (
    SB3OptunaConfig,
    SB3OptunaDashboardConfig,
    SB3OptunaParamConfig,
    SB3Pipeline,
    decode_trial_params,
    set_nested_value,
)


class TuningWorkflowConfig(BaseModel):
    """Configuration for hyperparameter optimization workflow.

    Attributes:
      study_name: Name of the Optuna study.
      db_filename: SQLite database file name for study persistence.
      db_dir: Directory where SQLite database files are stored.
      total_timesteps: Number of timesteps or epochs to train per trial.
      direction: Optimization direction, either 'maximize' or 'minimize'.
      metric: Scalar evaluation metric to optimize.
      n_trials: Number of trials to run during optimization.
      n_startup_trials: Number of initial random trials before TPE sampling.
      n_evaluations: Number of intermediate evaluations per trial.
      eval_freq: Optional explicit evaluation interval in timesteps.
      n_eval_episodes: Number of evaluation episodes per evaluation.
      n_jobs: Number of parallel trial workers.
      parallel_backend: Backend for parallel execution ('process' or 'thread').
      launch_dashboard: Whether to launch the Optuna web dashboard.
      dashboard_host: Host IP or hostname for Optuna dashboard.
      dashboard_port: Port number for Optuna dashboard web server.
      tune_params: List of hyperparameter search space definitions.
    """

    study_name: str
    db_filename: str
    db_dir: Path = Path("db")
    total_timesteps: int
    direction: Literal["maximize", "minimize"] = "maximize"
    metric: str = "mean_reward"
    n_trials: int = 25
    n_startup_trials: int = 5
    n_evaluations: int = 5
    eval_freq: int | None = None
    n_eval_episodes: int = 100
    n_jobs: int = 1
    parallel_backend: Literal["process", "thread"] = "process"
    launch_dashboard: bool = False
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    tune_params: list[SB3OptunaParamConfig] = Field(default_factory=list)


@dataclass(frozen=True)
class TuningEnvPaths:
    """Environment-specific configuration file paths for hyperparameter tuning.

    Attributes:
      base_pipeline: Path to the base single pipeline configuration template.
      tuning_config: Path to the Optuna tuning workflow configuration YAML.
      best_export: Target path where best parameters YAML is exported.
      replicate_export: Target path where replicate training YAML is exported.
    """

    base_pipeline: Path
    tuning_config: Path
    best_export: Path
    replicate_export: Path


def build_optuna_config(
    workflow_config: TuningWorkflowConfig,
) -> SB3OptunaConfig:
    """Build SB3OptunaConfig from TuningWorkflowConfig.

    Args:
      workflow_config: Declarative workflow configuration.

    Returns:
      Configured Optuna settings for SB3Pipeline.optimize.
    """
    workflow_config.db_dir.mkdir(parents=True, exist_ok=True)
    db_path = (workflow_config.db_dir / workflow_config.db_filename).resolve()
    storage_url = f"sqlite:///{db_path}"

    return SB3OptunaConfig(
        storage_url=storage_url,
        study_name=workflow_config.study_name,
        direction=workflow_config.direction,
        metric=workflow_config.metric,
        n_trials=workflow_config.n_trials,
        n_startup_trials=workflow_config.n_startup_trials,
        n_evaluations=workflow_config.n_evaluations,
        eval_freq=workflow_config.eval_freq,
        n_eval_episodes=workflow_config.n_eval_episodes,
        total_timesteps=workflow_config.total_timesteps,
        n_jobs=workflow_config.n_jobs,
        parallel_backend=workflow_config.parallel_backend,
        tune_params=workflow_config.tune_params,
        dashboard=SB3OptunaDashboardConfig(
            launch=workflow_config.launch_dashboard,
            host=workflow_config.dashboard_host,
            port=workflow_config.dashboard_port,
        ),
    )


def run_tuning_stage(
    pipeline: SB3Pipeline,
    optuna_config: SB3OptunaConfig,
    export_yaml_path: Path | None = None,
) -> optuna.Study:
    """Execute Optuna hyperparameter optimization and export results.

    Args:
      pipeline: Configured pipeline to optimize.
      optuna_config: Tuning parameters and sampler settings.
      export_yaml_path: Target file path to dump best parameter summary YAML.

    Returns:
      Completed Optuna study.
    """
    print(
        f"Starting Optuna study '{optuna_config.study_name}' "
        f"({optuna_config.n_trials} trials, {optuna_config.total_timesteps} steps/trial)..."
    )
    study = pipeline.optimize(optuna_config=optuna_config)

    print("\n" + "=" * 60)
    print(f"Optimization finished for {study.study_name}!")
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Value: {study.best_value:.4f}")
    print("Best Parameters:")
    decoded_params = decode_trial_params(
        study.best_params, optuna_config.tune_params
    )
    for param_name, val in decoded_params.items():
        print(f"  {param_name}: {val}")
    print("=" * 60 + "\n")

    if export_yaml_path is not None:
        pipeline.export_best_params(study, out_path=export_yaml_path)
        print(f"Exported best parameters to {export_yaml_path}")

    return study


def export_replicate_yaml(
    source_yaml_path: Path,
    target_yaml_path: Path,
    best_params: Mapping[str, Any],
    tune_params: Sequence[SB3OptunaParamConfig] = (),
    num_replicates: int = 5,
    replicate_signature: str = "rep_{rep_id}",
    replicate_start_id: int = 0,
) -> Path:
    """Create a 5-replicate configuration YAML with tuned parameters merged.

    Args:
      source_yaml_path: Base single pipeline YAML path.
      target_yaml_path: Destination path for replicate YAML.
      best_params: Best sampled parameters from Optuna.
      tune_params: Hyperparameter search-space specifications used for target resolution.
      num_replicates: Number of replicate runs to generate (default 5).
      replicate_signature: Template string for replicate folder names (default 'rep_{rep_id}').
      replicate_start_id: Starting replicate index (default 0).

    Returns:
      Path to the newly generated replicate YAML.
    """
    with source_yaml_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Wrap inside replicate_config and single_pipeline_config if not already wrapped
    if "replicate_config" in config_data:
        replicate_dict = config_data
        single_dict = replicate_dict["single_pipeline_config"]
    else:
        single_dict = config_data
        replicate_dict = {
            "replicate_config": {
                "num_replicates": num_replicates,
                "replicate_start_id": replicate_start_id,
                "replicate_signature": replicate_signature,
            },
            "single_pipeline_config": single_dict,
        }

    target_by_name = {
        param.name: param.target for param in tune_params if param.target
    }

    # Merge best parameters into single pipeline config
    for raw_key, value in best_params.items():
        key = target_by_name.get(raw_key, raw_key)
        if "." in key or key.endswith("]"):
            set_nested_value(single_dict, key, value)
        elif (
            "wrapper_kwargs" in single_dict
            and key in single_dict["wrapper_kwargs"]
        ):
            single_dict["wrapper_kwargs"][key] = value
        else:
            single_dict.setdefault("algo_kwargs", {})[key] = value

    target_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with target_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            replicate_dict, f, default_flow_style=False, sort_keys=False
        )

    print(f"Generated replicate configuration at {target_yaml_path}")
    return target_yaml_path
