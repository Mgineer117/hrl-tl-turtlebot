import os

from pydantic import BaseModel, computed_field


class SaveConfig(BaseModel):
    """Configuration for saving models."""

    models_dir: str = "out/poc/test"
    replicate_dir: str = ""
    model_name: str = "fourroom_ppo_test_fixed"
    model_filename: str = "final_model.zip"
    best_model_filename: str = "best_model.zip"
    monitor_dir: str = "monitor"
    tb_dir: str = "tb"
    eval_dir: str = "eval"
    eval_metrics_filename: str = "eval_metrics.yaml"
    include_extension: bool = True


SaveConfigReader = SaveConfig


class SavePathConfig(BaseModel):
    """Paths for saving models."""

    model_save_path: str
    monitor_save_dir: str
    tb_save_dir: str
    eval_save_dir: str
    eval_metrics_save_path: str
    animation_save_path: str
    tb_save_dir: str

    @computed_field
    @property
    def model_save_dir(self) -> str:
        return os.path.dirname(self.model_save_path)
