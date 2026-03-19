from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml


class ExperimentLogger:
    def __init__(self, experiment_name: str, config: dict | None = None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path("../runs") / f"{experiment_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        if config is not None:
            self.log_config(config)

    def log_config(self, config: dict):
        config_path = self.log_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def log_metrics(self, metrics: dict, step: int):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_image(self, tag, img_tensor, step):
        self.writer.add_image(tag, img_tensor, step)

    def close(self):
        self.writer.close()

