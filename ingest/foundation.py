import yaml
import logging
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

class PipelineFoundation:
    def __init__(self, config_path: str = "metadata/ingest_config.yaml"):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.root_dir / config_path
        self.config = self._load_config()
        self.run_id = self._generate_run_id()
        self.logger = None

    def _load_config(self) -> Dict:
        """Loads and validates configuration YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"run_{timestamp}_{suffix}"

    def setup_directories(self) -> None:
        """Creates necessary directory structure idempotent."""
        paths = self.config.get("paths", {})
        
        dirs_to_create = [
            self.root_dir / paths.get("raw_data", "data/raw"),
            self.root_dir / paths.get("processed_assets", "data/processed/assets"),
            self.root_dir / paths.get("processed_artifacts", "data/processed/artifacts"),
            self.root_dir / paths.get("logs", "data/logs"),
        ]

        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
            (d / ".gitkeep").touch()

    def setup_logging(self) -> logging.Logger:
        """Configures file and console logging."""
        log_dir = self.root_dir / self.config["paths"].get("logs", "data/logs")
        log_file = log_dir / f"{self.run_id}.log"

        logger = logging.getLogger(f"GridVision_{self.run_id}")

        logger.setLevel(logging.INFO)
        logger.propagate = False 

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(module)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.info(f"Pipeline initialized. RunID: {self.run_id}")
        logger.info(f"Directories verified at {self.root_dir}")
        logger.info(f"Config loaded from {self.config_path}")
        
        self.logger = logger
        
        return logger

if __name__ == "__main__":
    try:
        foundation = PipelineFoundation()
        foundation.setup_directories()
        logger = foundation.setup_logging()
        
        logger.info("Foundation stage complete.")
    except Exception as e:
        print(f"Critical Setup Error: {e}")