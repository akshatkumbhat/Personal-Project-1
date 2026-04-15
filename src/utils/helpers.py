import os
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
