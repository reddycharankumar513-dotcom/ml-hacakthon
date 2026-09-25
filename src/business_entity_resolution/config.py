"""
Configuration loader and helper routines.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG_PATH = Path("config.yaml")

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        # Fallback if executing from a subdirectory
        alt_path = Path(__file__).resolve().parents[2] / "config.yaml"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at {path} or {alt_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

def get_data_file_path(config: Dict[str, Any], file_key: str, split: str = "train") -> str:
    """Resolve raw data path across flat and nested directory structures."""
    raw_dir = Path(config["data"]["raw_dir"])
    alt_raw_dir = Path(config["data"]["alt_raw_dir"])
    file_name = config["data"][file_key]
    
    candidates = [
        raw_dir / file_name,
        raw_dir / split / file_name,
        raw_dir / "dataset" / split / file_name,
        raw_dir / "student_resource" / "dataset" / split / file_name,
        raw_dir / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / split / file_name,
        alt_raw_dir / split / file_name,
        alt_raw_dir / file_name,
        Path("..") / file_name
    ]
    
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
            
    return str((raw_dir / file_name).resolve())
