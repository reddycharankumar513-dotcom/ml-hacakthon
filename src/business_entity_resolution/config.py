"""
Configuration loader and helper routines.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "C:/Users/chara/Desktop/ml hackathon",
        "alt_raw_dir": "dataset",
        "train_source1": "train_source1.tsv",
        "train_source2": "train_source2.tsv",
        "train_source3": "train_source3.tsv",
        "ground_truth": "train_ground_truth.tsv",
        "test_source1": "test_source1.tsv",
        "test_source2": "test_source2.tsv",
        "test_source3": "test_source3.tsv"
    },
    "output": {
        "dir": "output",
        "matching_results": "matching_results.tsv",
        "candidate_pairs": "candidate_pairs.tsv"
    },
    "artifacts": {
        "models_dir": "artifacts/models",
        "metrics_dir": "artifacts/metrics",
        "reports_dir": "artifacts/reports",
        "cache_dir": "artifacts/cache"
    },
    "model": {
        "type": "lightgbm",
        "random_seed": 42,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "class_weight": "balanced",
        "n_jobs": -1
    },
    "blocking": {
        "max_candidates_per_entity": 100,
        "top_k_tfidf": 30
    },
    "training_pairs": {
        "easy_negative_ratio": 2.0,
        "hard_negative_ratio": 3.0,
        "max_negatives_per_positive": 10
    },
    "threshold": {
        "min_val": 0.1,
        "max_val": 0.95,
        "step": 0.02,
        "default_threshold": 0.65
    },
    "singleton": {
        "confidence_threshold": 0.35,
        "margin_threshold": 0.15
    },
    "graph": {
        "enabled": True,
        "min_edge_probability": 0.90,
        "max_component_size": 10
    }
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file with multi-path resolution and fallback."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    search_paths = [
        Path("config.yaml"),
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path(__file__).resolve().parents[1] / "config.yaml",
        Path(os.getcwd()) / "config.yaml",
        Path(os.getcwd()).parent / "config.yaml"
    ]
    
    for p in search_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg and isinstance(cfg, dict):
                        return cfg
            except Exception:
                pass
                
    return DEFAULT_CONFIG.copy()

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
