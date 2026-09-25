"""
Utility functions for logging, memory monitoring, and environment setup.
"""

import os
import sys
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configure structured logger."""
    logger = logging.getLogger("business_entity_resolution")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def set_seed(seed: int = 42):
    """Set random seed across standard libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_directories(config: Dict[str, Any]):
    """Ensure output and artifact directories exist."""
    dirs = [
        config["output"]["dir"],
        config["artifacts"]["models_dir"],
        config["artifacts"]["metrics_dir"],
        config["artifacts"]["reports_dir"],
        config["artifacts"]["cache_dir"]
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

class Timer:
    """Context manager for measuring execution time."""
    def __init__(self, name: str, logger: logging.Logger = None):
        self.name = name
        self.logger = logger or logging.getLogger("business_entity_resolution")
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        self.logger.info(f"Completed {self.name} in {self.elapsed:.2f} seconds.")
