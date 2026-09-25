#!/usr/bin/env python3
"""Script to train GBDT matcher model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.training import run_training_pipeline
from business_entity_resolution.utils import ensure_directories

def main():
    config = load_config()
    ensure_directories(config)
    run_training_pipeline(config)

if __name__ == "__main__":
    main()
