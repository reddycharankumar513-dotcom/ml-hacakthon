#!/usr/bin/env python3
"""
Top-level script for running inference on test set.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.inference import run_test_inference
from business_entity_resolution.utils import ensure_directories

def main():
    config = load_config()
    ensure_directories(config)
    model_path = os.path.join(config["artifacts"]["models_dir"], "matcher_model.joblib")
    thresh = config["threshold"]["default_threshold"]
    run_test_inference(config, model_path, threshold=thresh)

if __name__ == "__main__":
    main()
