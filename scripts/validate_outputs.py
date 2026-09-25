#!/usr/bin/env python3
"""Script to validate generated output TSVs."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.submission import run_submission_validator

def main():
    config = load_config()
    matching_path = os.path.join(config["output"]["dir"], config["output"]["matching_results"])
    candidate_path = os.path.join(config["output"]["dir"], config["output"]["candidate_pairs"])
    raw_dir = config["data"]["raw_dir"]
    test_dir = os.path.join(raw_dir, "test") if os.path.exists(os.path.join(raw_dir, "test")) else raw_dir
    
    success = run_submission_validator(matching_path, candidate_path, test_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
