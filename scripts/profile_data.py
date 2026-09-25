#!/usr/bin/env python3
"""Script to run data profiling and generate JSON/MD reports."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_all_datasets
from business_entity_resolution.profiling import generate_full_data_profile
from business_entity_resolution.utils import ensure_directories

def main():
    config = load_config()
    ensure_directories(config)
    datasets, _ = load_all_datasets(config)
    json_path = os.path.join(config["artifacts"]["reports_dir"], "data_profile.json")
    md_path = os.path.join(config["artifacts"]["reports_dir"], "data_profile.md")
    generate_full_data_profile(datasets, json_path, md_path)

if __name__ == "__main__":
    main()
