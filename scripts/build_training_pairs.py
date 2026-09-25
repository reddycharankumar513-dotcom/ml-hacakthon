#!/usr/bin/env python3
"""Script to build and save candidate training pairs and features."""

import os
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_all_datasets, parse_ground_truth_matches
from business_entity_resolution.normalization import process_normalization
from business_entity_resolution.blocking import MultiStrategyBlocker
from business_entity_resolution.features import extract_pair_features
from business_entity_resolution.training_data import build_labelled_training_pairs
from business_entity_resolution.utils import ensure_directories

def main():
    config = load_config()
    ensure_directories(config)
    
    datasets, df_gt = load_all_datasets(config)
    gt_dict = parse_ground_truth_matches(df_gt)
    
    df_s1 = process_normalization(datasets["s1"])
    df_target = process_normalization(pd.concat([datasets["s2"], datasets["s3"]], ignore_index=True))
    
    blocker = MultiStrategyBlocker(max_candidates_per_entity=config["blocking"]["max_candidates_per_entity"])
    df_cands = blocker.generate_candidates(df_s1, df_target)
    
    df_feats = extract_pair_features(df_cands, df_s1, df_target)
    df_labelled = build_labelled_training_pairs(df_feats, gt_dict)
    
    out_path = os.path.join(config["artifacts"]["cache_dir"], "training_pairs_cache.parquet")
    df_labelled.to_parquet(out_path, index=False)
    print(f"Saved training pairs cache ({len(df_labelled)} rows) to {out_path}")

if __name__ == "__main__":
    main()
