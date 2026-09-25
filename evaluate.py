#!/usr/bin/env python3
"""
Top-level script for evaluating matching results on ground truth datasets.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_ground_truth, parse_ground_truth_matches
from business_entity_resolution.evaluation import evaluate_macro_f05
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted matches against ground truth")
    parser.add_argument("--predictions", required=True, help="Path to predictions TSV")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth TSV")
    args = parser.parse_args()

    df_gt = load_ground_truth(args.ground_truth)
    gt_dict = parse_ground_truth_matches(df_gt)
    all_s1 = list(gt_dict.keys())

    df_pred = pd.read_csv(args.predictions, sep="\t", dtype=str, keep_default_na=False)
    pred_dict = {}
    for row in df_pred.itertuples(index=False):
        m = row.matched_entity_ids.strip()
        pred_dict[row.source1_entity_id] = set(m.split(",")) if m else set()

    metrics = evaluate_macro_f05(pred_dict, gt_dict, all_s1)
    print("\n--- Evaluation Summary ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
