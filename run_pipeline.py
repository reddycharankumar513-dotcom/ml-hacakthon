#!/usr/bin/env python3
"""
Main Competition Pipeline CLI Runner for Business Entity Resolution.
Usage:
  python run_pipeline.py --mode all --sample-size 50000
"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add src/ directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from business_entity_resolution.config import load_config
from business_entity_resolution.io import load_all_datasets, load_test_datasets
from business_entity_resolution.profiling import generate_full_data_profile
from business_entity_resolution.training import run_training_pipeline
from business_entity_resolution.inference import run_test_inference
from business_entity_resolution.error_analysis import run_error_analysis
from business_entity_resolution.ablation import run_ablation_experiments
from business_entity_resolution.submission import run_submission_validator
from business_entity_resolution.utils import setup_logging, ensure_directories

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Competition-Grade Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode",
        choices=["profile", "train", "validate", "test", "submit", "all"],
        default="all",
        help="Pipeline stage to execute (default: %(default)s)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size constraint for fast dev/testing"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    
    sample = args.sample_size or config["data"].get("sample_size")

    val_summary = {}
    test_stats = {}

    # 1. Profile Mode
    if args.mode in ["profile", "all"]:
        logger.info("Executing Data Profiling Stage...")
        datasets, _ = load_all_datasets(config, sample_size=sample)
        json_rep = os.path.join(config["artifacts"]["reports_dir"], "data_profile.json")
        md_rep = os.path.join(config["artifacts"]["reports_dir"], "data_profile.md")
        generate_full_data_profile(datasets, json_rep, md_rep)

    # 2. Train & Validate Mode
    if args.mode in ["train", "validate", "all"]:
        logger.info("Executing Model Training & Validation Stage...")
        val_summary = run_training_pipeline(config, sample_size=sample)
        
        # Diagnostics
        val_summary_path = os.path.join(config["artifacts"]["metrics_dir"], "validation_summary.json")
        if os.path.exists(val_summary_path):
            with open(val_summary_path, "r", encoding="utf-8") as f:
                val_summary = json.load(f)
                
        run_ablation_experiments(val_summary, os.path.join(config["artifacts"]["reports_dir"], "ablation.csv"))

    # 3. Test & Submit Mode
    if args.mode in ["test", "submit", "all"]:
        logger.info("Executing Test Inference & Submission Generation...")
        model_path = os.path.join(config["artifacts"]["models_dir"], "matcher_model.joblib")
        
        if not os.path.exists(model_path):
            logger.info("Model checkpoint not found. Training model first...")
            val_summary = run_training_pipeline(config, sample_size=sample)

        thresh = val_summary.get("optimal_threshold", config["threshold"]["default_threshold"])
        run_test_inference(config, model_path, threshold=thresh, sample_size=sample)

    # 4. Final Quality Gate & Completion Report
    if args.mode in ["submit", "all"]:
        matching_path = os.path.join(config["output"]["dir"], config["output"]["matching_results"])
        candidate_path = os.path.join(config["output"]["dir"], config["output"]["candidate_pairs"])
        
        raw_dir = config["data"]["raw_dir"]
        test_dir = os.path.join(raw_dir, "test") if os.path.exists(os.path.join(raw_dir, "test")) else raw_dir
        
        pass_val = run_submission_validator(matching_path, candidate_path, test_dir)
        val_status = "PASS" if pass_val else "FAIL"

        # Count test outputs
        n_test_s1 = 0
        n_matched = 0
        n_singletons = 0
        
        if os.path.exists(matching_path):
            df_m = pd.read_csv(matching_path, sep="\t", keep_default_na=False)
            n_test_s1 = len(df_m)
            n_singletons = int((df_m["matched_entity_ids"].str.strip() == "").sum())
            n_matched = n_test_s1 - n_singletons

        val_f05 = val_summary.get("validation_macro_f0_5", 0.0)
        cand_recall = val_summary.get("candidate_recall", 0.0)
        val_prec = val_summary.get("validation_precision", 0.0)
        val_rec = val_summary.get("validation_recall", 0.0)
        sing_acc = val_summary.get("singleton_accuracy", 0.0)

        print("\n========================================")
        print("BUSINESS ENTITY RESOLUTION COMPLETE")
        print("========================================\n")
        print(f"Validation F0.5: {val_f05:.4f}")
        print(f"Candidate Recall: {cand_recall:.4f}")
        print(f"Validation Precision: {val_prec:.4f}")
        print(f"Validation Recall: {val_rec:.4f}")
        print(f"Singleton Accuracy: {sing_acc:.4f}\n")
        print(f"Test S1 entities: {n_test_s1:,}")
        print(f"Predicted matches: {n_matched:,}")
        print(f"Predicted singletons: {n_singletons:,}\n")
        print(f"Output:")
        print(f"output/matching_results.tsv")
        print(f"output/candidate_pairs.tsv\n")
        print(f"Validation:")
        print(f"{val_status}\n")

if __name__ == "__main__":
    main()
