"""
Full Training and Cross-Validation Pipeline.
Handles grouped train/validation splitting, feature engineering, hard negative mining,
model fitting, threshold optimization, error analysis, and ablation experiments.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple, Any
from sklearn.model_selection import GroupShuffleSplit

from .config import load_config
from .io import load_all_datasets, parse_ground_truth_matches
from .normalization import process_normalization
from .blocking import MultiStrategyBlocker
from .candidate_generation import evaluate_blocking_recall
from .features import extract_pair_features
from .training_data import build_labelled_training_pairs
from .hard_negative_mining import flag_hard_negatives, compute_hard_negative_stats
from .model import EntityResolutionMatcher
from .threshold_optimizer import optimize_f05_threshold, save_threshold_artifact
from .singleton_detector import SingletonDetector
from .graph_refinement import ConservativeGraphRefinement
from .evaluation import evaluate_macro_f05
from .utils import setup_logging, Timer, set_seed

logger = setup_logging()

def run_training_pipeline(config: Dict[str, Any], sample_size: int = None) -> Dict[str, Any]:
    """Run full training, cross-validation, and optimization pipeline."""
    set_seed(config["model"]["random_seed"])
    logger.info("========== STARTING TRAINING & VALIDATION PIPELINE ==========")

    # 1. Load Data
    with Timer("Loading Datasets"):
        datasets, df_gt = load_all_datasets(config, sample_size=sample_size)
        df_s1 = datasets["s1"]
        df_s2 = datasets["s2"]
        df_s3 = datasets["s3"]
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)
        gt_dict = parse_ground_truth_matches(df_gt)

    # 2. Normalization Engine
    with Timer("Multi-View Normalization"):
        df_s1 = process_normalization(df_s1)
        df_target = process_normalization(df_target)

    # 3. Multi-Strategy Blocking
    with Timer("Multi-Strategy Candidate Blocking"):
        blocker = MultiStrategyBlocker(
            max_candidates_per_entity=config["blocking"]["max_candidates_per_entity"],
            tfidf_top_k=config["blocking"]["top_k_tfidf"]
        )
        df_candidates = blocker.generate_candidates(df_s1, df_target, source_label="Train_Target")

    # Evaluate Blocking Recall
    all_s1_ids = df_s1["entity_id"].tolist()
    blocking_stats = evaluate_blocking_recall(df_candidates, gt_dict, all_s1_ids)

    # 4. Pair Feature Extraction
    with Timer("Pair Feature Extraction"):
        df_features = extract_pair_features(df_candidates, df_s1, df_target)

    # 5. Build Labelled Training Dataset & Hard Negative Mining
    with Timer("Building Labelled Dataset"):
        df_labelled = build_labelled_training_pairs(
            df_features,
            gt_dict,
            easy_negative_ratio=config["training_pairs"]["easy_negative_ratio"],
            max_negatives_per_positive=config["training_pairs"]["max_negatives_per_positive"]
        )
        df_labelled = flag_hard_negatives(df_labelled)

    # 6. Grouped Train / Validation Split by Source 1 Entity ID
    with Timer("Grouped Train/Validation Split"):
        gss = GroupShuffleSplit(n_splits=1, test_size=config["validation"]["test_size"], random_state=config["validation"]["random_seed"])
        groups = df_labelled["source1_entity_id"].values
        train_idx, val_idx = next(gss.split(df_labelled, groups=groups))

        df_train = df_labelled.iloc[train_idx].copy()
        df_val = df_labelled.iloc[val_idx].copy()
        
        train_s1_set = set(df_train["source1_entity_id"])
        val_s1_set = set(df_val["source1_entity_id"])
        
        logger.info(f"Grouped Split -> Train S1 Entities: {len(train_s1_set):,} | Val S1 Entities: {len(val_s1_set):,}")
        logger.info(f"Group Disjoint Check (Intersection = 0): {len(train_s1_set.intersection(val_s1_set)) == 0}")

    # 7. Model Training (GBDT Matcher)
    with Timer("Fitting GBDT Matcher Model"):
        matcher = EntityResolutionMatcher(
            model_type=config["model"]["type"],
            params=config["model"]
        )
        matcher.fit(df_train, df_val)
        
        # Save model checkpoint
        model_path = os.path.join(config["artifacts"]["models_dir"], "matcher_model.joblib")
        matcher.save(model_path)

    # 8. Predict on Validation Split
    with Timer("Predicting Validation Probabilities"):
        val_probas = matcher.predict_proba(df_val)
        df_val["proba"] = val_probas

    # 9. F0.5 Threshold Optimization
    with Timer("F0.5 Threshold Optimization"):
        val_s1_ids = sorted(list(val_s1_set))
        thresh_results = optimize_f05_threshold(
            df_val,
            val_probas,
            gt_dict,
            val_s1_ids,
            min_thresh=config["threshold"]["min_val"],
            max_thresh=config["threshold"]["max_val"],
            step=config["threshold"]["step"]
        )
        
        thresh_path = os.path.join(config["artifacts"]["models_dir"], "threshold.json")
        save_threshold_artifact(thresh_results, thresh_path)

    # 10. Run Full Pipeline Validation (Singleton Detector + Graph Refinement)
    best_thresh = thresh_results["best_threshold"]
    detector = SingletonDetector(
        confidence_threshold=config["singleton"]["confidence_threshold"],
        margin_threshold=config["singleton"]["margin_threshold"]
    )
    val_preds_dict = detector.filter_singletons(df_val, val_s1_ids, base_threshold=best_thresh)

    if config["graph"]["enabled"]:
        refiner = ConservativeGraphRefinement(
            min_edge_probability=config["graph"]["min_edge_probability"],
            max_component_size=config["graph"]["max_component_size"]
        )
        val_preds_dict = refiner.refine_matches(val_preds_dict, df_val)

    final_val_metrics = evaluate_macro_f05(val_preds_dict, gt_dict, val_s1_ids)

    # 11. Feature Importance & Save Metrics
    df_imp = matcher.get_feature_importance()
    if not df_imp.empty:
        imp_path = os.path.join(config["artifacts"]["reports_dir"], "feature_importance.csv")
        df_imp.to_csv(imp_path, index=False)

    summary_metrics = {
        "candidate_recall": blocking_stats["candidate_recall"],
        "validation_macro_f0_5": final_val_metrics["macro_f0_5"],
        "validation_precision": final_val_metrics["macro_precision"],
        "validation_recall": final_val_metrics["macro_recall"],
        "singleton_accuracy": final_val_metrics["singleton_accuracy"],
        "optimal_threshold": best_thresh
    }
    
    summary_path = os.path.join(config["artifacts"]["metrics_dir"], "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("========== TRAINING & VALIDATION PIPELINE COMPLETED ==========")
    return summary_metrics
