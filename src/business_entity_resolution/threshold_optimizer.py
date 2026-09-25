"""
F0.5 Threshold Optimization Engine.
Sweeps decision thresholds to maximize macro-averaged F0.5 on validation split.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict
from .evaluation import evaluate_macro_f05
from .utils import setup_logging

logger = setup_logging()

def optimize_f05_threshold(
    df_val_features: pd.DataFrame,
    val_probabilities: np.ndarray,
    gt_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    min_thresh: float = 0.1,
    max_thresh: float = 0.95,
    step: float = 0.02
) -> Dict[str, Any]:
    """
    Grid search decision threshold to maximize macro F0.5 on validation dataset.
    Returns dictionary with best threshold and complete optimization metadata.
    """
    logger.info(f"Optimizing F0.5 threshold over range [{min_thresh:.2f}, {max_thresh:.2f}] with step {step:.2f}...")
    
    thresholds = np.arange(min_thresh, max_thresh + step / 2, step)
    
    df_eval = df_val_features[["source1_entity_id", "candidate_entity_id"]].copy()
    df_eval["proba"] = val_probabilities

    best_threshold = 0.5
    best_macro_f05 = -1.0
    best_metrics = {}
    sweep_history = []

    for t in thresholds:
        t = round(float(t), 4)
        
        # Filter predictions >= threshold
        pred_sub = df_eval[df_eval["proba"] >= t]
        
        pred_dict = defaultdict(set)
        for row in pred_sub.itertuples(index=False):
            pred_dict[row.source1_entity_id].add(row.candidate_entity_id)

        metrics = evaluate_macro_f05(pred_dict, gt_dict, all_s1_ids)
        metrics["threshold"] = t
        sweep_history.append(metrics)

        if metrics["macro_f0_5"] > best_macro_f05:
            best_macro_f05 = metrics["macro_f0_5"]
            best_threshold = t
            best_metrics = metrics

    logger.info(f"Optimal Threshold Selected: {best_threshold:.4f} -> Macro F0.5: {best_macro_f05:.4f}")
    
    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "best_precision": best_metrics.get("macro_precision", 0.0),
        "best_recall": best_metrics.get("macro_recall", 0.0),
        "singleton_accuracy": best_metrics.get("singleton_accuracy", 0.0),
        "sweep_history": sweep_history
    }

def save_threshold_artifact(threshold_data: Dict[str, Any], filepath: str):
    """Save optimized threshold metadata to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Saved threshold metadata to {filepath}")
