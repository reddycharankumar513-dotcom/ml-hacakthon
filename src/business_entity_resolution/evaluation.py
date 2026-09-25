"""
Official Competition Metric Evaluator.
Computes Macro-averaged F0.5 score per Source 1 entity with singleton handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple, Any
from .utils import setup_logging

logger = setup_logging()

def compute_entity_f05(
    pred_matches: Set[str],
    true_matches: Set[str]
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F0.5 score for a single Source 1 entity.
    Handles singletons (true_matches = empty set) and empty predictions.
    """
    n_pred = len(pred_matches)
    n_true = len(true_matches)
    
    # 1. Singleton case: Ground truth has no matches
    if n_true == 0:
        if n_pred == 0:
            return {"precision": 1.0, "recall": 1.0, "f0_5": 1.0, "is_singleton": True, "correct_singleton": True}
        else:
            return {"precision": 0.0, "recall": 0.0, "f0_5": 0.0, "is_singleton": True, "correct_singleton": False}

    # 2. Non-singleton case: Ground truth has >= 1 true match
    if n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f0_5": 0.0, "is_singleton": False, "correct_singleton": False}
        
    correct = len(pred_matches.intersection(true_matches))
    precision = correct / n_pred
    recall = correct / n_true
    
    if (0.25 * precision + recall) > 0:
        f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    else:
        f0_5 = 0.0
        
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0_5": float(f0_5),
        "is_singleton": False,
        "correct_singleton": False
    }

def evaluate_macro_f05(
    predictions_dict: Dict[str, Set[str]],
    ground_truth_dict: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Dict[str, float]:
    """
    Compute macro-averaged precision, recall, F1, F0.5, and singleton accuracy across all S1 entities.
    """
    precisions = []
    recalls = []
    f05_scores = []
    
    singleton_total = 0
    singleton_correct = 0

    for s1_id in all_s1_ids:
        preds = predictions_dict.get(s1_id, set())
        gt = ground_truth_dict.get(s1_id, set())
        
        res = compute_entity_f05(preds, gt)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f05_scores.append(res["f0_5"])
        
        if res["is_singleton"]:
            singleton_total += 1
            if res["correct_singleton"]:
                singleton_correct += 1

    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    
    macro_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)) if (macro_prec + macro_rec) > 0 else 0.0
    singleton_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0

    metrics = {
        "macro_f0_5": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "singleton_accuracy": float(singleton_acc),
        "total_s1_entities": len(all_s1_ids),
        "total_singletons": singleton_total,
        "correct_singletons": singleton_correct
    }
    
    logger.info(f"Evaluation Results -> Macro F0.5: {macro_f05:.4f} | Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f} | Singleton Acc: {singleton_acc:.4f}")
    return metrics
