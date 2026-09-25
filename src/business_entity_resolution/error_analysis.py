"""
Error Analysis & Diagnostic Reporting Engine.
"""

import os
import pandas as pd
from typing import Dict, Set, List, Any
from .utils import setup_logging

logger = setup_logging()

def run_error_analysis(
    df_val_features: pd.DataFrame,
    val_preds_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    output_csv: str
) -> pd.DataFrame:
    """
    Perform deep error analysis on validation predictions and export detailed report CSV.
    """
    logger.info("Running Error Analysis on validation results...")
    
    error_rows = []
    
    # Analyze candidate pairs
    for row in df_val_features.itertuples(index=False):
        s1_id = row.source1_entity_id
        cid = row.candidate_entity_id
        proba = getattr(row, "proba", 0.0)
        
        true_matches = gt_dict.get(s1_id, set())
        pred_matches = val_preds_dict.get(s1_id, set())
        
        is_true_match = 1 if cid in true_matches else 0
        is_pred_match = 1 if cid in pred_matches else 0
        
        error_type = None
        if is_true_match == 0 and is_pred_match == 1:
            error_type = "FALSE_POSITIVE"
        elif is_true_match == 1 and is_pred_match == 0:
            error_type = "FALSE_NEGATIVE"
        elif is_true_match == 1 and is_pred_match == 1 and proba < 0.65:
            error_type = "LOW_CONF_TRUE_MATCH"
        elif is_true_match == 0 and proba >= 0.80:
            error_type = "HIGH_CONF_FALSE_MATCH"

        if error_type:
            error_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "error_type": error_type,
                "model_probability": proba,
                "name_fuzz_wratio": getattr(row, "name_fuzz_wratio", 0.0),
                "addr_fuzz_wratio": getattr(row, "addr_fuzz_wratio", 0.0),
                "same_country": getattr(row, "same_country", 1.0),
                "country_conflict": getattr(row, "country_conflict", 0.0),
                "house_num_conflict": getattr(row, "house_num_conflict", 0.0),
                "blocker_count": getattr(row, "blocker_count", 1),
                "is_true_match": is_true_match,
                "is_predicted_match": is_pred_match
            })

    df_errors = pd.DataFrame(error_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_errors.to_csv(output_csv, index=False)
    
    logger.info(f"Saved {len(df_errors):,} error diagnostic entries to {output_csv}")
    return df_errors
