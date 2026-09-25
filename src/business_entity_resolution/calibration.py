"""
Probability Calibration and Confidence Margin Analysis Engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
from sklearn.calibration import CalibratedClassifierCV
from .utils import setup_logging

logger = setup_logging()

def compute_confidence_margins(
    df_features: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute entity-level confidence metrics per Source 1 record:
    - top_score
    - second_score
    - score_margin (top_score - second_score)
    - number_above_threshold
    """
    df_res = df_features[["source1_entity_id", "candidate_entity_id"]].copy()
    df_res["proba"] = probabilities

    grouped = df_res.groupby("source1_entity_id")
    
    margin_records = {}
    for s1_id, group in grouped:
        probs = group["proba"].sort_values(ascending=False).values
        top = float(probs[0]) if len(probs) > 0 else 0.0
        second = float(probs[1]) if len(probs) > 1 else 0.0
        margin = top - second
        n_above = int((probs >= threshold).sum())
        
        margin_records[s1_id] = {
            "top_score": top,
            "second_score": second,
            "score_margin": margin,
            "number_above_threshold": n_above
        }

    # Map back to df_res
    df_res["top_score"] = df_res["source1_entity_id"].map(lambda s: margin_records[s]["top_score"])
    df_res["second_score"] = df_res["source1_entity_id"].map(lambda s: margin_records[s]["second_score"])
    df_res["score_margin"] = df_res["source1_entity_id"].map(lambda s: margin_records[s]["score_margin"])
    df_res["number_above_threshold"] = df_res["source1_entity_id"].map(lambda s: margin_records[s]["number_above_threshold"])

    return df_res
