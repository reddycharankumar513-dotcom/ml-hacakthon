"""
Hard Negative Mining Module.
Identifies and tracks challenging non-match entity pairs that exhibit high textual/structural overlap.
"""

import pandas as pd
from typing import Dict, Any
from .utils import setup_logging

logger = setup_logging()

def flag_hard_negatives(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and flag hard negatives in candidate features dataframe.
    A hard negative is a non-match pair (label=0) that satisfies one of:
    1. High name similarity (>0.80) but conflicting house number or country.
    2. High address similarity (>0.85) but different name.
    3. Multiple blocking triggers (>2) but label=0.
    4. Exact postal code match but label=0.
    """
    if "label" not in df_features.columns:
        df_features["is_hard_negative"] = 0
        return df_features

    is_neg = (df_features["label"] == 0)
    
    cond1 = (df_features["name_fuzz_wratio"] > 0.80) & ((df_features["house_num_conflict"] == 1.0) | (df_features["country_conflict"] == 1.0))
    cond2 = (df_features["addr_fuzz_wratio"] > 0.85) & (df_features["name_fuzz_wratio"] < 0.60)
    cond3 = (df_features["blocker_count"] >= 3)
    cond4 = (df_features["postal_match"] == 1.0) & (df_features["name_fuzz_wratio"] > 0.70)
    
    hard_mask = is_neg & (cond1 | cond2 | cond3 | cond4)
    df_features["is_hard_negative"] = hard_mask.astype(int)
    
    total_negs = is_neg.sum()
    hard_count = hard_mask.sum()
    ratio = (hard_count / total_negs) if total_negs > 0 else 0.0
    
    logger.info(f"Hard Negative Mining: Identified {hard_count:,} hard negatives out of {total_negs:,} total negatives ({ratio:.2%}).")
    return df_features

def compute_hard_negative_stats(df_features: pd.DataFrame, preds: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute performance stats specifically on mined hard negatives."""
    if "is_hard_negative" not in df_features.columns:
        return {}
        
    hard_mask = (df_features["is_hard_negative"] == 1)
    if hard_mask.sum() == 0:
        return {"hard_negative_count": 0, "hard_negative_fpr": 0.0}
        
    hard_preds = preds[hard_mask]
    false_positives = (hard_preds >= threshold).sum()
    fpr = float(false_positives / len(hard_preds))
    
    return {
        "hard_negative_count": int(hard_mask.sum()),
        "hard_negative_false_positives": int(false_positives),
        "hard_negative_fpr": float(fpr)
    }
