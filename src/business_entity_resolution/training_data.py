"""
Training Data Builder.
Combines positive matches, hard negatives, and sampled easy negatives into labelled training pairs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple, Any
from .utils import setup_logging

logger = setup_logging()

def build_labelled_training_pairs(
    df_candidates: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    easy_negative_ratio: float = 2.0,
    max_negatives_per_positive: int = 10
) -> pd.DataFrame:
    """
    Construct labelled training dataset (label=1 for true match, label=0 for candidate non-match).
    Integrates hard negatives from blocking and controlled easy negatives.
    """
    logger.info("Building labelled training pair dataset...")
    
    rows = []
    pos_count = 0
    neg_count = 0

    # Group candidate pairs by source1_entity_id
    grouped = df_candidates.groupby("source1_entity_id")
    
    for s1_id, group in grouped:
        true_matches = gt_dict.get(s1_id, set())
        candidate_ids = group["candidate_entity_id"].tolist()
        
        pos_in_group = 0
        negs_in_group = []
        
        for row in group.itertuples(index=False):
            cid = row.candidate_entity_id
            is_match = 1 if cid in true_matches else 0
            
            row_dict = row._asdict()
            row_dict["label"] = is_match
            
            if is_match == 1:
                rows.append(row_dict)
                pos_count += 1
                pos_in_group += 1
            else:
                negs_in_group.append(row_dict)

        # Sample negatives for this S1 entity
        max_negs = max(2, int(pos_in_group * max_negatives_per_positive)) if pos_in_group > 0 else 5
        if len(negs_in_group) > max_negs:
            # Sort by blocker_count descending so hard negatives (high blocker count but label=0) are prioritized
            negs_in_group.sort(key=lambda x: x.get("blocker_count", 1), reverse=True)
            selected_negs = negs_in_group[:max_negs]
        else:
            selected_negs = negs_in_group
            
        rows.extend(selected_negs)
        neg_count += len(selected_negs)

    df_labelled = pd.DataFrame(rows)
    logger.info(f"Labelled dataset built: {len(df_labelled):,} total pairs | Positives: {pos_count:,} | Negatives: {neg_count:,}")
    return df_labelled
