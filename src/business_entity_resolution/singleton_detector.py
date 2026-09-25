"""
Singleton Detection Engine.
Explicitly identifies and validates unmatched Source 1 entities to prevent false merges.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict
from .utils import setup_logging

logger = setup_logging()

class SingletonDetector:
    """
    Evaluates candidate predictions per S1 entity and flags true singletons.
    """
    def __init__(self, confidence_threshold: float = 0.35, margin_threshold: float = 0.15):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def filter_singletons(
        self,
        df_candidates_with_probs: pd.DataFrame,
        all_s1_ids: List[str],
        base_threshold: float = 0.60
    ) -> Dict[str, Set[str]]:
        """
        Apply confidence rules to determine matched vs unmatched (singleton) S1 entities.
        Returns dict: s1_id -> set of filtered matched entity IDs.
        """
        matched_dict = defaultdict(set)
        
        # Group candidates by S1 ID
        grouped = df_candidates_with_probs.groupby("source1_entity_id")
        
        singletons_count = 0
        matched_count = 0

        for s1_id in all_s1_ids:
            if s1_id not in grouped.groups:
                singletons_count += 1
                continue
                
            group = grouped.get_group(s1_id)
            # Filter candidates above threshold
            high_conf = group[group["proba"] >= base_threshold]
            
            if high_conf.empty:
                # All candidate probabilities are below base threshold -> Singleton
                singletons_count += 1
            else:
                # Check for country conflict or extreme numeric conflict in high confidence candidates
                accepted = set()
                for row in high_conf.itertuples(index=False):
                    c_conflict = getattr(row, "country_conflict", 0.0)
                    h_conflict = getattr(row, "house_num_conflict", 0.0)
                    proba = getattr(row, "proba", 0.0)
                    
                    # Reject if strong country conflict or house num conflict unless proba is extremely high (>0.90)
                    if (c_conflict == 1.0 or h_conflict == 1.0) and proba < 0.90:
                        continue
                    accepted.add(row.candidate_entity_id)
                    
                if accepted:
                    matched_dict[s1_id] = accepted
                    matched_count += 1
                else:
                    singletons_count += 1

        logger.info(f"Singleton Detector -> Matched S1: {matched_count:,} | Singletons: {singletons_count:,}")
        return matched_dict
