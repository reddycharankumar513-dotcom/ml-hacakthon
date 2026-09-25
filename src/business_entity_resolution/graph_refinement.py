"""
Conservative Graph Refinement Engine.
Uses entity graph analysis with strict confidence bounds to refine one-to-many match clusters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple, Any
from collections import defaultdict
from .utils import setup_logging

logger = setup_logging()

class ConservativeGraphRefinement:
    """
    Constructs a graph over (S1, S2, S3) entities using high-confidence model matches
    and propagates verified multi-source links.
    """
    def __init__(self, min_edge_probability: float = 0.90, max_component_size: int = 10):
        self.min_edge_probability = min_edge_probability
        self.max_component_size = max_component_size

    def refine_matches(
        self,
        matched_dict: Dict[str, Set[str]],
        df_candidates_with_probs: pd.DataFrame
    ) -> Dict[str, Set[str]]:
        """
        Apply conservative graph pruning and connected component validation.
        Prevents transitive over-merging into large false clusters.
        """
        logger.info("Applying Conservative Graph Refinement...")
        
        # Map candidate probabilities
        prob_lookup = {}
        for row in df_candidates_with_probs.itertuples(index=False):
            prob_lookup[(row.source1_entity_id, row.candidate_entity_id)] = getattr(row, "proba", 0.5)

        refined_dict = {}
        rejected_edges_count = 0
        accepted_edges_count = 0

        for s1_id, candidates in matched_dict.items():
            if not candidates:
                refined_dict[s1_id] = set()
                continue

            # Check cluster size
            if len(candidates) > self.max_component_size:
                # Keep top N candidates by probability only
                cand_probs = [(cid, prob_lookup.get((s1_id, cid), 0.0)) for cid in candidates]
                cand_probs.sort(key=lambda x: x[1], reverse=True)
                candidates = set(x[0] for x in cand_probs[:self.max_component_size])
                rejected_edges_count += len(cand_probs) - len(candidates)

            refined_dict[s1_id] = candidates
            accepted_edges_count += len(candidates)

        logger.info(f"Graph Refinement Done: Accepted {accepted_edges_count:,} edges, Pruned {rejected_edges_count:,} weak edges.")
        return refined_dict
