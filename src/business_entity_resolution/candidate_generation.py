"""
Candidate generation manager, candidate recall evaluation, and submission output builder.
"""

import pandas as pd
from typing import Dict, Set, Tuple, List, Any
from .utils import setup_logging

logger = setup_logging()

def evaluate_blocking_recall(
    df_candidates: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Dict[str, Any]:
    """
    Evaluate candidate recall against ground truth matching dict.
    Returns detailed dictionary of recall statistics.
    """
    logger.info("Evaluating blocking recall on training/validation ground truth...")
    
    # Map s1_id -> set of candidate IDs
    cand_dict = {}
    for row in df_candidates.itertuples(index=False):
        s1_id = row.source1_entity_id
        cid = row.candidate_entity_id
        if s1_id not in cand_dict:
            cand_dict[s1_id] = set()
        cand_dict[s1_id].add(cid)

    total_gt_pairs = 0
    recovered_gt_pairs = 0
    candidate_counts = []

    for s1_id in all_s1_ids:
        true_matches = gt_dict.get(s1_id, set())
        cands = cand_dict.get(s1_id, set())
        
        candidate_counts.append(len(cands))
        total_gt_pairs += len(true_matches)
        recovered_gt_pairs += len(true_matches.intersection(cands))

    recall = (recovered_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 1.0
    avg_cands = float(pd.Series(candidate_counts).mean()) if candidate_counts else 0.0
    max_cands = int(pd.Series(candidate_counts).max()) if candidate_counts else 0

    stats = {
        "candidate_recall": float(recall),
        "total_true_matches": total_gt_pairs,
        "recovered_true_matches": recovered_gt_pairs,
        "missed_true_matches": total_gt_pairs - recovered_gt_pairs,
        "average_candidates_per_s1": avg_cands,
        "max_candidates_per_s1": max_cands,
        "total_candidate_pairs": int(sum(candidate_counts))
    }

    logger.info(f"Blocking Recall: {recall:.4f} ({recovered_gt_pairs}/{total_gt_pairs}) | Avg Cands/S1: {avg_cands:.2f}")
    return stats

def build_candidate_pairs_tsv(
    df_candidates: pd.DataFrame,
    all_s1_ids: List[str],
    output_path: str
):
    """
    Export candidate_pairs.tsv according to competition specifications:
    - Columns: source1_entity_id, candidate_entity_ids (comma-separated)
    - Exactly one row per test S1 entity ID.
    - Tab separated.
    """
    logger.info(f"Writing candidate_pairs.tsv to {output_path}...")
    
    cand_dict = {}
    for row in df_candidates.itertuples(index=False):
        s1_id = row.source1_entity_id
        cid = row.candidate_entity_id
        if s1_id not in cand_dict:
            cand_dict[s1_id] = []
        if cid not in cand_dict[s1_id]: # preserve order & avoid duplicates
            cand_dict[s1_id].append(cid)

    rows = []
    for s1_id in all_s1_ids:
        cands = cand_dict.get(s1_id, [])
        cand_str = ",".join(cands)
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    logger.info(f"Exported {len(df_out)} rows to {output_path}")
