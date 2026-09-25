"""
Submission Output Generator.
Builds and validates matching_results.tsv and candidate_pairs.tsv according to exact competition rules.
"""

import os
import subprocess
import sys
import pandas as pd
from typing import Dict, Set, List, Any
from .utils import setup_logging

logger = setup_logging()

def export_matching_results_tsv(
    matched_dict: Dict[str, Set[str]],
    all_s1_ids: List[str],
    output_path: str
):
    """
    Export matching_results.tsv according to exact competition format:
    - Columns: source1_entity_id, matched_entity_ids (comma-separated)
    - Exactly one row per test S1 entity ID.
    - Tab-separated (\t).
    """
    logger.info(f"Writing matching_results.tsv to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    matched_count = 0
    singleton_count = 0

    for s1_id in all_s1_ids:
        matches = sorted(list(matched_dict.get(s1_id, set())))
        if matches:
            match_str = ",".join(matches)
            matched_count += 1
        else:
            match_str = ""
            singleton_count += 1
            
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": match_str
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    
    logger.info(f"Exported {len(df_out):,} rows to {output_path} | Matched S1: {matched_count:,} | Singletons: {singleton_count:,}")

def run_submission_validator(
    matching_path: str,
    candidate_path: str,
    test_dir: str
) -> bool:
    """
    Run official competition validate_submission.py script.
    """
    official_script = r"C:\Users\chara\Desktop\ml hackathon\validate_submission.py"
    if not os.path.exists(official_script):
        official_script = os.path.join(test_dir, "validate_submission.py")

    if not os.path.exists(official_script):
        logger.warning(f"Official validator script not found at {official_script}. Skipping subprocess call.")
        return True

    cmd = [
        sys.executable,
        official_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    
    logger.info(f"Executing official validator: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        return res.returncode == 0
    except Exception as e:
        logger.error(f"Error running submission validator: {e}")
        return False
