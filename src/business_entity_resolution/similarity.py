"""
String similarity and edit distance computation functions.
Leverages RapidFuzz for high-speed string metrics.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from typing import Tuple, List, Dict, Any

def compute_string_similarities(str1: str, str2: str) -> Dict[str, float]:
    """Compute comprehensive string similarity suite between two single strings."""
    if not str1 or not str2:
        return {
            "levenshtein_sim": 0.0,
            "jaro_winkler_sim": 0.0,
            "fuzz_ratio": 0.0,
            "fuzz_partial": 0.0,
            "fuzz_wratio": 0.0,
            "token_sort": 0.0,
            "token_set": 0.0,
            "jaccard": 0.0,
            "len_diff": abs(len(str1) - len(str2)),
            "rel_len_diff": abs(len(str1) - len(str2)) / max(len(str1), len(str2), 1)
        }
        
    l1, l2 = len(str1), len(str2)
    max_l = max(l1, l2)
    
    # RapidFuzz scores return 0.0 - 100.0, normalize to 0.0 - 1.0
    ratio = fuzz.ratio(str1, str2) / 100.0
    partial = fuzz.partial_ratio(str1, str2) / 100.0
    wratio = fuzz.WRatio(str1, str2) / 100.0
    tsort = fuzz.token_sort_ratio(str1, str2) / 100.0
    tset = fuzz.token_set_ratio(str1, str2) / 100.0
    
    jw = distance.JaroWinkler.similarity(str1, str2)
    lev_dist = distance.Levenshtein.distance(str1, str2)
    lev_sim = 1.0 - (lev_dist / max_l)
    
    toks1 = set(str1.split())
    toks2 = set(str2.split())
    intersection = len(toks1.intersection(toks2))
    union = len(toks1.union(toks2))
    jaccard = (intersection / union) if union > 0 else 0.0
    
    return {
        "levenshtein_sim": float(lev_sim),
        "jaro_winkler_sim": float(jw),
        "fuzz_ratio": float(ratio),
        "fuzz_partial": float(partial),
        "fuzz_wratio": float(wratio),
        "token_sort": float(tsort),
        "token_set": float(tset),
        "jaccard": float(jaccard),
        "len_diff": float(abs(l1 - l2)),
        "rel_len_diff": float(abs(l1 - l2) / max_l)
    }

def compute_numeric_overlap(nums1: List[str], nums2: List[str]) -> Dict[str, float]:
    """Compute number overlap features between two lists of numeric tokens."""
    if not nums1 or not nums2:
        return {
            "num_count_1": len(nums1),
            "num_count_2": len(nums2),
            "num_exact_match": 0.0,
            "num_shared_count": 0.0,
            "num_jaccard": 0.0,
            "first_num_match": 0.0
        }
        
    s1, s2 = set(nums1), set(nums2)
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    jaccard = (inter / union) if union > 0 else 0.0
    
    first_match = 1.0 if nums1[0] == nums2[0] else 0.0
    exact = 1.0 if nums1 == nums2 else 0.0
    
    return {
        "num_count_1": len(nums1),
        "num_count_2": len(nums2),
        "num_exact_match": float(exact),
        "num_shared_count": float(inter),
        "num_jaccard": float(jaccard),
        "first_num_match": float(first_match)
    }
