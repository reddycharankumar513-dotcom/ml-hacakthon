"""
Pair Feature Engineering Engine.
Generates comprehensive name, address, country, structural, cross-field, and blocking features for candidate pairs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from .similarity import compute_string_similarities, compute_numeric_overlap
from .utils import setup_logging, Timer

logger = setup_logging()

def build_entity_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build fast in-memory dictionary lookup for entity features."""
    lookup = {}
    for row in df.itertuples(index=False):
        eid = row.entity_id
        lookup[eid] = {
            "name_raw": getattr(row, "name_raw", ""),
            "name_normalized": getattr(row, "name_normalized", ""),
            "name_alphanumeric": getattr(row, "name_alphanumeric", ""),
            "name_without_suffix": getattr(row, "name_without_legal_suffix", ""),
            "address_normalized": getattr(row, "address_normalized", ""),
            "address_alphanumeric": getattr(row, "address_alphanumeric", ""),
            "country_normalized": getattr(row, "country_normalized", ""),
            "address_numbers": getattr(row, "address_numbers", []),
            "postal_candidates": getattr(row, "address_postal_candidates", []),
            "house_number": getattr(row, "house_number_candidate", "")
        }
    return lookup

def extract_pair_features(
    df_candidates: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_target: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract full feature set for each candidate pair in df_candidates.
    Returns DataFrame containing candidate identifiers + all numeric features.
    """
    logger.info(f"Extracting pair features for {len(df_candidates):,} candidate pairs...")
    
    with Timer("Building Entity Lookups"):
        s1_lookup = build_entity_lookup(df_s1)
        target_lookup = build_entity_lookup(df_target)

    feature_rows = []
    
    for row in df_candidates.itertuples(index=False):
        s1_id = row.source1_entity_id
        cid = row.candidate_entity_id
        blocker_count = getattr(row, "blocker_count", 1)
        blocker_names = getattr(row, "blocker_names", "")
        
        e1 = s1_lookup.get(s1_id)
        e2 = target_lookup.get(cid)
        
        if not e1 or not e2:
            continue
            
        feats = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cid,
            "blocker_count": blocker_count,
            "is_s2_match": 1.0 if cid.startswith("S2-") else 0.0,
            
            # Blocker flag provenance
            "block_exact_name": 1.0 if "exact_name" in blocker_names else 0.0,
            "block_suffix_stripped": 1.0 if "suffix_stripped_name" in blocker_names else 0.0,
            "block_name_prefix": 1.0 if "name_prefix" in blocker_names else 0.0,
            "block_tfidf_char": 1.0 if "tfidf_char_ngram" in blocker_names else 0.0,
            "block_address_token": 1.0 if "address_token_overlap" in blocker_names else 0.0,
            "block_postal": 1.0 if "postal_code" in blocker_names else 0.0,
            "block_house_num": 1.0 if "house_num_name_tok" in blocker_names else 0.0,
            "block_country_name": 1.0 if "country_name_token" in blocker_names else 0.0,
            "block_tfidf_word": 1.0 if "tfidf_word_combined" in blocker_names else 0.0,
        }
        
        # 1. Exact Boolean Flags
        feats["exact_name_match"] = 1.0 if e1["name_normalized"] and e1["name_normalized"] == e2["name_normalized"] else 0.0
        feats["exact_alphanumeric_name"] = 1.0 if e1["name_alphanumeric"] and e1["name_alphanumeric"] == e2["name_alphanumeric"] else 0.0
        feats["exact_suffix_stripped_name"] = 1.0 if e1["name_without_suffix"] and e1["name_without_suffix"] == e2["name_without_suffix"] else 0.0
        feats["exact_address_match"] = 1.0 if e1["address_normalized"] and e1["address_normalized"] == e2["address_normalized"] else 0.0
        feats["exact_alphanumeric_address"] = 1.0 if e1["address_alphanumeric"] and e1["address_alphanumeric"] == e2["address_alphanumeric"] else 0.0

        # 2. Name Similarities
        name_sims = compute_string_similarities(e1["name_normalized"], e2["name_normalized"])
        for k, v in name_sims.items():
            feats[f"name_{k}"] = v
            
        # 3. Address Similarities
        addr_sims = compute_string_similarities(e1["address_normalized"], e2["address_normalized"])
        for k, v in addr_sims.items():
            feats[f"addr_{k}"] = v

        # 4. Country Features (Open-Set handling)
        c1, c2 = e1["country_normalized"], e2["country_normalized"]
        feats["same_country"] = 1.0 if c1 and c1 == c2 else 0.0
        feats["country_missing"] = 1.0 if not c1 or not c2 else 0.0
        feats["country_conflict"] = 1.0 if c1 and c2 and c1 != c2 else 0.0

        # 5. Numeric & Address Special Features
        num_sims = compute_numeric_overlap(e1["address_numbers"], e2["address_numbers"])
        for k, v in num_sims.items():
            feats[k] = v
            
        p1, p2 = e1["postal_candidates"], e2["postal_candidates"]
        feats["postal_match"] = 1.0 if p1 and p2 and set(p1).intersection(set(p2)) else 0.0
        
        h1, h2 = e1["house_number"], e2["house_number"]
        feats["house_num_match"] = 1.0 if h1 and h2 and h1 == h2 else 0.0
        feats["house_num_conflict"] = 1.0 if h1 and h2 and h1 != h2 else 0.0

        # 6. Structural Features
        n_toks1 = len(e1["name_normalized"].split())
        n_toks2 = len(e2["name_normalized"].split())
        feats["name_tok_count_diff"] = float(abs(n_toks1 - n_toks2))
        
        a_toks1 = len(e1["address_normalized"].split())
        a_toks2 = len(e2["address_normalized"].split())
        feats["addr_tok_count_diff"] = float(abs(a_toks1 - a_toks2))

        # 7. Cross-Field Features
        name_sim = feats["name_fuzz_wratio"]
        addr_sim = feats["addr_fuzz_wratio"]
        
        feats["cross_name_addr_prod"] = name_sim * addr_sim
        feats["cross_name_addr_min"] = min(name_sim, addr_sim)
        feats["cross_name_addr_max"] = max(name_sim, addr_sim)
        feats["cross_name_addr_mean"] = (name_sim + addr_sim) / 2.0
        
        feats["strong_name_and_strong_addr"] = 1.0 if (name_sim > 0.80 and addr_sim > 0.80) else 0.0
        feats["strong_name_weak_addr"] = 1.0 if (name_sim > 0.85 and addr_sim < 0.40) else 0.0
        feats["weak_name_strong_addr"] = 1.0 if (name_sim < 0.40 and addr_sim > 0.85) else 0.0

        feature_rows.append(feats)

    df_feats = pd.DataFrame(feature_rows)
    logger.info(f"Engineered {len(df_feats.columns) - 2} features for {len(df_feats):,} pairs.")
    return df_feats
