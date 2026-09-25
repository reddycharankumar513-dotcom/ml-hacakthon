"""
Unit tests for pair feature extraction.
"""

import pandas as pd
import pytest
from business_entity_resolution.normalization import process_normalization
from business_entity_resolution.features import extract_pair_features

def test_pair_feature_extraction():
    df_s1 = pd.DataFrame([{
        "entity_id": "S1-00001",
        "business_name": "Tesla Inc",
        "business_address": "1 Tesla Rd, Austin, TX",
        "country": "US"
    }])
    
    df_target = pd.DataFrame([{
        "entity_id": "S2-00010",
        "business_name": "Tesla Motors Incorporated",
        "business_address": "1 Tesla Road, Austin, Texas",
        "country": "US"
    }])

    df_s1 = process_normalization(df_s1)
    df_target = process_normalization(df_target)

    candidates = pd.DataFrame([{
        "source1_entity_id": "S1-00001",
        "candidate_entity_id": "S2-00010",
        "blocker_count": 2,
        "blocker_names": "exact_name,address_token_overlap"
    }])

    feats = extract_pair_features(candidates, df_s1, df_target)

    assert len(feats) == 1
    assert feats["name_fuzz_wratio"].iloc[0] > 0.70
    assert feats["same_country"].iloc[0] == 1.0
    assert feats["house_num_match"].iloc[0] == 1.0
