"""
Unit tests for multi-strategy blocking.
"""

import pandas as pd
import pytest
from business_entity_resolution.normalization import process_normalization
from business_entity_resolution.blocking import MultiStrategyBlocker

def test_blocking_candidate_recovery():
    df_s1 = pd.DataFrame([{
        "entity_id": "S1-00001",
        "business_name": "Google LLC",
        "business_address": "1600 Amphitheatre Pkwy, Mountain View",
        "country": "US"
    }])
    
    df_target = pd.DataFrame([{
        "entity_id": "S2-00047",
        "business_name": "Google Incorporated",
        "business_address": "1600 Amphitheatre Parkway, Mountain View",
        "country": "US"
    }])

    df_s1 = process_normalization(df_s1)
    df_target = process_normalization(df_target)

    blocker = MultiStrategyBlocker(max_candidates_per_entity=50, tfidf_top_k=20)
    cands = blocker.generate_candidates(df_s1, df_target)

    assert not cands.empty
    assert "S2-00047" in cands["candidate_entity_id"].values
