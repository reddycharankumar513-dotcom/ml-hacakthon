"""
Unit tests for submission formatting and exporter validation.
"""

import os
import pandas as pd
import pytest
from business_entity_resolution.submission import export_matching_results_tsv
from business_entity_resolution.candidate_generation import build_candidate_pairs_tsv

def test_export_formatting(tmp_path):
    all_s1 = ["S1-00001", "S1-00002"]
    matched_dict = {"S1-00001": {"S2-00047", "S3-00812"}, "S1-00002": set()}
    
    out_m = tmp_path / "matching_results.tsv"
    export_matching_results_tsv(matched_dict, all_s1, str(out_m))
    
    assert out_m.exists()
    df_m = pd.read_csv(out_m, sep="\t", dtype=str, keep_default_na=False)
    assert list(df_m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(df_m) == 2
    assert df_m.iloc[0]["matched_entity_ids"] == "S2-00047,S3-00812"
    assert df_m.iloc[1]["matched_entity_ids"] == ""
