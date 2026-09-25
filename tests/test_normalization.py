"""
Unit tests for multi-view normalization engine.
"""

import pandas as pd
import pytest
from business_entity_resolution.normalization import (
    normalize_text_basic,
    remove_legal_suffixes,
    expand_legal_suffixes,
    expand_address_abbrevs,
    extract_numeric_tokens,
    extract_postal_candidates,
    process_normalization
)

def test_legal_suffix_expansion_and_removal():
    raw_name = "ABC PVT. LTD."
    expanded = expand_legal_suffixes(raw_name)
    assert "PRIVATE" in expanded
    assert "LIMITED" in expanded

    stripped = remove_legal_suffixes(normalize_text_basic(pd.Series([raw_name]))[0])
    assert stripped == "abc"

def test_address_abbrevs():
    raw_addr = "12 MG RD., NEAR SBI ATM"
    expanded = expand_address_abbrevs(raw_addr)
    assert "ROAD" in expanded

def test_numeric_extractions():
    addr = "Flat 402, Building 12, MG Road 560001"
    nums = extract_numeric_tokens(addr)
    postals = extract_postal_candidates(addr)
    assert "402" in nums
    assert "12" in nums
    assert "560001" in postals

def test_process_normalization_dataframe():
    df = pd.DataFrame([{
        "entity_id": "S1-00001",
        "business_name": "Acme Corp. Pvt Ltd",
        "business_address": "100 Main St, Suite 5, 10001",
        "country": "US"
    }])
    df_norm = process_normalization(df)
    assert "name_normalized" in df_norm.columns
    assert "address_normalized" in df_norm.columns
    assert df_norm["country_normalized"].iloc[0] == "us"
