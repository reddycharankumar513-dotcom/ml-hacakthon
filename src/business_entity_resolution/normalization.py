"""
Multi-view normalization and number extraction engine.
Preserves raw values while engineering multi-faceted string views.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Set, Dict, Tuple, Any

# Common Legal Suffix Mapping (Raw/Abbreviated -> Normalized Standard)
LEGAL_SUFFIXES_MAP = {
    r"\bPVT\.?\b": "PRIVATE",
    r"\bLTD\.?\b": "LIMITED",
    r"\bCORP\.?\b": "CORPORATION",
    r"\bCO\.?\b": "COMPANY",
    r"\bINC\.?\b": "INCORPORATED",
    r"\bLLC\.?\b": "LIMITED LIABILITY COMPANY",
    r"\bPLC\.?\b": "PUBLIC LIMITED COMPANY",
    r"\bLLP\.?\b": "LIMITED LIABILITY PARTNERSHIP",
    r"\bENT\.?\b": "ENTERPRISES",
    r"\bINT\.?\b": "INTERNATIONAL",
    r"\bMFG\.?\b": "MANUFACTURING",
    r"\bSER\.?\b": "SERVICES",
    r"\bTECH\.?\b": "TECHNOLOGY",
    r"\bSYS\.?\b": "SYSTEMS"
}

# Common Address Abbreviation Mapping
ADDRESS_ABBREV_MAP = {
    r"\bST\.?\b": "STREET",
    r"\bRD\.?\b": "ROAD",
    r"\bAVE\.?\b": "AVENUE",
    r"\bBLVD\.?\b": "BOULEVARD",
    r"\bDR\.?\b": "DRIVE",
    r"\bLN\.?\b": "LANE",
    r"\bCT\.?\b": "COURT",
    r"\bPKWY\.?\b": "PARKWAY",
    r"\bSTE\.?\b": "SUITE",
    r"\bAPT\.?\b": "APARTMENT",
    r"\bFLR?\.?\b": "FLOOR",
    r"\bBLDG\.?\b": "BUILDING",
    r"\bNR\.?\b": "NEAR",
    r"\bOPP\.?\b": "OPPOSITE"
}

LEGAL_SUFFIX_STRIP_REGEX = re.compile(
    r"\b(PRIVATE|LIMITED|CORPORATION|COMPANY|INCORPORATED|LLC|PLC|LLP|PVT|LTD|CORP|CO|INC)\b",
    re.IGNORECASE
)

def unicode_normalize(text: str) -> str:
    """Apply NFKD unicode normalization and remove non-spacing marks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

def normalize_text_basic(series: pd.Series) -> pd.Series:
    """Basic normalization: lowercasing, unicode NFKD, ampersand expansion, punctuation cleanup."""
    s = series.fillna("").astype(str)
    # Unicode NFKD
    s = s.apply(unicode_normalize)
    s = s.str.lower()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def remove_legal_suffixes(text: str) -> str:
    """Strip legal suffixes from normalized text."""
    if not text:
        return ""
    res = LEGAL_SUFFIX_STRIP_REGEX.sub("", text)
    return re.sub(r"\s+", " ", res).strip()

def expand_legal_suffixes(text: str) -> str:
    """Expand legal suffix abbreviations."""
    if not text:
        return ""
    res = text
    for pattern, replacement in LEGAL_SUFFIXES_MAP.items():
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)
    return res

def expand_address_abbrevs(text: str) -> str:
    """Expand address abbreviations."""
    if not text:
        return ""
    res = text
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)
    return res

def extract_numeric_tokens(text: str) -> List[str]:
    """Extract all numeric tokens from string."""
    if not text:
        return []
    return re.findall(r"\b\d+\b", text)

def extract_postal_candidates(text: str) -> List[str]:
    """Extract 5 or 6 digit PIN/ZIP codes."""
    if not text:
        return []
    return re.findall(r"\b\d{5,6}\b", text)

def process_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce multi-view normalized columns for business_name, business_address, and country.
    Returns modified DataFrame with engineered normalized views.
    """
    logger_msg = f"Normalizing dataset with {len(df)} rows..."
    print(logger_msg)
    
    # Raw values preserved
    df["name_raw"] = df["business_name"].fillna("").astype(str)
    df["address_raw"] = df["business_address"].fillna("").astype(str)
    df["country_raw"] = df["country"].fillna("").astype(str)
    
    # 1. Lowercase views
    df["name_lower"] = df["name_raw"].str.lower()
    df["address_lower"] = df["address_raw"].str.lower()
    df["country_normalized"] = df["country_raw"].apply(unicode_normalize).str.lower().str.strip()
    
    # 2. Basic Normalized Views
    df["name_normalized"] = normalize_text_basic(df["name_raw"])
    df["address_normalized"] = normalize_text_basic(df["address_raw"])
    
    # 3. Alphanumeric views
    df["name_alphanumeric"] = df["name_normalized"].str.replace(r"\s+", "", regex=True)
    df["address_alphanumeric"] = df["address_normalized"].str.replace(r"\s+", "", regex=True)
    
    # 4. Suffix expansions & stripping for names
    expanded_names = df["name_lower"].apply(expand_legal_suffixes)
    df["name_expanded"] = normalize_text_basic(expanded_names)
    df["name_without_legal_suffix"] = df["name_normalized"].apply(remove_legal_suffixes)
    
    # 5. Token views & sorted tokens
    df["name_tokens"] = df["name_normalized"].str.split()
    df["name_sorted_tokens"] = df["name_tokens"].apply(lambda tok: " ".join(sorted(tok)) if isinstance(tok, list) else "")
    
    df["address_tokens"] = df["address_normalized"].str.split()
    df["address_sorted_tokens"] = df["address_tokens"].apply(lambda tok: " ".join(sorted(tok)) if isinstance(tok, list) else "")
    
    # Address expanded view
    expanded_addrs = df["address_lower"].apply(expand_address_abbrevs)
    df["address_expanded"] = normalize_text_basic(expanded_addrs)
    
    # 6. Prefixes & Numeric extractions
    df["name_prefix"] = df["name_without_legal_suffix"].str.slice(0, 5)
    
    df["address_numbers"] = df["address_raw"].apply(extract_numeric_tokens)
    df["address_postal_candidates"] = df["address_raw"].apply(extract_postal_candidates)
    df["first_number"] = df["address_numbers"].apply(lambda nums: nums[0] if nums else "")
    df["house_number_candidate"] = df["address_numbers"].apply(lambda nums: nums[0] if nums and len(nums[0]) <= 4 else "")
    
    return df
