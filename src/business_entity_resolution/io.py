"""
Data I/O module for loading and parsing TSV datasets.
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Set
from .config import get_data_file_path
from .utils import setup_logging

logger = setup_logging()

REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

def load_source_tsv(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load a single source TSV dataset with strict tab separation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source TSV file not found: {file_path}")
        
    logger.info(f"Loading TSV: {file_path}")
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string"
        },
        keep_default_na=False,
        nrows=sample_size
    )
    
    # Ensure expected columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {file_path}")
            
    # Clean string null representations
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    
    return df

def load_ground_truth(file_path: str, valid_s1_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """Load train ground truth TSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
    logger.info(f"Loading Ground Truth: {file_path}")
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
        keep_default_na=False
    )
    
    for col in GROUND_TRUTH_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing ground truth column '{col}' in {file_path}")
            
    df["source1_entity_id"] = df["source1_entity_id"].astype(str)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str)
    
    if valid_s1_ids is not None:
        df = df[df["source1_entity_id"].isin(valid_s1_ids)].copy()
        
    return df

def parse_ground_truth_matches(df_gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth DataFrame into mapping: s1_id -> set of matched s2/s3 ids."""
    gt_dict = {}
    for row in df_gt.itertuples(index=False):
        s1_id = row.source1_entity_id
        raw_matches = row.matched_entity_ids.strip()
        if raw_matches:
            matches = set(m.strip() for m in raw_matches.split(",") if m.strip())
        else:
            matches = set()
        gt_dict[s1_id] = matches
    return gt_dict

def load_all_datasets(config: Dict[str, Any], sample_size: Optional[int] = None) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Load train or test datasets based on configuration."""
    sample = sample_size or config["data"].get("sample_size")
    
    data_dict = {}
    # Train files
    p1 = get_data_file_path(config, "train_source1", "train")
    p2 = get_data_file_path(config, "train_source2", "train")
    p3 = get_data_file_path(config, "train_source3", "train")
    gt_p = get_data_file_path(config, "ground_truth", "train")
    
    data_dict["s1"] = load_source_tsv(p1, sample_size=sample)
    
    # Filter S2 and S3 if sampling S1 to keep dataset balanced, or load all
    data_dict["s2"] = load_source_tsv(p2, sample_size=sample * 3 if sample else None)
    data_dict["s3"] = load_source_tsv(p3, sample_size=sample * 3 if sample else None)
    
    gt_df = load_ground_truth(gt_p, valid_s1_ids=set(data_dict["s1"]["entity_id"]))
    
    return data_dict, gt_df

def load_test_datasets(config: Dict[str, Any], sample_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load test datasets."""
    sample = sample_size or config["data"].get("sample_size")
    
    data_dict = {}
    p1 = get_data_file_path(config, "test_source1", "test")
    p2 = get_data_file_path(config, "test_source2", "test")
    p3 = get_data_file_path(config, "test_source3", "test")
    
    data_dict["s1"] = load_source_tsv(p1, sample_size=sample)
    data_dict["s2"] = load_source_tsv(p2, sample_size=sample * 3 if sample else None)
    data_dict["s3"] = load_source_tsv(p3, sample_size=sample * 3 if sample else None)
    
    return data_dict
