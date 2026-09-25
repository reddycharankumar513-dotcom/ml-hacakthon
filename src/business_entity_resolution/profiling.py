"""
Data profiling engine for dataset inspection and quality reporting.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from .utils import setup_logging

logger = setup_logging()

def profile_single_dataframe(df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
    """Generate detailed stats for a single DataFrame."""
    logger.info(f"Profiling dataset: {source_name}")
    n_rows = len(df)
    n_cols = len(df.columns)
    
    col_profiles = {}
    for col in df.columns:
        series = df[col].astype(str)
        empty_mask = series.str.strip() == ""
        ws_mask = (series.str.len() > 0) & (series.str.strip() == "")
        
        col_profiles[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "empty_string_count": int(empty_mask.sum()),
            "whitespace_only_count": int(ws_mask.sum()),
            "unique_count": int(series.nunique()),
            "duplicate_count": int(n_rows - series.nunique())
        }

    # Distributions
    name_lengths = df["business_name"].astype(str).str.len() if "business_name" in df.columns else pd.Series([], dtype=int)
    addr_lengths = df["business_address"].astype(str).str.len() if "business_address" in df.columns else pd.Series([], dtype=int)
    
    country_counts = df["country"].value_counts().to_dict() if "country" in df.columns else {}
    
    # Repeated normalized names/addresses
    norm_names = df["business_name"].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True) if "business_name" in df.columns else pd.Series([])
    norm_addrs = df["business_address"].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True) if "business_address" in df.columns else pd.Series([])
    
    profile = {
        "source_name": source_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "column_profiles": col_profiles,
        "country_distribution": {str(k): int(v) for k, v in country_counts.items()},
        "name_length_stats": {
            "mean": float(name_lengths.mean()) if len(name_lengths) > 0 else 0.0,
            "min": int(name_lengths.min()) if len(name_lengths) > 0 else 0,
            "max": int(name_lengths.max()) if len(name_lengths) > 0 else 0,
            "p50": float(name_lengths.median()) if len(name_lengths) > 0 else 0.0,
            "p95": float(name_lengths.quantile(0.95)) if len(name_lengths) > 0 else 0.0
        },
        "address_length_stats": {
            "mean": float(addr_lengths.mean()) if len(addr_lengths) > 0 else 0.0,
            "min": int(addr_lengths.min()) if len(addr_lengths) > 0 else 0,
            "max": int(addr_lengths.max()) if len(addr_lengths) > 0 else 0,
            "p50": float(addr_lengths.median()) if len(addr_lengths) > 0 else 0.0,
            "p95": float(addr_lengths.quantile(0.95)) if len(addr_lengths) > 0 else 0.0
        },
        "top_repeated_names": norm_names.value_counts().head(5).to_dict() if len(norm_names) > 0 else {},
        "top_repeated_addresses": norm_addrs.value_counts().head(5).to_dict() if len(norm_addrs) > 0 else {}
    }
    
    return profile

def generate_full_data_profile(datasets: Dict[str, pd.DataFrame], output_json: str, output_md: str) -> Dict[str, Any]:
    """Profile all datasets and export JSON & MD reports."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    
    full_profile = {}
    for name, df in datasets.items():
        full_profile[name] = profile_single_dataframe(df, name)
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_profile, f, indent=2)
        
    logger.info(f"Saved data profile JSON to {output_json}")
    
    # Generate Markdown report
    md_lines = [
        "# Business Entity Resolution — Data Profile Report\n",
        "## Summary Overview\n",
        "| Dataset | Rows | Columns | Unique Entities | Primary Countries |",
        "|---|---|---|---|---|"
    ]
    
    for name, p in full_profile.items():
        countries = ", ".join(list(p["country_distribution"].keys())[:3])
        uniq = p["column_profiles"].get("entity_id", {}).get("unique_count", 0)
        md_lines.append(f"| {name} | {p['n_rows']:,} | {p['n_cols']} | {uniq:,} | {countries} |")
        
    md_lines.append("\n## Detailed Breakdown\n")
    for name, p in full_profile.items():
        md_lines.append(f"### Dataset: {name}\n")
        md_lines.append(f"- **Total Rows:** {p['n_rows']:,}")
        md_lines.append(f"- **Country Distribution:** {p['country_distribution']}")
        md_lines.append(f"- **Name Length Stats (chars):** Mean={p['name_length_stats']['mean']:.1f}, Median={p['name_length_stats']['p50']}, Max={p['name_length_stats']['max']}")
        md_lines.append(f"- **Address Length Stats (chars):** Mean={p['address_length_stats']['mean']:.1f}, Median={p['address_length_stats']['p50']}, Max={p['address_length_stats']['max']}\n")
        
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    logger.info(f"Saved data profile Markdown to {output_md}")
    return full_profile
