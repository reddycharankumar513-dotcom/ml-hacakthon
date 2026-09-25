"""
Multi-strategy blocking and candidate retrieval engine.
Combines 11 independent blocking strategies to achieve high blocking recall.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

from .utils import setup_logging, Timer

logger = setup_logging()

class MultiStrategyBlocker:
    """
    Implements 11 independent candidate generation blocks and builds a unified candidate pool.
    """
    def __init__(self, max_candidates_per_entity: int = 100, tfidf_top_k: int = 30):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.tfidf_top_k = tfidf_top_k

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_target: pd.DataFrame, # Combined S2 and S3 dataframe
        source_label: str = "S2_S3"
    ) -> pd.DataFrame:
        """
        Generate candidate pairs between df_s1 and df_target.
        Returns a DataFrame of candidate pairs with provenance columns:
        [source1_entity_id, candidate_entity_id, blocker_count, blocker_names]
        """
        logger.info(f"Generating candidates between {len(df_s1)} S1 entities and {len(df_target)} target entities...")
        
        # Dict mapping (s1_id, candidate_id) -> set of blocker names
        pair_provenance = defaultdict(set)

        # ----------------------------------------------------
        # Block 1: Exact Normalized Name
        # ----------------------------------------------------
        with Timer("Block 1: Exact Normalized Name"):
            target_map = defaultdict(list)
            for cid, key in zip(df_target["entity_id"], df_target["name_normalized"]):
                if key:
                    target_map[key].append(cid)
                    
            for s1_id, key in zip(df_s1["entity_id"], df_s1["name_normalized"]):
                if key and key in target_map:
                    for cid in target_map[key]:
                        pair_provenance[(s1_id, cid)].add("exact_name")

        # ----------------------------------------------------
        # Block 2: Name without Legal Suffix
        # ----------------------------------------------------
        with Timer("Block 2: Name without Legal Suffix"):
            target_map = defaultdict(list)
            for cid, key in zip(df_target["entity_id"], df_target["name_without_legal_suffix"]):
                if key and len(key) >= 3:
                    target_map[key].append(cid)
                    
            for s1_id, key in zip(df_s1["entity_id"], df_s1["name_without_legal_suffix"]):
                if key and len(key) >= 3 and key in target_map:
                    for cid in target_map[key][:50]: # cap exact key list
                        pair_provenance[(s1_id, cid)].add("suffix_stripped_name")

        # ----------------------------------------------------
        # Block 3: Name Prefix + Country
        # ----------------------------------------------------
        with Timer("Block 3: Name Prefix + Country"):
            target_map = defaultdict(list)
            for cid, prefix, country in zip(df_target["entity_id"], df_target["name_prefix"], df_target["country_normalized"]):
                if prefix and len(prefix) >= 4:
                    key = f"{country}_{prefix}"
                    target_map[key].append(cid)
                    
            for s1_id, prefix, country in zip(df_s1["entity_id"], df_s1["name_prefix"], df_s1["country_normalized"]):
                if prefix and len(prefix) >= 4:
                    key = f"{country}_{prefix}"
                    if key in target_map:
                        for cid in target_map[key][:30]:
                            pair_provenance[(s1_id, cid)].add("name_prefix")

        # ----------------------------------------------------
        # Block 4: Character N-Gram TF-IDF Retrieval
        # ----------------------------------------------------
        with Timer("Block 4: Character N-Gram TF-IDF"):
            target_text = df_target["name_without_legal_suffix"].tolist()
            s1_text = df_s1["name_without_legal_suffix"].tolist()
            
            if target_text and s1_text:
                effective_min_df = 1 if len(target_text) < 2 else 2
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 4),
                    min_df=effective_min_df,
                    max_features=40000,
                    dtype=np.float32
                )
                target_matrix = vectorizer.fit_transform(target_text)
                s1_matrix = vectorizer.transform(s1_text)
                
                # Multiply S1 x Target.T
                sim_matrix = s1_matrix.dot(target_matrix.T)
                
                s1_ids = df_s1["entity_id"].values
                target_ids = df_target["entity_id"].values
                
                # Get top K candidates per S1 row
                for i in range(sim_matrix.shape[0]):
                    row = sim_matrix.getrow(i)
                    if row.nnz > 0:
                        top_indices = row.indices[np.argsort(row.data)[-self.tfidf_top_k:]]
                        s1_id = s1_ids[i]
                        for idx in top_indices:
                            pair_provenance[(s1_id, target_ids[idx])].add("tfidf_char_ngram")

        # ----------------------------------------------------
        # Block 5: Address Token Overlap + Country
        # ----------------------------------------------------
        with Timer("Block 5: Address Token Overlap"):
            target_map = defaultdict(list)
            for cid, sorted_toks, country in zip(df_target["entity_id"], df_target["address_sorted_tokens"], df_target["country_normalized"]):
                if sorted_toks and len(sorted_toks) >= 5:
                    # Pick first two sorted address tokens + country
                    toks = sorted_toks.split()[:2]
                    key = f"{country}_{'_'.join(toks)}"
                    target_map[key].append(cid)
                    
            for s1_id, sorted_toks, country in zip(df_s1["entity_id"], df_s1["address_sorted_tokens"], df_s1["country_normalized"]):
                if sorted_toks and len(sorted_toks) >= 5:
                    toks = sorted_toks.split()[:2]
                    key = f"{country}_{'_'.join(toks)}"
                    if key in target_map:
                        for cid in target_map[key][:20]:
                            pair_provenance[(s1_id, cid)].add("address_token_overlap")

        # ----------------------------------------------------
        # Block 6: Postal / PIN Candidate + Country
        # ----------------------------------------------------
        with Timer("Block 6: Postal / PIN Candidate"):
            target_map = defaultdict(list)
            for cid, postals, country in zip(df_target["entity_id"], df_target["address_postal_candidates"], df_target["country_normalized"]):
                for p in postals:
                    key = f"{country}_{p}"
                    target_map[key].append(cid)
                    
            for s1_id, postals, country in zip(df_s1["entity_id"], df_s1["address_postal_candidates"], df_s1["country_normalized"]):
                for p in postals:
                    key = f"{country}_{p}"
                    if key in target_map:
                        for cid in target_map[key][:25]:
                            pair_provenance[(s1_id, cid)].add("postal_code")

        # ----------------------------------------------------
        # Block 7: House Number + First Name Token
        # ----------------------------------------------------
        with Timer("Block 7: House Number + First Name Token"):
            target_map = defaultdict(list)
            for cid, hnum, name_toks in zip(df_target["entity_id"], df_target["house_number_candidate"], df_target["name_tokens"]):
                if hnum and isinstance(name_toks, list) and len(name_toks) > 0:
                    first_tok = name_toks[0]
                    if len(first_tok) >= 3:
                        key = f"{hnum}_{first_tok}"
                        target_map[key].append(cid)
                        
            for s1_id, hnum, name_toks in zip(df_s1["entity_id"], df_s1["house_number_candidate"], df_s1["name_tokens"]):
                if hnum and isinstance(name_toks, list) and len(name_toks) > 0:
                    first_tok = name_toks[0]
                    if len(first_tok) >= 3:
                        key = f"{hnum}_{first_tok}"
                        if key in target_map:
                            for cid in target_map[key][:25]:
                                pair_provenance[(s1_id, cid)].add("house_num_name_tok")

        # ----------------------------------------------------
        # Block 8: Country + First Significant Name Token
        # ----------------------------------------------------
        with Timer("Block 8: Country + Name Token"):
            target_map = defaultdict(list)
            for cid, country, name_toks in zip(df_target["entity_id"], df_target["country_normalized"], df_target["name_tokens"]):
                if isinstance(name_toks, list):
                    for tok in name_toks:
                        if len(tok) >= 5 and tok not in {"private", "limited", "company", "corporation", "services"}:
                            key = f"{country}_{tok}"
                            target_map[key].append(cid)
                            break
                            
            for s1_id, country, name_toks in zip(df_s1["entity_id"], df_s1["country_normalized"], df_s1["name_tokens"]):
                if isinstance(name_toks, list):
                    for tok in name_toks:
                        if len(tok) >= 5 and tok not in {"private", "limited", "company", "corporation", "services"}:
                            key = f"{country}_{tok}"
                            if key in target_map:
                                for cid in target_map[key][:20]:
                                    pair_provenance[(s1_id, cid)].add("country_name_token")
                            break

        # ----------------------------------------------------
        # Block 10: Combined Name + Address Word TF-IDF
        # ----------------------------------------------------
        with Timer("Block 10: Combined Name + Address Word TF-IDF"):
            target_text = (df_target["name_normalized"] + " " + df_target["address_normalized"]).tolist()
            s1_text = (df_s1["name_normalized"] + " " + df_s1["address_normalized"]).tolist()
            
            if target_text and s1_text:
                effective_min_df_word = 1 if len(target_text) < 2 else 2
                vectorizer_word = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=effective_min_df_word,
                    max_features=30000,
                    dtype=np.float32
                )
                target_matrix = vectorizer_word.fit_transform(target_text)
                s1_matrix = vectorizer_word.transform(s1_text)
                
                sim_matrix = s1_matrix.dot(target_matrix.T)
                
                s1_ids = df_s1["entity_id"].values
                target_ids = df_target["entity_id"].values
                
                for i in range(sim_matrix.shape[0]):
                    row = sim_matrix.getrow(i)
                    if row.nnz > 0:
                        top_indices = row.indices[np.argsort(row.data)[-20:]]
                        s1_id = s1_ids[i]
                        for idx in top_indices:
                            pair_provenance[(s1_id, target_ids[idx])].add("tfidf_word_combined")

        # Compile candidate pairs into DataFrame
        logger.info(f"Total candidate pair links generated: {len(pair_provenance):,}")
        
        # Enforce max_candidates_per_entity per S1 ID
        s1_candidates = defaultdict(list)
        for (s1_id, cid), blockers in pair_provenance.items():
            if not cid.startswith("S1-"): # strict check: S2/S3 target only
                s1_candidates[s1_id].append((cid, len(blockers), ",".join(sorted(blockers))))

        rows = []
        for s1_id, cand_list in s1_candidates.items():
            # Sort by blocker_count descending
            cand_list.sort(key=lambda x: x[1], reverse=True)
            # Cap top K candidates per S1 entity
            for cid, b_count, b_names in cand_list[:self.max_candidates_per_entity]:
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "blocker_count": b_count,
                    "blocker_names": b_names
                })

        df_candidates = pd.DataFrame(rows)
        if df_candidates.empty:
            df_candidates = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "blocker_count", "blocker_names"])
            
        return df_candidates
