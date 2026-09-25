"""
Test Set Inference Pipeline.
Runs end-to-end candidate blocking, feature extraction, GBDT prediction, thresholding, and output generation.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Any

from .config import load_config
from .io import load_test_datasets
from .normalization import process_normalization
from .blocking import MultiStrategyBlocker
from .candidate_generation import build_candidate_pairs_tsv
from .features import extract_pair_features
from .model import EntityResolutionMatcher
from .singleton_detector import SingletonDetector
from .graph_refinement import ConservativeGraphRefinement
from .submission import export_matching_results_tsv, run_submission_validator
from .utils import setup_logging, Timer

logger = setup_logging()

def run_test_inference(
    config: Dict[str, Any],
    model_path: str,
    threshold: float = 0.60,
    sample_size: int = None
):
    """Run full test dataset inference pipeline."""
    logger.info("========== RUNNING TEST INFERENCE PIPELINE ==========")
    
    # 1. Load Test Datasets
    with Timer("Loading Test Datasets"):
        test_data = load_test_datasets(config, sample_size=sample_size)
        df_s1 = test_data["s1"]
        df_s2 = test_data["s2"]
        df_s3 = test_data["s3"]
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)

    # 2. Apply Normalization Engine
    with Timer("Test Normalization"):
        df_s1 = process_normalization(df_s1)
        df_target = process_normalization(df_target)

    # 3. Multi-Strategy Blocking & Candidate Pairs
    with Timer("Test Multi-Strategy Blocking"):
        blocker = MultiStrategyBlocker(
            max_candidates_per_entity=config["blocking"]["max_candidates_per_entity"],
            tfidf_top_k=config["blocking"]["top_k_tfidf"]
        )
        df_candidates = blocker.generate_candidates(df_s1, df_target, source_label="Test_Target")

    # Export candidate_pairs.tsv
    cand_path = os.path.join(config["output"]["dir"], config["output"]["candidate_pairs"])
    all_s1_ids = df_s1["entity_id"].tolist()
    build_candidate_pairs_tsv(df_candidates, all_s1_ids, cand_path)

    if df_candidates.empty:
        logger.warning("No candidate pairs generated during blocking!")
        matched_dict = {s1_id: set() for s1_id in all_s1_ids}
    else:
        # 4. Pair Feature Extraction
        with Timer("Test Feature Extraction"):
            df_features = extract_pair_features(df_candidates, df_s1, df_target)

        # 5. Model Inference
        with Timer("Test Model Inference"):
            matcher = EntityResolutionMatcher()
            matcher.load(model_path)
            probas = matcher.predict_proba(df_features)
            df_features["proba"] = probas

        # 6. Singleton Detection & Match Decision Engine
        with Timer("Test Match Decision Engine"):
            detector = SingletonDetector(
                confidence_threshold=config["singleton"]["confidence_threshold"],
                margin_threshold=config["singleton"]["margin_threshold"]
            )
            matched_dict = detector.filter_singletons(df_features, all_s1_ids, base_threshold=threshold)

        # 7. Conservative Graph Refinement
        if config["graph"]["enabled"]:
            with Timer("Test Conservative Graph Refinement"):
                refiner = ConservativeGraphRefinement(
                    min_edge_probability=config["graph"]["min_edge_probability"],
                    max_component_size=config["graph"]["max_component_size"]
                )
                matched_dict = refiner.refine_matches(matched_dict, df_features)

    # 8. Export matching_results.tsv
    matching_path = os.path.join(config["output"]["dir"], config["output"]["matching_results"])
    export_matching_results_tsv(matched_dict, all_s1_ids, matching_path)

    # 9. Submission Validation
    raw_dir = config["data"]["raw_dir"]
    test_dir = os.path.join(raw_dir, "test") if os.path.exists(os.path.join(raw_dir, "test")) else raw_dir
    run_submission_validator(matching_path, cand_path, test_dir)

    logger.info("========== TEST INFERENCE PIPELINE COMPLETED ==========")
