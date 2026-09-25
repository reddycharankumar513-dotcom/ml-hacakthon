# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity Engineering  
**Team Members:** Lead ML Engineer  
**Submission Date:** September 2026

---

## 1. Executive Summary
We built a competition-grade, scalable Business Entity Resolution pipeline targeting the precision-heavy macro $F_{0.5}$ metric. The architecture integrates multi-view text normalization, an 11-strategy candidate generation blocking engine, a 57-feature pair feature extractor, a LightGBM GBDT binary matcher with hard negative mining, an explicit singleton detector, and conservative graph refinement.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during data profiling and exploration:
- **Precision-Heavy Target ($F_{0.5}$)**: False merges are penalized twice as severely as missed matches.
- **Noise Patterns**: Significant legal suffix variations (Pvt/Private/Ltd/Limited), address abbreviations (St/Rd/Ave), municipal PIN code omissions, and landmark references.
- **Open-Set Country Categorical**: Unseen test country (`France`) requires open-set categorical handling rather than hard-coded rules.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Strategy Blocking + GBDT Binary Matcher + Singleton Detector + Conservative Graph Refinement.
- **Core Innovation**: Fine-grained F0.5 grid search threshold optimization combined with explicit singleton rejection rules and hard negative mining.

---

## 3. Candidate Generation (Blocking)

- **Blocking Strategies Used**:
  1. Exact Normalized Name
  2. Suffix-Stripped Name
  3. Name Prefix + Country
  4. Character 3-4 Gram TF-IDF Cosine Retrieval
  5. Address Token Overlap + Country
  6. Postal/PIN Candidate Code + Country
  7. House/Building Number + First Name Token
  8. Country + Name Token
  9. Combined Name & Address Word TF-IDF
  10. Provenance tracking (`blocker_count`, `blocker_names`).
- **Candidate Pairs Generated**: Top 100 candidate pairs per Source 1 entity.

---

## 4. Matching Model

**Features Used (57 total)**:
- **Name Features**: Levenshtein similarity, Jaro-Winkler, RapidFuzz Ratio, Partial Ratio, WRatio, Token Sort, Token Set, Jaccard, length diffs.
- **Address Features**: Token overlap, edit distances, house number agreement/conflict, PIN code matching.
- **Country Features**: `same_country`, `country_missing`, `country_conflict`.
- **Structural & Cross-Field**: `cross_name_addr_prod`, `cross_name_addr_min`, `cross_name_addr_max`, `strong_name_and_strong_addr`, `strong_name_weak_addr`, `weak_name_strong_addr`.
- **Blocking Provenance**: `blocker_count`, individual blocker pass flags.

**Model Type**: LightGBM Binary Classifier (`lgb.LGBMClassifier`) trained with balanced class weights on Grouped S1 train/val splits.  
**Threshold Selection Method**: Fine-grained grid search optimization (0.10 to 0.95) maximizing validation Macro $F_{0.5}$.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score**: Validated on grouped held-out S1 validation split.
- **Common False Positives**: Entities sharing generic names or multi-tenant building numbers. Controlled via house number conflict flags and high confidence margin thresholds.
- **Common False Negatives**: Heavily abbreviated names with misspelled addresses.

---

## 6. Conclusion
The developed solution delivers a robust, highly scalable entity resolution pipeline adhering strictly to all competition rules, license requirements, and formatting constraints.

---

## Appendix

### A. Code Artefacts
Complete runnable code is located in `c:\Users\chara\Desktop\ml hackathon\prottoype\`:
- Entry point: `python run_pipeline.py --mode all`
- Outputs generated: `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- Submissions validated via `python validate_submission.py`.
