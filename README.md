# Competition-Grade Business Entity Resolution System
## Amazon ML Challenge 2026

An end-to-end, competition-grade machine learning system for resolving noisy business entities across three independent data sources (Source 1 reference canonical records vs. Source 2/3 noisy records). Built to maximize the macro-averaged $F_{0.5}$ metric while strictly controlling false merges.

---

## 1. Project Overview & Key Principles

- **Precision-Heavy Target**: Evaluated on Macro $F_{0.5}$ per Source 1 entity. $F_{0.5}$ weights Precision 2× over Recall, penalizing false merges heavily.
- **One-to-Many Match Architecture**: Source 1 entities can match 0, 1, or multiple records from Source 2/3.
- **No External Data Lookup**: 100% compliant with competition constraints—zero geocoding, external APIs, or web registries.
- **Open-Set Country Categorical**: Treats `country` as an open-set categorical feature, handling unseen countries (such as France in test set).
- **Hard Negative Mining & Singleton Detection**: Mines challenging non-match entity pairs with high textual overlap and explicitly detects unmatched singletons.

---

## 2. Directory Structure

```
business_entity_resolution/
├── README.md
├── requirements.txt
├── config.yaml
├── run_pipeline.py
├── train.py
├── predict.py
├── evaluate.py
├── validate_submission.py
├── src/
│   └── business_entity_resolution/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── profiling.py
│       ├── normalization.py
│       ├── blocking.py
│       ├── candidate_generation.py
│       ├── similarity.py
│       ├── features.py
│       ├── training_data.py
│       ├── hard_negative_mining.py
│       ├── model.py
│       ├── calibration.py
│       ├── threshold_optimizer.py
│       ├── singleton_detector.py
│       ├── graph_refinement.py
│       ├── inference.py
│       ├── evaluation.py
│       ├── submission.py
│       └── utils.py
├── scripts/
│   ├── profile_data.py
│   ├── build_training_pairs.py
│   ├── train_model.py
│   ├── run_validation.py
│   ├── generate_submission.py
│   └── validate_outputs.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_blocking_analysis.ipynb
│   ├── 03_feature_analysis.ipynb
│   └── 04_error_analysis.ipynb
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── reports/
│   └── cache/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
└── tests/
    ├── test_normalization.py
    ├── test_blocking.py
    ├── test_features.py
    ├── test_evaluation.py
    └── test_submission.py
```

---

## 3. Installation & Setup

1. **Clone/Place Code Repository**:
   Ensure Python 3.8+ is installed.

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Dataset Path**:
   Update `config.yaml` with raw data paths or place `.tsv` files in `dataset/` or current workspace root.

---

## 4. Pipeline Execution Commands

### Execute Full End-to-End Pipeline
Runs data profiling, model training, grouped validation, F0.5 threshold optimization, test inference, output generation, and official submission validation:
```bash
python run_pipeline.py --mode all
```

### Run Pipeline Stages Individually
- **Data Profiling**:
  ```bash
  python run_pipeline.py --mode profile
  ```
- **Model Training & Validation**:
  ```bash
  python run_pipeline.py --mode train
  ```
- **Test Inference & Submission Output**:
  ```bash
  python run_pipeline.py --mode submit
  ```
- **Run Unit Tests**:
  ```bash
  python -m pytest tests/
  ```

---

## 5. Methodology Architecture

1. **Multi-View Normalization Engine**:
   Preserves raw values while engineering lowercased, basic normalized, alphanumeric, legal-suffix stripped, expanded abbreviation, and numeric-token views.

2. **Multi-Strategy Blocking Engine**:
   Combines 11 independent blocking rules:
   - Exact Normalized Name
   - Suffix-Stripped Name
   - Name Prefix + Country
   - Character N-Gram TF-IDF Cosine Retrieval
   - Address Token Overlap
   - Postal/PIN Candidate Code
   - House/Building Number + First Name Token
   - Country + Name Token
   - Combined Name & Address Word TF-IDF
   - Provenance tracking (`blocker_count`, `blocker_names`)

3. **Rich Pair Feature Engineering**:
   Engineers 57+ features per candidate pair spanning RapidFuzz fuzzy ratios, Jaro-Winkler, Levenshtein, token Jaccard, address number overlap, country conflicts, structural length differences, cross-field interaction terms, and blocker flags.

4. **Grouped Validation & Hard Negative Mining**:
   Splits train/val sets grouped strictly by Source 1 Entity ID (`GroupShuffleSplit`) to prevent data leakage. Mines hard negative pairs with high similarity but distinct identities.

5. **GBDT Matcher & Macro F0.5 Threshold Optimizer**:
   Trains LightGBM/XGBoost binary classifier. Sweeps threshold range to maximize Macro $F_{0.5}$ metric per Source 1 entity.

6. **Singleton Detection & Conservative Graph Refinement**:
   Explicitly isolates unmatched S1 entities to protect precision and refines multi-source matches via connected graph component pruning.
