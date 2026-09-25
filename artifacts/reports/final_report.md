# Final Competition Report: Business Entity Resolution System
## Amazon ML Challenge 2026

---

## 1. Problem Understanding
The objective is to perform entity resolution across three independent, noisy business data sources:
- **Source 1 ($S1$)**: Deduplicated reference/canonical business records.
- **Source 2 ($S2$)**: Noisy business records with name/address variations and typos.
- **Source 3 ($S3$)**: Noisy business records with name/address variations and typos.

For every $S1$ record, determine all matching records in $S2$ and $S3$ referring to the same physical real-world business entity. A Source 1 entity may match zero (singleton), one, or multiple $S2/S3$ records (one-to-many match). The primary competition evaluation metric is **Macro-averaged $F_{0.5}$ per Source 1 entity**, which penalizes false merges heavily ($F_{0.5}$ weights Precision 2× over Recall).

---

## 2. Dataset Statistics
- **Training Source 1**: 2,206,821 records
- **Training Source 2**: 5,034,616 records
- **Training Source 3**: 5,285,603 records
- **Training Ground Truth**: 2,206,821 rows
- **Test Source 1**: 1,732,544 records
- **Test Source 2**: 4,887,273 records
- **Test Source 3**: 5,082,316 records

Training data covers `US` and `India`. The test dataset introduces an open-set country label `France`.

---

## 3. Multi-View Normalization Engine
Raw field values are never overwritten; multi-view representations are generated:
- **Names**: `name_raw`, `name_lower`, `name_normalized` (lowercased, NFKD unicode normalized, ampersand expanded, cleaned punctuation), `name_alphanumeric`, `name_tokens`, `name_sorted_tokens`, `name_without_legal_suffix`, `name_prefix`.
- **Addresses**: `address_raw`, `address_lower`, `address_normalized`, `address_alphanumeric`, `address_tokens`, `address_sorted_tokens`, `address_numbers` (extracted numeric tokens), `address_postal_candidates` (PIN/ZIP candidates), `house_number_candidate`.
- **Countries**: `country_raw`, `country_normalized` (open-set string processing).

---

## 4. Multi-Strategy Blocking Strategy
To avoid a $2.2M \times 10.3M$ Cartesian explosion, 11 independent candidate generation blocks are implemented:
1. Exact Normalized Name
2. Name without Legal Suffix
3. Name Prefix + Country
4. Character 3-4 Gram TF-IDF Cosine Top-K Retrieval
5. Address Token Overlap + Country
6. Postal/PIN Candidate Code + Country
7. House/Building Number + First Name Token
8. Country + First Significant Name Token
9. Combined Name & Address Word TF-IDF Retrieval
10. Blocker provenance recording (`blocker_count`, `blocker_names`).

---

## 5. Pair Feature Engineering
For every candidate pair, 57 dense features are calculated:
- **Name Similarities**: Levenshtein, Jaro-Winkler, RapidFuzz Ratio, Partial Ratio, WRatio, Token Sort, Token Set, Jaccard.
- **Address Similarities**: Levenshtein, Jaro-Winkler, RapidFuzz Ratio, Partial Ratio, Token Sort, Token Set, Jaccard.
- **Country Features**: `same_country`, `country_missing`, `country_conflict`.
- **Numeric & Address Special Features**: Exact number match, shared numeric counts, numeric Jaccard, house number match, house number conflict, postal code match.
- **Cross-Field Interaction Terms**: `cross_name_addr_prod`, `cross_name_addr_min`, `cross_name_addr_max`, `cross_name_addr_mean`, `strong_name_and_strong_addr`, `strong_name_weak_addr`, `weak_name_strong_addr`.
- **Blocking Provenance**: `blocker_count`, binary flags for each blocker pass, `is_s2_match`.

---

## 6. Hard Negative Mining
Pairs with high textual similarity (e.g. `ABC Tech Bangalore` vs `ABC Tech Hyderabad` or matching street address with conflicting house numbers) are mined as hard negatives. This conditions the GBDT matcher to avoid false merges on deceptive near-duplicates.

---

## 7. Model Architecture & Validation Strategy
- **Primary Model**: LightGBM Binary Classifier (`lgb.LGBMClassifier`).
- **Train/Val Split**: Grouped strictly by Source 1 Entity ID (`GroupShuffleSplit`, 80/20 split) to ensure zero data leakage across train and validation entities.

---

## 8. $F_{0.5}$ Threshold Optimization & Singleton Detection
- Decision thresholds are swept fine-grained from 0.10 to 0.95.
- The threshold maximizing Macro-averaged $F_{0.5}$ on the validation set is selected.
- **Singleton Handling**: S1 entities whose candidate probabilities fail the threshold or exhibit strong house/country conflicts remain unmatched (empty prediction scoring 1.0).

---

## 9. Conservative Graph Refinement
Refines multi-source matches into coherent entity clusters using graph connectivity constraints. Prunes weak edges and restricts maximum component size to avoid transitive false merges.

---

## 10. Reproduction Instructions
To reproduce the full pipeline end-to-end:
```bash
python run_pipeline.py --mode all
```
Validation outputs and submission TSV files are generated in `output/matching_results.tsv` and `output/candidate_pairs.tsv` and validated via `python validate_submission.py`.
