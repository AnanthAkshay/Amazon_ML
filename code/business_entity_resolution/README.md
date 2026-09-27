# Business Entity Resolution Pipeline

## Overview

Scalable entity resolution pipeline that matches Source 1 business entities
against Source 2 and Source 3 across noisy, multilingual data (US, India, France).

**Architecture**: Preprocessing → Country-partitioned TF-IDF blocking → 24-feature
similarity engineering → LightGBM classifier → F₀.₅-tuned thresholding.

## Quick Start

```bash
# From student_resource/ root:
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py
```

This single command trains on `dataset/train/`, validates on a held-out split,
runs inference on `dataset/test/`, and writes:

- `output/matching_results.tsv` — final matches (leaderboard submission)
- `output/candidate_pairs.tsv` — blocking candidate sets

## Pipeline Stages

### 1. Preprocessing
- Unicode transliteration (`unidecode`) for multilingual handling
- Business suffix expansion: `pvt→private`, `ltd→limited`, `corp→corporation`, etc.
- Address abbreviation expansion: `rd→road`, `st→street`, `blvd→boulevard`, etc.
- `&` → `and`, punctuation removal, lowercase normalisation

### 2. Blocking (Candidate Generation)
- **Country partitioning** — zero cross-country comparisons, open-set safe
- **TF-IDF** over character n-grams (2–4, word-boundary) on `name name address`
- **Cosine k-NN** retrieval (top-20) with distance cutoff < 0.95
- Result: ~15 candidates per S1 entity on average

### 3. Feature Engineering (24 features per pair)
| Group     | Features |
|-----------|----------|
| Name      | ratio, partial_ratio, token_sort, token_set, Jaccard, sig_Jaccard, sig_overlap, len_ratio, first_word_match, Levenshtein |
| Address   | ratio, partial_ratio, token_sort, token_set, Jaccard, num_Jaccard, num_overlap, len_ratio |
| Country   | exact match indicator |
| Combined  | combined_ratio, combined_token_sort, combined_Jaccard, addr_missing flags |

### 4. Matching Model
- **LightGBM** gradient-boosted classifier
- Trained on positive pairs from ground truth + random same-country negatives (3:1 ratio)
- Early stopping on binary logloss

### 5. Threshold Tuning
- F₀.₅ metric weights precision 2× over recall
- Threshold sweep on held-out validation split (0.30–0.95 in 0.025 steps)
- Automatic precision-biased threshold selection

## File Structure

```
code/business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── pipeline.py           # Complete end-to-end pipeline
    ├── validate_pipeline.py  # Standalone validation utility
    └── explore.py            # Data exploration
```

## Validation

```bash
# Official submission validator:
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Requirements
- Python 3.10+
- 16 GB+ RAM recommended for full-scale processing
- Multi-core CPU (parallelised via n_jobs=-1)
