#!/usr/bin/env python3
"""
Fast Test Inference Pipeline for Business Entity Resolution.
Replaces brute-force NearestNeighbors with sparse_dot_topn (C++ OpenMP sparse matrix multiplication).
Loads the pre-trained LightGBM model from output/model.txt and executes inference in ~25-35 minutes.
"""

import gc
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from unidecode import unidecode
from rapidfuzz import fuzz, distance
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
import sparse_dot_topn as sdt
from tqdm import tqdm

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
DATA_DIR    = "dataset"
TEST_DIR    = os.path.join(DATA_DIR, "test")
OUTPUT_DIR  = "output"
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model.txt")
THRESHOLD   = 0.980   # Tuned operating threshold from Phase 2 validation

BLOCKING_TOP_K     = 10
TFIDF_NGRAM        = (3, 4)
TFIDF_MIN_DF       = 3
TFIDF_MAX_DF       = 0.05
TFIDF_MAX_FEATURES = 200_000
BATCH_SIZE         = 50_000
N_THREADS          = 16

BUSINESS_SUFFIXES = {
    "pvt": "private", "ltd": "limited", "llc": "llc",
    "inc": "incorporated", "corp": "corporation", "co": "company",
    "assoc": "association", "intl": "international", "natl": "national",
    "mfg": "manufacturing", "svc": "services", "svcs": "services",
    "tech": "technology", "eng": "engineering", "engg": "engineering",
    "grp": "group", "props": "properties", "prop": "properties",
    "cons": "consulting", "consult": "consulting", "mgmt": "management",
    "llp": "llp", "plc": "plc", "sa": "sa", "sarl": "sarl",
    "sas": "sas", "eurl": "eurl", "gmbh": "gmbh",
}

ADDRESS_ABBREVS = {
    "st": "street", "rd": "road", "ave": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "fl": "floor",
    "ste": "suite", "apt": "apartment", "bldg": "building",
    "nr": "near", "no": "number", "dist": "district",
    "rue": "rue", "rte": "route", "rr": "rural route",
}

_SUFFIX_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in BUSINESS_SUFFIXES) + r")\b")
_ADDR_RE   = re.compile(r"\b(" + "|".join(re.escape(k) for k in ADDRESS_ABBREVS) + r")\b")

NAME_STOP = frozenset({
    "the", "of", "and", "in", "for", "a", "an", "to", "at", "by",
    "on", "is", "its", "with", "from", "or", "as", "private",
    "limited", "incorporated", "corporation", "company", "llc",
    "llp", "group", "holdings", "enterprises", "services",
    "solutions", "industries", "international", "plc", "sa",
    "sarl", "sas", "eurl", "gmbh",
})

FEATURE_NAMES = [
    # Name (0–9)
    "name_ratio", "name_partial_ratio", "name_token_sort", "name_token_set",
    "name_jaccard", "name_sig_jaccard", "name_sig_overlap", "name_len_ratio",
    "name_first_word", "name_lev_dist",
    # Address (10–17)
    "addr_ratio", "addr_partial_ratio", "addr_token_sort", "addr_token_set",
    "addr_jaccard", "addr_num_jaccard", "addr_num_overlap", "addr_len_ratio",
    # Country (18)
    "country_match",
    # Combined (19–23)
    "combined_ratio", "combined_token_sort", "combined_jaccard",
    "addr1_missing", "addr2_missing",
]
N_FEATURES = len(FEATURE_NAMES)


def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = unidecode(str(text)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name):
    t = normalize_text(name)
    t = _SUFFIX_RE.sub(lambda m: BUSINESS_SUFFIXES[m.group()], t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_addr(addr):
    t = normalize_text(addr)
    t = _ADDR_RE.sub(lambda m: ADDRESS_ABBREVS[m.group()], t)
    return re.sub(r"\s+", " ", t).strip()


def compute_pair_features_normed(nn1, na1, nc1, nn2, na2, nc2):
    """Compute 24 similarity features on pre-normalized strings."""
    f = np.zeros(N_FEATURES, dtype=np.float32)

    # ── Name ──
    f[0] = fuzz.ratio(nn1, nn2) / 100.0
    f[1] = fuzz.partial_ratio(nn1, nn2) / 100.0
    f[2] = fuzz.token_sort_ratio(nn1, nn2) / 100.0
    f[3] = fuzz.token_set_ratio(nn1, nn2) / 100.0

    t1, t2 = set(nn1.split()), set(nn2.split())
    inter = len(t1 & t2)
    union = len(t1 | t2)
    f[4] = inter / max(union, 1)

    sig1 = {tok for tok in nn1.split() if tok not in NAME_STOP and len(tok) > 1}
    sig2 = {tok for tok in nn2.split() if tok not in NAME_STOP and len(tok) > 1}
    sig_inter = len(sig1 & sig2)
    f[5] = sig_inter / max(len(sig1 | sig2), 1)
    f[6] = sig_inter / max(min(len(sig1), len(sig2)), 1)
    f[7] = min(len(nn1), len(nn2)) / max(len(nn1), len(nn2), 1)

    w1, w2 = nn1.split()[:1], nn2.split()[:1]
    f[8] = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0

    if nn1 and nn2:
        f[9] = 1.0 - Levenshtein.distance(nn1, nn2) / max(len(nn1), len(nn2))

    # ── Address ──
    f[10] = fuzz.ratio(na1, na2) / 100.0
    f[11] = fuzz.partial_ratio(na1, na2) / 100.0
    f[12] = fuzz.token_sort_ratio(na1, na2) / 100.0
    f[13] = fuzz.token_set_ratio(na1, na2) / 100.0

    at1, at2 = set(na1.split()), set(na2.split())
    f[14] = len(at1 & at2) / max(len(at1 | at2), 1)

    nums1 = set(re.findall(r"\d+", na1))
    nums2 = set(re.findall(r"\d+", na2))
    num_inter = len(nums1 & nums2)
    f[15] = num_inter / max(len(nums1 | nums2), 1)
    f[16] = num_inter / max(min(len(nums1), len(nums2)), 1)
    f[17] = min(len(na1), len(na2)) / max(len(na1), len(na2), 1)

    # ── Country ──
    f[18] = 1.0 if nc1 == nc2 else 0.0

    # ── Combined ──
    comb1, comb2 = nn1 + " " + na1, nn2 + " " + na2
    f[19] = fuzz.ratio(comb1, comb2) / 100.0
    f[20] = fuzz.token_sort_ratio(comb1, comb2) / 100.0
    f[21] = len((t1 | at1) & (t2 | at2)) / max(len((t1 | at1) | (t2 | at2)), 1)

    f[22] = 1.0 if (not na1 or na1 == "nan") else 0.0
    f[23] = 1.0 if (not na2 or na2 == "nan") else 0.0

    return f


def compute_features_batch(pairs):
    """Vectorized batch feature extractor."""
    out = np.zeros((len(pairs), N_FEATURES), dtype=np.float32)
    for i, p in enumerate(pairs):
        out[i] = compute_pair_features_normed(*p)
    return out


def main():
    t_start = time.time()
    print("=" * 64)
    print("FAST TEST INFERENCE — BUSINESS ENTITY RESOLUTION")
    print(f"Model path: {MODEL_PATH} | Threshold: {THRESHOLD:.3f}")
    print("=" * 64, flush=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    print("Loading pre-trained LightGBM model …", flush=True)
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"  Model loaded successfully ({model.num_trees()} trees).", flush=True)

    print("\nLoading test datasets …", flush=True)
    t = time.time()
    te_s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t", dtype=str)
    te_s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str)
    te_s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str)
    print(f"  S1={len(te_s1):,}  S2={len(te_s2):,}  S3={len(te_s3):,}  ({time.time()-t:.1f}s)", flush=True)

    s2s3 = pd.concat([te_s2, te_s3], ignore_index=True)
    del te_s2, te_s3
    gc.collect()

    s1_ordered_ids = te_s1["entity_id"].tolist()
    all_matching = {}
    all_candidates = {}

    countries = sorted(te_s1["country"].dropna().unique())
    print(f"\nCountries to process: {countries}", flush=True)

    for country in countries:
        c_start = time.time()
        print("\n" + "-" * 56, flush=True)
        print(f"Processing Country: {country}", flush=True)
        print("-" * 56, flush=True)

        s1_mask = te_s1["country"].values == country
        s2s3_mask = s2s3["country"].values == country

        c_s1_ids = te_s1["entity_id"].values[s1_mask]
        c_s1_names = te_s1["business_name"].fillna("").values[s1_mask]
        c_s1_addrs = te_s1["business_address"].fillna("").values[s1_mask]
        c_s1_countries = te_s1["country"].fillna("").values[s1_mask]

        c_s2s3_ids = s2s3["entity_id"].values[s2s3_mask]
        c_s2s3_names = s2s3["business_name"].fillna("").values[s2s3_mask]
        c_s2s3_addrs = s2s3["business_address"].fillna("").values[s2s3_mask]
        c_s2s3_countries = s2s3["country"].fillna("").values[s2s3_mask]

        n_s1 = len(c_s1_ids)
        n_s2s3 = len(c_s2s3_ids)
        print(f"  {country}: {n_s1:,} S1 entities × {n_s2s3:,} S2+S3 pool", flush=True)

        if n_s1 == 0 or n_s2s3 == 0:
            for sid in c_s1_ids:
                all_matching[sid] = []
                all_candidates[sid] = []
            continue

        # 1. Pre-normalize text and build blocking texts
        print("  Normalizing strings & building blocking text …", flush=True)
        t0 = time.time()
        s1_nn = [normalize_name(x) for x in c_s1_names]
        s1_na = [normalize_addr(x) for x in c_s1_addrs]
        s1_nc = [normalize_text(x) for x in c_s1_countries]
        s1_block = [f"{n} {n} {a}" for n, a in zip(s1_nn, s1_na)]

        s2s3_nn = [normalize_name(x) for x in c_s2s3_names]
        s2s3_na = [normalize_addr(x) for x in c_s2s3_addrs]
        s2s3_nc = [normalize_text(x) for x in c_s2s3_countries]
        s2s3_block = [f"{n} {n} {a}" for n, a in zip(s2s3_nn, s2s3_na)]
        print(f"    Normalized in {time.time()-t0:.1f}s", flush=True)

        # 2. Vectorize S2+S3 with TF-IDF
        print("  Fitting TF-IDF on S2+S3 pool …", flush=True)
        t0 = time.time()
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            dtype=np.float32,
        )
        s2s3_mat = vec.fit_transform(s2s3_block)
        s2s3_mat_t = s2s3_mat.T.tocsr()  # Pre-transpose for fast multiplication
        print(f"    TF-IDF shape: {s2s3_mat.shape} ({time.time()-t0:.1f}s)", flush=True)

        # 3. Batch candidate retrieval and classification
        print(f"  Running sparse blocking & ML classification (batch size {BATCH_SIZE:,}) …", flush=True)
        n_batches = (n_s1 + BATCH_SIZE - 1) // BATCH_SIZE
        country_cands = 0
        country_matches = 0

        pbar = tqdm(range(0, n_s1, BATCH_SIZE), total=n_batches, desc=f"  {country}")
        for bs in pbar:
            be = min(bs + BATCH_SIZE, n_s1)
            b_texts = s1_block[bs:be]

            # Vectorize query batch
            b_mat = vec.transform(b_texts)

            # Fast sparse dot product top-K using OpenMP
            topn_csr = sdt.sp_matmul_topn(
                b_mat,
                s2s3_mat_t,
                top_n=BLOCKING_TOP_K,
                threshold=0.15,
                sort=False,
                n_threads=N_THREADS,
            )

            batch_pairs = []
            offsets = [0]

            for j in range(be - bs):
                sid = c_s1_ids[bs + j]
                cand_idxs = topn_csr.indices[topn_csr.indptr[j] : topn_csr.indptr[j + 1]]

                if len(cand_idxs) == 0:
                    all_candidates[sid] = []
                    all_matching[sid] = []
                    offsets.append(len(batch_pairs))
                    continue

                cand_eids = [c_s2s3_ids[ix] for ix in cand_idxs]
                all_candidates[sid] = cand_eids
                country_cands += len(cand_eids)

                nn1, na1, nc1 = s1_nn[bs + j], s1_na[bs + j], s1_nc[bs + j]
                for ix in cand_idxs:
                    batch_pairs.append((
                        nn1, na1, nc1,
                        s2s3_nn[ix], s2s3_na[ix], s2s3_nc[ix]
                    ))
                offsets.append(len(batch_pairs))

            if batch_pairs:
                feats = compute_features_batch(batch_pairs)
                probs = model.predict(pd.DataFrame(feats, columns=FEATURE_NAMES))

                for j in range(be - bs):
                    sid = c_s1_ids[bs + j]
                    st_idx = offsets[j]
                    en_idx = offsets[j + 1]
                    if st_idx == en_idx:
                        continue
                    cand_eids = all_candidates[sid]
                    p_slice = probs[st_idx:en_idx]
                    matched = [cand_eids[k] for k, p in enumerate(p_slice) if p >= THRESHOLD]
                    all_matching[sid] = matched
                    country_matches += len(matched)

            pbar.set_postfix({
                "cands": f"{country_cands:,}",
                "matches": f"{country_matches:,}",
            })

        print(f"  ✓ {country} completed in {(time.time()-c_start)/60:.1f} min | Cands: {country_cands:,} | Matches: {country_matches:,}", flush=True)

        # Free memory before next country
        del s1_nn, s1_na, s1_nc, s1_block, s2s3_nn, s2s3_na, s2s3_nc, s2s3_block, s2s3_mat, s2s3_mat_t, vec
        gc.collect()

    # ────────────────────────────────────────────────────────────────
    # Write Final Output TSVs
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64, flush=True)
    print("WRITING SUBMISSION TSV FILES", flush=True)
    print("=" * 64, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    match_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cand_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    print(f"Writing {match_file} …", flush=True)
    with open(match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ordered_ids:
            m = all_matching.get(sid, [])
            f.write(f"{sid}\t{','.join(m)}\n")

    print(f"Writing {cand_file} …", flush=True)
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ordered_ids:
            c = all_candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(c)}\n")

    n_with_match = sum(1 for v in all_matching.values() if v)
    n_with_cand = sum(1 for v in all_candidates.values() if v)
    avg_cand = np.mean([len(v) for v in all_candidates.values()])
    avg_match = np.mean([len(v) for v in all_matching.values()])

    print(f"\nFinal Statistics:")
    print(f"  Total S1 entities:  {len(s1_ordered_ids):,}")
    print(f"  With candidates:    {n_with_cand:,} ({n_with_cand/len(s1_ordered_ids)*100:.1f}%)")
    print(f"  With matches:       {n_with_match:,} ({n_with_match/len(s1_ordered_ids)*100:.1f}%)")
    print(f"  Avg cands/entity:   {avg_cand:.2f}")
    print(f"  Avg matches/entity: {avg_match:.2f}")
    print(f"\nTotal elapsed time: {(time.time()-t_start)/60:.1f} min")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
