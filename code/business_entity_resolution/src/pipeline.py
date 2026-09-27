"""
Business Entity Resolution Pipeline
====================================
End-to-end pipeline: preprocessing → blocking → features → LightGBM → output.

Usage (from student_resource/):
    python code/business_entity_resolution/src/pipeline.py

Produces:
    output/matching_results.tsv   (leaderboard submission)
    output/candidate_pairs.tsv    (blocking audit)
"""

import os
import sys
import re
import gc
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode
from tqdm import tqdm

warnings.filterwarnings("ignore")
# Handle encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ────────────────────────────────────────────────────────────────
# Paths & hyper-parameters
# ────────────────────────────────────────────────────────────────
DATA_DIR   = "dataset"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Blocking
BLOCKING_TOP_K      = 20       # kNN neighbours per S1 entity
TFIDF_NGRAM         = (2, 4)   # char-wb n-gram range
TFIDF_MAX_FEATURES  = 300_000
BATCH_SIZE          = 20_000   # S1 entities per kNN batch

# Training
TRAIN_GT_SAMPLE     = 200_000  # max GT rows to sample for training
MAX_POS_PAIRS       = 400_000  # cap on positive pairs
NEG_RATIO           = 3        # negatives per positive
LGB_ROUNDS          = 500
LGB_EARLY_STOP      = 50

# Validation
VAL_SIZE            = 10_000   # S1 entities held out for threshold tuning

# ────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ────────────────────────────────────────────────────────────────

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
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "fl": "floor",
    "ste": "suite", "apt": "apartment", "bldg": "building",
    "nr": "near", "no": "number", "dist": "district",
    "rue": "rue", "rte": "route", "rr": "rural route",
}

_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BUSINESS_SUFFIXES) + r")\b"
)
_ADDR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ADDRESS_ABBREVS) + r")\b"
)

NAME_STOP = frozenset({
    "the", "of", "and", "in", "for", "a", "an", "to", "at", "by",
    "on", "is", "its", "with", "from", "or", "as", "private",
    "limited", "incorporated", "corporation", "company", "llc",
    "llp", "group", "holdings", "enterprises", "services",
    "solutions", "industries", "international", "plc", "sa",
    "sarl", "sas", "eurl", "gmbh",
})


def normalize_text(text):
    """Basic: transliterate → lower → strip punctuation → collapse spaces."""
    if pd.isna(text) or text is None:
        return ""
    text = unidecode(str(text)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name):
    """Normalize a business name (expand abbreviations)."""
    t = normalize_text(name)
    t = _SUFFIX_RE.sub(lambda m: BUSINESS_SUFFIXES[m.group()], t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_addr(addr):
    """Normalize an address (expand abbreviations)."""
    t = normalize_text(addr)
    t = _ADDR_RE.sub(lambda m: ADDRESS_ABBREVS[m.group()], t)
    return re.sub(r"\s+", " ", t).strip()


def get_name_tokens(normed_name):
    """Significant name tokens after removing stop words."""
    return [tok for tok in normed_name.split()
            if tok not in NAME_STOP and len(tok) > 1]


def create_blocking_text(name, addr, country):
    """Build a single string for TF-IDF blocking.  Name is repeated to weight it 2×."""
    n = normalize_name(name)
    a = normalize_addr(addr)
    return f"{n} {n} {a}"


# ────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (24 features per pair)
# ────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Name (0–9)
    "name_ratio", "name_partial_ratio", "name_token_sort", "name_token_set",
    "name_jaccard", "name_sig_jaccard", "name_sig_overlap", "name_len_ratio",
    "name_first_word", "name_levenshtein",
    # Address (10–17)
    "addr_ratio", "addr_partial_ratio", "addr_token_sort", "addr_token_set",
    "addr_jaccard", "addr_num_jaccard", "addr_num_overlap", "addr_len_ratio",
    # Country (18)
    "country_match",
    # Combined (19–23)
    "combined_ratio", "combined_token_sort", "combined_jaccard",
    "addr1_missing", "addr2_missing",
]
N_FEATURES = len(FEATURE_NAMES)  # 24


def compute_pair_features(n1, a1, c1, n2, a2, c2):
    """Return a float32 array of 24 similarity features for one pair."""
    nn1, nn2 = normalize_name(n1), normalize_name(n2)
    na1, na2 = normalize_addr(a1), normalize_addr(a2)
    nc1, nc2 = normalize_text(c1), normalize_text(c2)

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

    sig1, sig2 = set(get_name_tokens(nn1)), set(get_name_tokens(nn2))
    sig_inter = len(sig1 & sig2)
    f[5] = sig_inter / max(len(sig1 | sig2), 1)
    f[6] = sig_inter / max(min(len(sig1), len(sig2)), 1)
    f[7] = min(len(nn1), len(nn2)) / max(len(nn1), len(nn2), 1)

    w1, w2 = nn1.split(), nn2.split()
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
    """Vectorised wrapper – pairs is a list of (n1,a1,c1,n2,a2,c2) tuples."""
    out = np.zeros((len(pairs), N_FEATURES), dtype=np.float32)
    for i, (n1, a1, c1, n2, a2, c2) in enumerate(pairs):
        out[i] = compute_pair_features(n1, a1, c1, n2, a2, c2)
    return out


# ────────────────────────────────────────────────────────────────
# 3. TRAINING  (positive pairs + random negatives → LightGBM)
# ────────────────────────────────────────────────────────────────

def load_source(path):
    return pd.read_csv(path, sep="\t", dtype={"entity_id": str})


def _index_df(df):
    """DataFrame → dict[entity_id → row-dict]."""
    return {row["entity_id"]: row for _, row in df.iterrows()}


def prepare_training_data(s1, s2, s3, gt, max_pos=MAX_POS_PAIRS, neg_ratio=NEG_RATIO):
    """Build balanced training set of (positive, negative) feature vectors."""
    print("  Indexing sources …")
    s1d = s1.set_index("entity_id").to_dict("index")
    s2d = s2.set_index("entity_id").to_dict("index")
    s3d = s3.set_index("entity_id").to_dict("index")

    # ── Positives ──
    print("  Collecting positive pairs …")
    pos = []
    for _, row in tqdm(gt.iterrows(), total=len(gt), desc="  pos"):
        s1_id = row["source1_entity_id"]
        if pd.isna(row["matched_entity_ids"]):
            continue
        rec1 = s1d.get(s1_id)
        if rec1 is None:
            continue
        for mid in str(row["matched_entity_ids"]).split(","):
            mid = mid.strip()
            rec2 = s2d.get(mid) if mid.startswith("S2") else s3d.get(mid)
            if rec2 is not None:
                pos.append((
                    rec1["business_name"], rec1["business_address"], rec1["country"],
                    rec2["business_name"], rec2["business_address"], rec2["country"],
                ))
            if len(pos) >= max_pos:
                break
        if len(pos) >= max_pos:
            break
    print(f"    → {len(pos)} positive pairs")

    # ── Negatives (random within same country) ──
    print("  Sampling negatives …")
    pool_by_country = defaultdict(list)
    for d in (s2d, s3d):
        for rec in d.values():
            pool_by_country[rec["country"]].append(rec)

    np.random.seed(42)
    n_neg = min(len(pos) * neg_ratio, max_pos * neg_ratio)
    neg = []
    s1_vals = list(s1d.values())
    np.random.shuffle(s1_vals)
    for rec1 in s1_vals:
        if len(neg) >= n_neg:
            break
        p = pool_by_country.get(rec1["country"], [])
        if not p:
            continue
        for _ in range(neg_ratio):
            rec2 = p[np.random.randint(len(p))]
            neg.append((
                rec1["business_name"], rec1["business_address"], rec1["country"],
                rec2["business_name"], rec2["business_address"], rec2["country"],
            ))
            if len(neg) >= n_neg:
                break
    print(f"    → {len(neg)} negative pairs")

    # ── Features ──
    print("  Computing features …")
    all_pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int8)
    idx = np.arange(len(all_pairs))
    np.random.shuffle(idx)
    all_pairs = [all_pairs[i] for i in idx]
    labels = labels[idx]

    chunks = []
    cs = 5_000
    for i in tqdm(range(0, len(all_pairs), cs), desc="  feats"):
        chunks.append(compute_features_batch(all_pairs[i : i + cs]))
    X = np.vstack(chunks)
    print(f"    → X {X.shape}, pos-rate {labels.mean():.3f}")
    return X, labels


def train_lgbm(X, y):
    """Train LightGBM and tune the classification threshold for F_0.5."""
    print("\n── Training LightGBM ──")
    n_val = int(len(X) * 0.1)
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
    X_tr, y_tr   = X[idx[n_val:]], y[idx[n_val:]]

    tr_ds  = lgb.Dataset(pd.DataFrame(X_tr,  columns=FEATURE_NAMES), label=y_tr)
    val_ds = lgb.Dataset(pd.DataFrame(X_val, columns=FEATURE_NAMES), label=y_val,
                         reference=tr_ds)

    params = dict(
        objective="binary", metric=["binary_logloss", "auc"],
        boosting_type="gbdt", num_leaves=127, learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=50, lambda_l1=0.1, lambda_l2=0.1,
        verbose=-1, n_jobs=-1,
        scale_pos_weight=int(sum(y_tr == 0)) / max(int(sum(y_tr == 1)), 1),
    )

    model = lgb.train(
        params, tr_ds, num_boost_round=LGB_ROUNDS,
        valid_sets=[val_ds],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(LGB_EARLY_STOP)],
    )

    # ── Threshold search on pair-level F_0.5 ──
    preds = model.predict(pd.DataFrame(X_val, columns=FEATURE_NAMES))
    best_f05, best_thr = 0.0, 0.5
    for thr in np.arange(0.30, 0.95, 0.025):
        pred_bin = (preds >= thr).astype(int)
        tp = int(((pred_bin == 1) & (y_val == 1)).sum())
        fp = int(((pred_bin == 1) & (y_val == 0)).sum())
        fn = int(((pred_bin == 0) & (y_val == 1)).sum())
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f05 = (1.25 * pr * rc) / max(0.25 * pr + rc, 1e-10)
        if f05 > best_f05:
            best_f05, best_thr = f05, thr
    print(f"  Best pair-level threshold: {best_thr:.3f}  (F_0.5 = {best_f05:.4f})")

    # ── Feature importance ──
    imp = model.feature_importance(importance_type="gain")
    for i in np.argsort(imp)[::-1][:8]:
        print(f"    {FEATURE_NAMES[i]:25s}  {imp[i]:.0f}")

    return model, best_thr


# ────────────────────────────────────────────────────────────────
# 4. HELD-OUT VALIDATION  (macro F_0.5 exactly like the challenge)
# ────────────────────────────────────────────────────────────────

def f05_single(true_set, pred_set):
    """F_0.5 for one S1 entity."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    pr = tp / len(pred_set)
    rc = tp / len(true_set)
    if pr + rc == 0:
        return 0.0
    return (1.25 * pr * rc) / (0.25 * pr + rc)


def run_validation(model, initial_threshold, s1, s2, s3, gt_val):
    """Evaluate on a held-out GT split using the same blocking + inference logic."""
    print("\n── Held-out validation ──")

    # Build ground truth dict
    gt_dict = {}
    all_matched_ids = set()
    for _, row in gt_val.iterrows():
        sid = row["source1_entity_id"]
        if pd.isna(row["matched_entity_ids"]):
            gt_dict[sid] = set()
        else:
            ids = {m.strip() for m in str(row["matched_entity_ids"]).split(",")}
            gt_dict[sid] = ids
            all_matched_ids |= ids

    val_s1_set = set(gt_dict.keys())
    val_s1_df = s1[s1["entity_id"].isin(val_s1_set)].copy()

    # Sample S2/S3 plus all true matches to form realistic evaluation pool
    s2_sample = s2.sample(min(100_000, len(s2)), random_state=42)
    s3_sample = s3.sample(min(100_000, len(s3)), random_state=42)
    s2_matched = s2[s2["entity_id"].isin(all_matched_ids)]
    s3_matched = s3[s3["entity_id"].isin(all_matched_ids)]

    val_s2s3 = pd.concat([s2_sample, s3_sample, s2_matched, s3_matched]).drop_duplicates("entity_id")
    s2s3_set = set(val_s2s3["entity_id"])
    print(f"  Val S1: {len(val_s1_df):,}, Val S2+S3 pool: {len(val_s2s3):,}")

    val_s1_records = val_s1_df.to_dict("records")
    val_s2s3_records = val_s2s3.to_dict("records")

    # Partition by country
    from collections import defaultdict as dd
    s1_by_c   = dd(list)
    s2s3_by_c = dd(list)
    for rec in val_s1_records:
        s1_by_c[rec["country"]].append((rec["entity_id"], rec))
    for rec in val_s2s3_records:
        s2s3_by_c[rec["country"]].append((rec["entity_id"], rec))

    # sid -> list of (cand_eid, probability)
    cand_scores = {sid: [] for sid in gt_dict}
    cand_counts = []

    for country in sorted(s1_by_c.keys()):
        s1_items   = s1_by_c[country]
        s2s3_items = s2s3_by_c.get(country, [])
        if not s2s3_items:
            continue

        s1_texts   = [create_blocking_text(r["business_name"], r["business_address"], r["country"])
                      for _, r in s1_items]
        s2s3_texts = [create_blocking_text(r["business_name"], r["business_address"], r["country"])
                      for _, r in s2s3_items]

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=TFIDF_NGRAM,
                              max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
                              dtype=np.float32)
        s2s3_mat = vec.fit_transform(s2s3_texts)
        k = min(BLOCKING_TOP_K, len(s2s3_items))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(s2s3_mat)

        for bstart in range(0, len(s1_items), BATCH_SIZE):
            bend = min(bstart + BATCH_SIZE, len(s1_items))
            batch_tfidf = vec.transform(s1_texts[bstart:bend])
            dists, idxs = nn.kneighbors(batch_tfidf, n_neighbors=k)

            for j in range(bend - bstart):
                sid, s1_rec = s1_items[bstart + j]
                ci = idxs[j][dists[j] < 0.95]
                if len(ci) == 0:
                    cand_counts.append(0)
                    continue
                cand_counts.append(len(ci))
                pairs = []
                cand_eids = []
                for idx in ci:
                    r2_eid, r2 = s2s3_items[idx]
                    cand_eids.append(r2_eid)
                    pairs.append((
                        s1_rec["business_name"], s1_rec["business_address"], s1_rec["country"],
                        r2["business_name"], r2["business_address"], r2["country"],
                    ))
                feats = compute_features_batch(pairs)
                probs = model.predict(pd.DataFrame(feats, columns=FEATURE_NAMES))
                cand_scores[sid] = list(zip(cand_eids, probs))

        del s2s3_mat, nn, vec
        gc.collect()

    def eval_threshold(thr):
        scores = []
        for sid in gt_dict:
            true_set = {m for m in gt_dict[sid] if m in s2s3_set}
            pred_set = {eid for eid, p in cand_scores.get(sid, []) if p >= thr}
            scores.append(f05_single(true_set, pred_set))
        return np.mean(scores)

    best_thr = initial_threshold
    best_f05 = eval_threshold(best_thr)
    print(f"  Threshold {best_thr:.3f} -> Macro F_0.5: {best_f05:.4f}")

    for bump in [0.02, 0.04, 0.06, 0.08, 0.10]:
        alt_thr = min(best_thr + bump, 0.98)
        alt_f05 = eval_threshold(alt_thr)
        print(f"  Testing threshold {alt_thr:.3f} ... F_0.5: {alt_f05:.4f}", end="")
        if alt_f05 > best_f05:
            print(f"  ↑ improved ({best_f05:.4f} → {alt_f05:.4f})")
            best_f05, best_thr = alt_f05, alt_thr
        else:
            print("  ↓ no improvement")

    # Final breakdown at best threshold
    final_scores = []
    singleton_scores = []
    nonsingleton_scores = []
    for sid in gt_dict:
        true_set = {m for m in gt_dict[sid] if m in s2s3_set}
        pred_set = {eid for eid, p in cand_scores.get(sid, []) if p >= best_thr}
        s = f05_single(true_set, pred_set)
        final_scores.append(s)
        if not gt_dict[sid]:
            singleton_scores.append(s)
        else:
            nonsingleton_scores.append(s)

    print(f"  ★ Final chosen threshold: {best_thr:.3f}, Macro F_0.5: {best_f05:.4f}")
    if singleton_scores:
        print(f"  Singleton F_0.5:     {np.mean(singleton_scores):.4f}  (n={len(singleton_scores)})")
    if nonsingleton_scores:
        print(f"  Non-singleton F_0.5: {np.mean(nonsingleton_scores):.4f}  (n={len(nonsingleton_scores)})")
    if cand_counts:
        print(f"  Avg candidates/ent:  {np.mean(cand_counts):.1f}")

    return best_f05, best_thr


# ────────────────────────────────────────────────────────────────
# 5. BLOCKING + INFERENCE  (country-partitioned, batched)
# ────────────────────────────────────────────────────────────────

def run_inference_country(model, threshold, s1_texts, s2s3_texts,
                          s1_ids, s2s3_ids, s1_data, s2s3_data, country):
    """Block and classify for one country partition.  Returns (matching, candidates)."""
    n_s1, n_s2s3 = len(s1_texts), len(s2s3_texts)
    if n_s2s3 == 0:
        empty = {sid: [] for sid in s1_ids}
        return empty, dict(empty)

    print(f"    {country}: {n_s1:,} S1 × {n_s2s3:,} S2+S3")

    # TF-IDF index
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=TFIDF_NGRAM,
                          max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
                          dtype=np.float32)
    s2s3_mat = vec.fit_transform(s2s3_texts)

    k = min(BLOCKING_TOP_K, n_s2s3)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(s2s3_mat)

    matching, candidates = {}, {}
    total_cands, total_matches = 0, 0

    for bs in range(0, n_s1, BATCH_SIZE):
        be = min(bs + BATCH_SIZE, n_s1)
        batch_mat = vec.transform(s1_texts[bs:be])
        dists, idxs = nn.kneighbors(batch_mat, n_neighbors=k)

        batch_pairs = []
        offsets = [0]

        for j in range(be - bs):
            sid = s1_ids[bs + j]
            s1_rec = s1_data[bs + j]

            ci = idxs[j][dists[j] < 0.95]
            if len(ci) == 0:
                matching[sid] = []
                candidates[sid] = []
                offsets.append(len(batch_pairs))
                continue

            cand_eids = [s2s3_ids[ix] for ix in ci]
            candidates[sid] = cand_eids
            total_cands += len(cand_eids)

            for ix in ci:
                r2 = s2s3_data[ix]
                batch_pairs.append((
                    s1_rec["business_name"], s1_rec["business_address"], s1_rec["country"],
                    r2["business_name"], r2["business_address"], r2["country"],
                ))
            offsets.append(len(batch_pairs))

        if batch_pairs:
            feats = compute_features_batch(batch_pairs)
            probs = model.predict(pd.DataFrame(feats, columns=FEATURE_NAMES))
            for j in range(be - bs):
                sid = s1_ids[bs + j]
                start_idx = offsets[j]
                end_idx = offsets[j + 1]
                if start_idx == end_idx:
                    continue
                cand_eids = candidates[sid]
                ent_probs = probs[start_idx:end_idx]
                matched = [cand_eids[k2] for k2, p in enumerate(ent_probs) if p >= threshold]
                matching[sid] = matched
                total_matches += len(matched)

        pct = be / n_s1 * 100
        print(f"      batch {be:,}/{n_s1:,}  ({pct:.0f}%)")

    del s2s3_mat, nn, vec
    gc.collect()
    print(f"      → cands={total_cands:,}  matches={total_matches:,}")
    return matching, candidates


# ────────────────────────────────────────────────────────────────
# 6. MAIN
# ────────────────────────────────────────────────────────────────

def main():
    T0 = time.time()

    # ════════════════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════════════════
    print("=" * 64)
    print("PHASE 1 — TRAINING")
    print("=" * 64)

    print("\nLoading training data …")
    t = time.time()
    tr_s1 = load_source(os.path.join(TRAIN_DIR, "train_source1.tsv"))
    tr_s2 = load_source(os.path.join(TRAIN_DIR, "train_source2.tsv"))
    tr_s3 = load_source(os.path.join(TRAIN_DIR, "train_source3.tsv"))
    gt    = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")
    print(f"  S1={len(tr_s1):,}  S2={len(tr_s2):,}  S3={len(tr_s3):,}  GT={len(gt):,}"
          f"  ({time.time()-t:.0f}s)")

    # Split GT into train / val
    np.random.seed(42)
    perm = np.random.permutation(len(gt))
    gt_val   = gt.iloc[perm[:VAL_SIZE]].copy()
    gt_train = gt.iloc[perm[VAL_SIZE : VAL_SIZE + TRAIN_GT_SAMPLE]].copy()

    print(f"  GT train split: {len(gt_train):,},  GT val split: {len(gt_val):,}")

    model_path = os.path.join(OUTPUT_DIR, "model.txt")
    if os.path.exists(model_path):
        print(f"\n  Found existing model at {model_path}, loading...")
        model = lgb.Booster(model_file=model_path)
        threshold = 0.900
    else:
        X, y = prepare_training_data(tr_s1, tr_s2, tr_s3, gt_train)
        model, threshold = train_lgbm(X, y)
        del X, y; gc.collect()
        model.save_model(model_path)
        print(f"  Model saved to {model_path}")

    # ════════════════════════════════════════════════════════════
    # VALIDATION
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("PHASE 2 — VALIDATION  (held-out training split)")
    print("=" * 64)

    val_f05, threshold = run_validation(model, threshold, tr_s1, tr_s2, tr_s3, gt_val)
    print(f"\n  ★ Selected operating threshold: {threshold:.3f},  Val F_0.5: {val_f05:.4f}")

    # Free training data
    del tr_s1, tr_s2, tr_s3, gt, gt_train, gt_val
    gc.collect()

    # ════════════════════════════════════════════════════════════
    # TEST INFERENCE
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("PHASE 3 — TEST INFERENCE")
    print("=" * 64)

    print("\nLoading test data …")
    t = time.time()
    te_s1 = load_source(os.path.join(TEST_DIR, "test_source1.tsv"))
    te_s2 = load_source(os.path.join(TEST_DIR, "test_source2.tsv"))
    te_s3 = load_source(os.path.join(TEST_DIR, "test_source3.tsv"))
    print(f"  S1={len(te_s1):,}  S2={len(te_s2):,}  S3={len(te_s3):,}  ({time.time()-t:.0f}s)")

    s2s3 = pd.concat([te_s2, te_s3], ignore_index=True)
    del te_s2, te_s3; gc.collect()

    print("\nPreparing blocking texts …")
    t = time.time()
    s1_block = np.array([
        create_blocking_text(r["business_name"], r["business_address"], r["country"])
        for _, r in tqdm(te_s1.iterrows(), total=len(te_s1), desc="  S1")
    ])
    s2s3_block = np.array([
        create_blocking_text(r["business_name"], r["business_address"], r["country"])
        for _, r in tqdm(s2s3.iterrows(), total=len(s2s3), desc="  S2+S3")
    ])
    print(f"  ({time.time()-t:.0f}s)")

    s1_countries   = te_s1["country"].values
    s2s3_countries = s2s3["country"].values
    s1_ids_all     = te_s1["entity_id"].values
    s2s3_ids_all   = s2s3["entity_id"].values
    s1_data_all    = te_s1.to_dict("records")
    s2s3_data_all  = s2s3.to_dict("records")

    all_matching, all_candidates = {}, {}

    countries = sorted(set(s1_countries))
    print(f"\nCountries: {countries}")

    for country in countries:
        print(f"\n  Processing {country} …")
        s1m   = s1_countries == country
        s2m   = s2s3_countries == country
        m, c  = run_inference_country(
            model, threshold,
            s1_block[s1m], s2s3_block[s2m],
            s1_ids_all[s1m], s2s3_ids_all[s2m],
            [s1_data_all[i] for i in np.where(s1m)[0]],
            [s2s3_data_all[i] for i in np.where(s2m)[0]],
            country,
        )
        all_matching.update(m)
        all_candidates.update(c)

    # ════════════════════════════════════════════════════════════
    # WRITE OUTPUT
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("PHASE 4 — WRITING OUTPUT")
    print("=" * 64)

    ordered = te_s1["entity_id"].tolist()

    with open(os.path.join(OUTPUT_DIR, "matching_results.tsv"), "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered:
            ids = all_matching.get(sid, [])
            fh.write(f"{sid}\t{','.join(ids)}\n")

    with open(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in ordered:
            ids = all_candidates.get(sid, [])
            fh.write(f"{sid}\t{','.join(ids)}\n")

    # ── Summary stats ──
    n_match = sum(1 for v in all_matching.values() if v)
    n_cand  = sum(1 for v in all_candidates.values() if v)
    avg_c   = np.mean([len(v) for v in all_candidates.values()])
    avg_m   = np.mean([len(v) for v in all_matching.values()])
    print(f"  Total S1 entities:    {len(ordered):,}")
    print(f"  With candidates:      {n_cand:,} ({n_cand/len(ordered)*100:.1f}%)")
    print(f"  With matches:         {n_match:,} ({n_match/len(ordered)*100:.1f}%)")
    print(f"  Avg cands/entity:     {avg_c:.1f}")
    print(f"  Avg matches/entity:   {avg_m:.2f}")
    print(f"\n  Total time: {(time.time()-T0)/60:.1f} min")
    print("=" * 64)
    print("DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()
