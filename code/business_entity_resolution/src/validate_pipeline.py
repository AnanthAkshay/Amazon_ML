"""
Quick validation run on a small sample of training data
to verify the pipeline works end-to-end.
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

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")

# Import pipeline functions
sys.path.insert(0, os.path.join('code', 'business_entity_resolution', 'src'))
from pipeline import (
    normalize_name, normalize_addr, normalize_text,
    create_blocking_text, compute_pair_features, compute_features_batch,
    FEATURE_NAMES, get_name_tokens
)


def f05_score(y_true_lists, y_pred_lists):
    """Compute macro-averaged F_0.5 score as per challenge rules."""
    scores = []
    for true_ids, pred_ids in zip(y_true_lists, y_pred_lists):
        true_set = set(true_ids)
        pred_set = set(pred_ids)
        
        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not true_set and pred_set:
            scores.append(0.0)
            continue
        if true_set and not pred_set:
            scores.append(0.0)
            continue
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if prec + rec == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            scores.append(f05)
    
    return np.mean(scores)


def main():
    t0 = time.time()
    
    # Load training data
    print("Loading training data...")
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep='\t', dtype={'entity_id': str})
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep='\t', dtype={'entity_id': str})
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep='\t', dtype={'entity_id': str})
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep='\t')
    
    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT: {len(gt)}")
    
    # Use a validation sample
    np.random.seed(42)
    VAL_SIZE = 5000      # S1 entities for validation
    TRAIN_SIZE = 50000   # S1 entities for training
    
    # Stratified sample: include singletons
    gt['has_match'] = gt['matched_entity_ids'].notna()
    
    all_indices = np.arange(len(gt))
    np.random.shuffle(all_indices)
    
    val_indices = all_indices[:VAL_SIZE]
    train_indices = all_indices[VAL_SIZE:VAL_SIZE + TRAIN_SIZE]
    
    gt_val = gt.iloc[val_indices]
    gt_train = gt.iloc[train_indices]
    
    # Get the S1 IDs for val
    val_s1_ids = set(gt_val['source1_entity_id'])
    train_s1_ids = set(gt_train['source1_entity_id'])
    
    # Get all matched S2/S3 IDs that appear in training
    train_matched_ids = set()
    for _, row in gt_train.iterrows():
        if pd.notna(row['matched_entity_ids']):
            for mid in str(row['matched_entity_ids']).split(','):
                train_matched_ids.add(mid.strip())
    
    val_matched_ids = set()
    for _, row in gt_val.iterrows():
        if pd.notna(row['matched_entity_ids']):
            for mid in str(row['matched_entity_ids']).split(','):
                val_matched_ids.add(mid.strip())
    
    print(f"  Val S1: {len(val_s1_ids)}, Train S1: {len(train_s1_ids)}")
    print(f"  Train matched IDs: {len(train_matched_ids)}, Val matched IDs: {len(val_matched_ids)}")
    
    # === TRAINING ===
    print("\n=== TRAINING ===")
    
    # Index for fast lookup
    s1_dict = s1.set_index('entity_id').to_dict('index')
    s2_dict = s2.set_index('entity_id').to_dict('index')
    s3_dict = s3.set_index('entity_id').to_dict('index')
    
    # Positive pairs from training split
    pos_pairs = []
    for _, row in tqdm(gt_train.iterrows(), total=len(gt_train), desc="Positives"):
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']):
            continue
        s1_rec = s1_dict.get(s1_id)
        if s1_rec is None:
            continue
        for mid in str(row['matched_entity_ids']).split(','):
            mid = mid.strip()
            rec = s2_dict.get(mid) if mid.startswith('S2') else s3_dict.get(mid)
            if rec is not None:
                pos_pairs.append((
                    s1_rec['business_name'], s1_rec['business_address'], s1_rec['country'],
                    rec['business_name'], rec['business_address'], rec['country'],
                ))
            if len(pos_pairs) >= 100000:
                break
        if len(pos_pairs) >= 100000:
            break
    
    print(f"Positive pairs: {len(pos_pairs)}")
    
    # Negative pairs
    s2s3_by_country = defaultdict(list)
    s2_list = list(s2_dict.values())[:200000]
    s3_list = list(s3_dict.values())[:200000]
    for rec in s2_list:
        s2s3_by_country[rec['country']].append(rec)
    for rec in s3_list:
        s2s3_by_country[rec['country']].append(rec)
    
    neg_pairs = []
    for s1_rec in list(s1_dict.values())[:50000]:
        if len(neg_pairs) >= len(pos_pairs) * 3:
            break
        pool = s2s3_by_country.get(s1_rec['country'], [])
        if not pool:
            continue
        for _ in range(3):
            other = pool[np.random.randint(len(pool))]
            neg_pairs.append((
                s1_rec['business_name'], s1_rec['business_address'], s1_rec['country'],
                other['business_name'], other['business_address'], other['country'],
            ))
    
    print(f"Negative pairs: {len(neg_pairs)}")
    
    # Compute features
    all_pairs = pos_pairs + neg_pairs
    labels = np.array([1]*len(pos_pairs) + [0]*len(neg_pairs), dtype=np.int8)
    
    idx = np.arange(len(all_pairs))
    np.random.shuffle(idx)
    all_pairs = [all_pairs[i] for i in idx]
    labels = labels[idx]
    
    chunks = []
    for i in tqdm(range(0, len(all_pairs), 5000), desc="Features"):
        chunks.append(compute_features_batch(all_pairs[i:i+5000]))
    X = np.vstack(chunks)
    y = labels
    
    print(f"X: {X.shape}, pos rate: {y.mean():.3f}")
    
    # Train LightGBM
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    train_data = lgb.Dataset(X_df, label=y)
    
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'scale_pos_weight': sum(y == 0) / max(sum(y == 1), 1),
    }
    
    model = lgb.train(params, train_data, num_boost_round=300,
                      callbacks=[lgb.log_evaluation(50)])
    
    # === VALIDATION ===
    print("\n=== VALIDATION ===")
    
    # Get val S1 records  
    val_s1_records = s1[s1['entity_id'].isin(val_s1_ids)].copy()
    
    # Get S2+S3 records that we need for validation
    # Include all matched IDs + a pool of random S2/S3 for blocking
    needed_s2s3_ids = val_matched_ids.copy()
    
    # Add random S2/S3 to simulate real blocking (need enough to test blocking recall)
    all_s2_ids = s2['entity_id'].values
    all_s3_ids = s3['entity_id'].values
    
    # Sample S2/S3 per country
    s2_sample = s2.sample(min(100000, len(s2)), random_state=42)
    s3_sample = s3.sample(min(100000, len(s3)), random_state=42)
    
    # Make sure matched IDs are included
    s2_matched = s2[s2['entity_id'].isin(needed_s2s3_ids & set(all_s2_ids))]
    s3_matched = s3[s3['entity_id'].isin(needed_s2s3_ids & set(all_s3_ids))]
    
    val_s2s3 = pd.concat([s2_sample, s3_sample, s2_matched, s3_matched]).drop_duplicates('entity_id')
    print(f"Val S2+S3 pool: {len(val_s2s3)}")
    
    # Create blocking texts
    val_s1_block = np.array([
        create_blocking_text(r['business_name'], r['business_address'], r['country'])
        for _, r in val_s1_records.iterrows()
    ])
    val_s2s3_block = np.array([
        create_blocking_text(r['business_name'], r['business_address'], r['country'])
        for _, r in val_s2s3.iterrows()
    ])
    
    val_s1_countries = val_s1_records['country'].values
    val_s2s3_countries = val_s2s3['country'].values
    val_s1_ids_arr = val_s1_records['entity_id'].values
    val_s2s3_ids_arr = val_s2s3['entity_id'].values
    val_s2s3_data = val_s2s3.to_dict('records')
    val_s1_data = val_s1_records.to_dict('records')
    
    # Run blocking + matching per country
    all_predictions = {}
    all_candidate_counts = []
    
    countries = sorted(set(val_s1_countries))
    
    for country in countries:
        s1_mask = val_s1_countries == country
        s2s3_mask = val_s2s3_countries == country
        
        s1_texts = val_s1_block[s1_mask]
        s2s3_texts = val_s2s3_block[s2s3_mask]
        s1_ids = val_s1_ids_arr[s1_mask]
        s2s3_ids = val_s2s3_ids_arr[s2s3_mask]
        s2s3_idx = np.where(s2s3_mask)[0]
        s1_idx = np.where(s1_mask)[0]
        
        n_s2s3 = len(s2s3_texts)
        if n_s2s3 == 0:
            for sid in s1_ids:
                all_predictions[sid] = []
            continue
        
        print(f"\n  {country}: {len(s1_texts)} S1, {n_s2s3} S2+S3")
        
        # TF-IDF blocking
        TOP_K = 15
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            max_features=300000, sublinear_tf=True, dtype=np.float32
        )
        s2s3_tfidf = vectorizer.fit_transform(s2s3_texts)
        
        actual_k = min(TOP_K, n_s2s3)
        nn = NearestNeighbors(n_neighbors=actual_k, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(s2s3_tfidf)
        
        # Query S1
        batch_size = 5000
        for b_start in range(0, len(s1_texts), batch_size):
            b_end = min(b_start + batch_size, len(s1_texts))
            batch_tfidf = vectorizer.transform(s1_texts[b_start:b_end])
            dists, indices = nn.kneighbors(batch_tfidf, n_neighbors=actual_k)
            
            for j in range(b_end - b_start):
                s1_id = s1_ids[b_start + j]
                s1_rec = val_s1_data[s1_idx[b_start + j]]
                
                cand_local_idxs = indices[j]
                cand_dists = dists[j]
                
                # Filter by distance
                mask = cand_dists < 0.95
                cand_local_idxs = cand_local_idxs[mask]
                
                if len(cand_local_idxs) == 0:
                    all_predictions[s1_id] = []
                    all_candidate_counts.append(0)
                    continue
                
                cand_eids = [s2s3_ids[ci] for ci in cand_local_idxs]
                cand_global_idxs = [s2s3_idx[ci] for ci in cand_local_idxs]
                all_candidate_counts.append(len(cand_eids))
                
                # Compute features
                pairs = []
                for gi in cand_global_idxs:
                    rec = val_s2s3_data[gi]
                    pairs.append((
                        s1_rec['business_name'], s1_rec['business_address'], s1_rec['country'],
                        rec['business_name'], rec['business_address'], rec['country'],
                    ))
                
                feats = compute_features_batch(pairs)
                feats_df = pd.DataFrame(feats, columns=FEATURE_NAMES)
                probs = model.predict(feats_df)
                
                matched = [cand_eids[k] for k, p in enumerate(probs) if p >= 0.5]
                all_predictions[s1_id] = matched
    
    # === EVALUATE ===
    print("\n\n=== EVALUATION ===")
    
    # Build ground truth for val
    gt_val_dict = {}
    for _, row in gt_val.iterrows():
        s1_id = row['source1_entity_id']
        if pd.isna(row['matched_entity_ids']):
            gt_val_dict[s1_id] = []
        else:
            gt_val_dict[s1_id] = [m.strip() for m in str(row['matched_entity_ids']).split(',')]
    
    # Compute F_0.5
    true_lists = []
    pred_lists = []
    
    # Also check blocking recall
    block_recall_hits = 0
    block_recall_total = 0
    
    for s1_id in gt_val_dict:
        true_ids = gt_val_dict[s1_id]
        pred_ids = all_predictions.get(s1_id, [])
        
        # Only count IDs that are in our S2+S3 pool
        true_in_pool = [tid for tid in true_ids if tid in set(val_s2s3['entity_id'])]
        
        true_lists.append(true_in_pool)
        pred_lists.append(pred_ids)
        
        # Blocking recall
        if true_in_pool:
            block_recall_total += len(true_in_pool)
            # Check if any candidate list contains the true ID
            # We need the candidates, not just the matches
            # For now, approximate with predictions
    
    score = f05_score(true_lists, pred_lists)
    
    # Breakdown
    singleton_scores = []
    non_singleton_scores = []
    for true_ids, pred_ids in zip(true_lists, pred_lists):
        if not true_ids:
            singleton_scores.append(1.0 if not pred_ids else 0.0)
        else:
            true_set = set(true_ids)
            pred_set = set(pred_ids)
            if not pred_set:
                non_singleton_scores.append(0.0)
                continue
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
            non_singleton_scores.append(f05)
    
    print(f"\nOverall F_0.5: {score:.4f}")
    print(f"  Singletons F_0.5: {np.mean(singleton_scores):.4f} (n={len(singleton_scores)})")
    print(f"  Non-singletons F_0.5: {np.mean(non_singleton_scores):.4f} (n={len(non_singleton_scores)})")
    if all_candidate_counts:
        print(f"  Avg candidates/entity: {np.mean(all_candidate_counts):.1f}")
    
    # Check various thresholds
    print("\nThreshold analysis would require storing probs - running with default 0.5")
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
