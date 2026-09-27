import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

# Index for fast lookup
s2_idx = s2.set_index('entity_id')
s3_idx = s3.set_index('entity_id')

# Look at first few matched entities
for i, row in gt.head(5).iterrows():
    s1_id = row['source1_entity_id']
    s1_rec = s1[s1['entity_id']==s1_id].iloc[0]
    print(f"=== {s1_id} ===")
    print(f"  S1 Name: {s1_rec['business_name']}")
    print(f"  S1 Addr: {s1_rec['business_address']}")
    print(f"  S1 Country: {s1_rec['country']}")
    
    if pd.notna(row['matched_entity_ids']):
        for mid in str(row['matched_entity_ids']).split(','):
            mid = mid.strip()
            try:
                if mid.startswith('S2'):
                    rec = s2_idx.loc[mid]
                else:
                    rec = s3_idx.loc[mid]
                print(f"  {mid} Name: {rec['business_name']}")
                print(f"  {mid} Addr: {rec['business_address']}")
                print(f"  {mid} Country: {rec['country']}")
            except KeyError:
                print(f"  {mid}: NOT FOUND")
    print()
