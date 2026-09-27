import sys
sys.path.insert(0, 'code/business_entity_resolution/src')
import pandas as pd, sparse_dot_topn as sdt
from sklearn.feature_extraction.text import TfidfVectorizer
from fast_inference import normalize_name, normalize_addr

s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=50)
s2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t', nrows=1000)

s1_fr = s1[s1['country'] == 'France']
s2_fr = s2[s2['country'] == 'France']

s1_block = [f"{normalize_name(n)} {normalize_name(n)} {normalize_addr(a)}" for n, a in zip(s1_fr['business_name'], s1_fr['business_address'])]
s2_block = [f"{normalize_name(n)} {normalize_name(n)} {normalize_addr(a)}" for n, a in zip(s2_fr['business_name'], s2_fr['business_address'])]

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=300000, sublinear_tf=True)
s2_mat = vec.fit_transform(s2_block)
s1_mat = vec.transform(s1_block)

topn = sdt.sp_matmul_topn(s1_mat, s2_mat.T.tocsr(), top_n=5, threshold=0.05, sort=True)
print('Shape:', topn.shape)
for i in range(min(5, len(s1_fr))):
    idxs = topn.indices[topn.indptr[i]:topn.indptr[i+1]]
    data = topn.data[topn.indptr[i]:topn.indptr[i+1]]
    print(f"Query {i}: {s1_fr.iloc[i]['business_name']}")
    for idx, sim in zip(idxs, data):
        print(f"   -> ({sim:.3f}) {s2_fr.iloc[idx]['business_name']}")
