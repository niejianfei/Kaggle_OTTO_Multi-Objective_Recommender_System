import glob
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import implicit
IS_TRAIN = True
type_transform = {"clicks": 0, "carts": 1, "orders": 2}


def load_data(path):
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_transform).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


if IS_TRAIN:
    test_df = load_data('/home/niejianfei/otto/CV/data/*_parquet/*')
else:
    test_df = load_data('/home/niejianfei/otto/LB/data/*_parquet/*')


dic1 = {0: 1, 1: 5, 2: 4}
test_df['type'] = test_df['type'].map(dic1)
grouped_df = test_df.groupby(['session', 'aid']).sum().reset_index()

# sparse_content_person = sparse.csr_matrix(
#     (grouped_df['type'].astype(float), (grouped_df['aid'], grouped_df['session'])))
sparse_person_content = sparse.csr_matrix(
    (grouped_df['type'].astype(float), (grouped_df['session'], grouped_df['aid'])))

print(sparse_person_content.shape)
# print(sparse_person_content.shape)

alpha = 15
sparse_person_content = (sparse_person_content * alpha).astype('double')

# from implicit.nearest_neighbours import bm25_weight
# # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
# # and to reduce the weight given to popular items
# artist_user_plays = bm25_weight(sparse_person_content, K1=100, B=0.8)

model1 = implicit.bpr.BayesianPersonalizedRanking(factors=64, regularization=0.1)
model2 = implicit.als.AlternatingLeastSquares(factors=64, regularization=0.1)
model3 = implicit.lmf.LogisticMatrixFactorization(factors=64, regularization=0.6)

models = [model1, model2, model3]
names = ['bpr', 'als', 'lmf']

for model, name in zip(models, names):
    model.fit(sparse_person_content)
    user_emb = model.user_factors.to_numpy()
    print("user")
    print(user_emb[0], len(user_emb))
    print("item")
    item_emb = model.item_factors.to_numpy()
    print(item_emb[0], len(item_emb))
    print('save')
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'
    np.save(f'/home/niejianfei/otto/{stage}/preprocess/{name}_user_emb', user_emb)
    np.save(f'/home/niejianfei/otto/{stage}/preprocess/{name}_item_emb', item_emb)




