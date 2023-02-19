import glob
import pickle
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_validate(path):
    type_transform = {"clicks": 0, "carts": 1, "orders": 2}
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_transform).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


def calculate_deepwalk_similarity(string, model):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10 -10'
    sim = []
    aid = 'item_' + list[0]
    for i in list[1:]:
        simm = model.similarity(f'item_{i}', aid)
        sim.append(simm)
    sim_mean = sum(sim) / len(sim)
    sim_max = max(sim)
    return str(sim_mean) + ' ' + str(sim_max)


#  deepwalk,i2i相似度buys和clicks相似度的mean和max
def deepwalk_i2i_similarity1(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)
    print('开始筛选buys')
    valid1 = valid[valid['type'] != 0]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)
    print('开始筛选clicks')
    valid2 = valid[valid['type'] == 0]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('开始读取词向量！！')
    word2vec_last_month = gensim.models.KeyedVectors.load_word2vec_format(
        f'/home/niejianfei/otto/{stage}/preprocess/deepwalk_last_month.w2v',
        binary=False)
    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)
            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_deepwalk_similarity(x, word2vec_last_month))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['buys_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['buys_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[chunk['buys_sim_mean'] != -10])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_deepwalk_similarity(x, word2vec_last_month))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['clicks_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['clicks_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[chunk['clicks_sim_mean'] != -10])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str'])
            print(chunk[['buys_sim_mean', 'buys_sim_max', 'clicks_sim_mean', 'clicks_sim_max']])
            print(chunk)
            chunk.to_parquet(path)


#  deepwalk,i2i相似度orders和carts相似度的mean和max
def deepwalk_i2i_similarity2(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)
    print('开始筛选buys')
    valid1 = valid[valid['type'] == 2]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)
    print('开始筛选clicks')
    valid2 = valid[valid['type'] == 1]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('开始读取词向量！！')
    word2vec_last_month = gensim.models.KeyedVectors.load_word2vec_format(
        f'/home/niejianfei/otto/{stage}/preprocess/deepwalk_last_month.w2v',
        binary=False)
    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)
            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_deepwalk_similarity(x, word2vec_last_month))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['orders_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['orders_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[chunk['orders_sim_mean'] != -10])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_deepwalk_similarity(x, word2vec_last_month))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['carts_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['carts_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[chunk['carts_sim_mean'] != -10])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str'])
            print(chunk[['orders_sim_mean', 'orders_sim_max', 'carts_sim_mean', 'carts_sim_max']])
            print(chunk)
            chunk.to_parquet(path)


def calculate_deepwalk_similarity_tail(string, model):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10 -10 -10'
    sim = []
    aid = 'item_' + list[0]
    for i in list[1:]:
        simm = model.similarity(f'item_{i}', aid)
        sim.append(simm)
    if len(sim) >= 3:
        return str(sim[-1]) + ' ' + str(sim[-2]) + ' ' + str(sim[-3])
    elif len(sim) == 2:
        return str(sim[-1]) + ' ' + str(sim[-2]) + ' -10'
    else:
        return str(sim[-1]) + ' -10 -10'


def deepwalk_i2i_similarity_tail(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)
    print('开始筛选')

    valid1 = valid[valid['type'] != 0]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)

    valid2 = valid[valid['type'] == 0]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('开始读取词向量！！')
    word2vec_last_month = gensim.models.KeyedVectors.load_word2vec_format(f'/home/niejianfei/otto/{stage}/preprocess/deepwalk_last_month.w2v',
                                                                          binary=False)
    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)
            print(chunk.columns)

            print('merge')
            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(lambda x: calculate_deepwalk_similarity_tail(x, word2vec_last_month))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['buys_sim_-1'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['buys_sim_-2'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            chunk['buys_sim_-3'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[2]))
            print(chunk[chunk['buys_sim_-1'] != -10])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_deepwalk_similarity_tail(x, word2vec_last_month))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['clicks_sim_-1'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['clicks_sim_-2'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            chunk['clicks_sim_-3'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[2]))
            print(chunk[chunk['clicks_sim_-1'] != -10])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str'])
            print(chunk[['buys_sim_-1', 'buys_sim_-2', 'clicks_sim_-1', 'clicks_sim_-2']])
            print(chunk.columns)
            print(chunk)
            chunk.to_parquet(path)


def calculate_deepwalk_u2i_similarity(string, model):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10'
    aid_emb = np.array(model[f'item_{list[0]}'])
    user_emb = np.zeros(64)
    for i in list[1:]:
        user_emb += np.array(model[f'item_{i}']) / (len(list) - 1)

    cos_sim = cosine_similarity(aid_emb.reshape(1, -1), user_emb.reshape(1, -1))

    return str(cos_sim[0][0])


#  deepwalk,u2i相似度orders和carts相似度的mean和max
def deepwalk_u2i_similarity(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)

    print('开始筛选order')
    valid1 = valid[valid['type'] == 2]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print(df.head(10))
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    print(sentences_df)
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)

    print('开始筛选cart')
    valid2 = valid[valid['type'] == 1]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print(df1.head(10))
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    print(sentences_df1)
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('开始筛选click')
    valid3 = valid[valid['type'] == 0]
    print(valid3)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df2 = valid3.sort_values(by=["session", "ts"], ascending=True)
    print(df2.head(10))
    print('生成list')
    sentences_df2 = pd.DataFrame(df2.groupby('session')['aid'].agg(list))
    sentences_df2.columns = ['clicks']
    print(sentences_df2)
    sentences_df2["clicks_str1"] = sentences_df2.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df2 = sentences_df2.drop(columns='clicks')
    print(sentences_df2)

    print('开始读取词向量！！')
    word2vec_last_month = gensim.models.KeyedVectors.load_word2vec_format('/home/niejianfei/deepwalk_last_month.w2v',
                                                                          binary=False)
    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)
            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('order开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_deepwalk_u2i_similarity(x, word2vec_last_month))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['orders_user_item_sim'] = chunk['sim_score_str'].astype('float32')
            print(chunk[chunk['orders_user_item_sim'] != -10])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('cart开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_deepwalk_u2i_similarity(x, word2vec_last_month))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['carts_user_item_sim'] = chunk['clicks_sim_score_str'].astype('float32')
            print(chunk[chunk['carts_user_item_sim'] != -10])

            chunk = chunk.merge(sentences_df2, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list1'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str1'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str1'] = chunk['clicks_sim_list1'].apply(
                lambda x: calculate_deepwalk_u2i_similarity(x, word2vec_last_month))
            print(chunk[['clicks_str1', 'clicks_sim_list1', 'clicks_sim_score_str1']])
            chunk['clicks_user_item_sim'] = chunk['clicks_sim_score_str1'].astype('float32')
            print(chunk[chunk['clicks_user_item_sim'] != -10])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str', 'clicks_str1', 'clicks_sim_list1',
                         'clicks_sim_score_str1'])
            print(chunk[['orders_user_item_sim', 'carts_user_item_sim', 'clicks_user_item_sim']])
            print(chunk.columns)
            print(chunk)
            chunk.to_parquet(path)


def calculate_prone_similarity(string, model, aid_num_dict):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10 -10'
    sim = []
    aid = list[0]
    if int(aid) in aid_num_dict:
        for i in list[1:]:
            if int(i) in aid_num_dict:
                simm = model.similarity(str(aid_num_dict[int(i)]), str(aid_num_dict[int(aid)]))
                sim.append(simm)
    if len(sim) == 0:
        return '-10 -10'
    sim_mean = sum(sim) / len(sim)
    sim_max = max(sim)
    return str(sim_mean) + ' ' + str(sim_max)


#  prone,i2i相似度buys和clicks相似度的mean和max
def prone_i2i_similarity(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)
    print('开始筛选')
    valid1 = valid[valid['type'] != 0]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)

    valid2 = valid[valid['type'] == 0]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('开始读取词向量！！')
    proNE_last_month = gensim.models.KeyedVectors.load_word2vec_format(
        f"/home/niejianfei/otto/{stage}/preprocess/proNE_last_month_enhanced.emb",
        binary=False)

    print("开始读取aim_num映射文件！！！")
    f_read = open(f'/home/niejianfei/otto/{stage}/preprocess/aid_num_dict.pkl', 'rb')
    aid_num_dict = pickle.load(f_read)
    f_read.close()
    print('输出', aid_num_dict[0])
    print("aim_num映射文件读取完毕！！！")

    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)

            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_prone_similarity(x, proNE_last_month, aid_num_dict))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['proNE_buys_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['proNE_buys_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[chunk['proNE_buys_sim_mean'] != -10])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_prone_similarity(x, proNE_last_month, aid_num_dict))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['proNE_clicks_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['proNE_clicks_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[chunk['proNE_clicks_sim_mean'] != -10])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str'])
            print(chunk[['proNE_buys_sim_mean', 'proNE_buys_sim_max', 'proNE_clicks_sim_mean', 'proNE_clicks_sim_max']])
            print(chunk)
            chunk.to_parquet(path)


def calculate_prone_similarity_tail(string, model, aid_num_dict):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10 -10 -10'
    sim = []
    aid = list[0]
    if int(aid) in aid_num_dict:
        for i in list[1:]:
            if int(i) in aid_num_dict:
                simm = model.similarity(str(aid_num_dict[int(i)]), str(aid_num_dict[int(aid)]))
                sim.append(simm)
    if len(sim) >= 3:
        return str(sim[-1]) + ' ' + str(sim[-2]) + ' ' + str(sim[-3])
    elif len(sim) == 2:
        return str(sim[-1]) + ' ' + str(sim[-2]) + ' -10'
    elif len(sim) == 1:
        return str(sim[-1]) + ' -10 -10'
    else:
        return '-10 -10 -10'


def prone_i2i_similarity_tail(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)
    print('开始筛选')

    valid1 = valid[valid['type'] != 0]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)

    valid2 = valid[valid['type'] == 0]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('开始读取词向量！！')
    proNE_last_month = gensim.models.KeyedVectors.load_word2vec_format(
        f"/home/niejianfei/otto/{stage}/preprocess/proNE_last_month_enhanced.emb",
        binary=False)

    print("开始读取aim_num映射文件！！！")
    f_read = open(f'/home/niejianfei/otto/{stage}/preprocess/aid_num_dict.pkl', 'rb')
    aid_num_dict = pickle.load(f_read)
    f_read.close()
    print('输出', aid_num_dict[0])
    print("aim_num映射文件读取完毕！！！")

    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)
            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)
            print(chunk.columns)

            print('merge')
            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_prone_similarity_tail(x, proNE_last_month, aid_num_dict))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['proNE_buys_sim_-1'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['proNE_buys_sim_-2'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            chunk['proNE_buys_sim_-3'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[2]))
            print(chunk[chunk['proNE_buys_sim_-1'] != -10])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_prone_similarity_tail(x, proNE_last_month, aid_num_dict))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['proNE_clicks_sim_-1'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['proNE_clicks_sim_-2'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            chunk['proNE_clicks_sim_-3'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[2]))
            print(chunk[chunk['proNE_clicks_sim_-1'] != -10])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str'])
            print(chunk[['proNE_buys_sim_-1', 'proNE_buys_sim_-2', 'proNE_clicks_sim_-1', 'proNE_clicks_sim_-2']])
            print(chunk.columns)
            print(chunk)
            chunk.to_parquet(path)


def calculate_MF_similarity(string, array):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10' + ' -10' * 3
    sim = []
    aid = int(list[0])
    for i in list[1:]:
        simm = cosine_similarity(array[aid].reshape(1, -1), array[int(i)].reshape(1, -1))[0][0]
        sim.append(simm)
    sim_sum = sum(sim)
    sim_mean = sim_sum / len(sim)
    sim_max = max(sim)

    return str(sim_mean) + ' ' + str(sim_max) + ' ' + str(sim_sum) + ' ' + str(sim[-1])


# bpr,als,lmf,u2i相似度
def bpr_als_lmf_u2i_similarity(stage, candidate_type, start, end):
    print('bpr')
    bpr_user_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/bpr_user_emb.npy')
    bpr_item_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/bpr_item_emb.npy')
    print('als')
    als_user_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/als_user_emb.npy')
    als_item_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/als_item_emb.npy')
    print('lmf')
    lmf_user_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/lmf_user_emb.npy')
    lmf_item_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/lmf_item_emb.npy')

    for t in candidate_type:
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)

            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)

            chunk['list'] = chunk['session'].astype('str') + ' ' + chunk['aid'].astype('str')
            print(chunk)
            chunk['bpr_user_item_sim'] = chunk['list'].map(
                lambda x: np.dot(bpr_user_emb[int(x.split(' ')[0])], bpr_item_emb[int(x.split(' ')[1])]))
            print(chunk['bpr_user_item_sim'].describe())

            chunk['als_user_item_sim'] = chunk['list'].map(
                lambda x: np.dot(als_user_emb[int(x.split(' ')[0])], als_item_emb[int(x.split(' ')[1])]))
            print(chunk['als_user_item_sim'].describe())

            chunk['lmf_user_item_sim'] = chunk['list'].map(
                lambda x: np.dot(lmf_user_emb[int(x.split(' ')[0])], lmf_item_emb[int(x.split(' ')[1])]))
            print(chunk['lmf_user_item_sim'].describe())

            print(chunk)
            chunk.to_parquet(path)


def bpr_als_lmf_i2i_similarity(stage, candidate_type, start, end):
    print('开始读取数据！！！')
    valid = load_validate(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    print(valid)
    print('开始筛选')

    valid1 = valid[valid['type'] != 0]
    print(valid1)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df = valid1.sort_values(by=["session", "ts"], ascending=True)
    print(df.head(10))
    print('生成list')
    sentences_df = pd.DataFrame(df.groupby('session')['aid'].agg(list))
    sentences_df.columns = ['carts_and_orders']
    print(sentences_df)
    sentences_df["carts_and_orders_str"] = sentences_df.carts_and_orders.apply(lambda x: " ".join(map(str, x)))
    sentences_df = sentences_df.drop(columns='carts_and_orders')
    print(sentences_df)

    valid2 = valid[valid['type'] == 0]
    print(valid2)
    print('开始排序')
    # 分别对session_id聚合，对时间进行排序
    df1 = valid2.sort_values(by=["session", "ts"], ascending=True)
    print(df1.head(10))
    print('生成list')
    sentences_df1 = pd.DataFrame(df1.groupby('session')['aid'].agg(list))
    sentences_df1.columns = ['clicks']
    print(sentences_df1)
    sentences_df1["clicks_str"] = sentences_df1.clicks.apply(lambda x: " ".join(map(str, x)))
    sentences_df1 = sentences_df1.drop(columns='clicks')
    print(sentences_df1)

    print('bpr')
    bpr_item_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/bpr_item_emb.npy')
    print('als')
    als_item_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/als_item_emb.npy')
    print('lmf')
    lmf_item_emb = np.load(f'/home/niejianfei/otto/{stage}/preprocess/lmf_item_emb.npy')

    for t in candidate_type:
        # 只导入训练数据
        print('开始导入数据')
        for i in range(start, end):
            path = f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)

            chunk = chunk.astype("float32")
            chunk['session'] = chunk['session'].astype('int32')
            chunk['aid'] = chunk['aid'].astype('int32')
            print(chunk)

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_MF_similarity(x, bpr_item_emb))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['bpr_buys_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0])).astype('float32')
            chunk['bpr_buys_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1])).astype('float32')
            chunk['bpr_buys_sim_sum'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[2])).astype('float32')
            chunk['bpr_buys_sim_-1'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[3])).astype('float32')
            chunk['bpr_buys_sim_-2'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[4])).astype('float32')
            chunk['bpr_buys_sim_-3'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[5])).astype('float32')
            print(chunk[chunk['bpr_buys_sim_-3'] != -10])
            print(chunk)

            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_MF_similarity(x, als_item_emb))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['als_buys_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0])).astype('float32')
            chunk['als_buys_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1])).astype('float32')
            chunk['als_buys_sim_sum'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[2])).astype('float32')
            chunk['als_buys_sim_-1'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[3])).astype('float32')
            chunk['als_buys_sim_-2'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[4])).astype('float32')
            chunk['als_buys_sim_-3'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[5])).astype('float32')

            print(chunk[chunk['als_buys_sim_-3'] != -10])
            print(chunk)

            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(
                lambda x: calculate_MF_similarity(x, lmf_item_emb))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['lmf_buys_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0])).astype('float32')
            chunk['lmf_buys_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1])).astype('float32')
            chunk['lmf_buys_sim_sum'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[2])).astype('float32')
            chunk['lmf_buys_sim_-1'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[3])).astype('float32')
            chunk['lmf_buys_sim_-2'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[4])).astype('float32')
            chunk['lmf_buys_sim_-3'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[5])).astype('float32')

            print(chunk[chunk['lmf_buys_sim_-3'] != -10])
            print(chunk)

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_MF_similarity(x, bpr_item_emb))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['bpr_clicks_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0])).astype('float32')
            chunk['bpr_clicks_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1])).astype('float32')
            chunk['bpr_clicks_sim_sum'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[2])).astype('float32')
            chunk['bpr_clicks_sim_-1'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[3])).astype('float32')
            chunk['bpr_clicks_sim_-2'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[4])).astype('float32')
            chunk['bpr_clicks_sim_-3'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[5])).astype('float32')
            print(chunk[chunk['bpr_clicks_sim_-3'] != -10])
            print(chunk)

            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_MF_similarity(x, als_item_emb))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['als_clicks_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0])).astype(
                'float32')
            chunk['als_clicks_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1])).astype(
                'float32')
            chunk['als_clicks_sim_sum'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[2])).astype(
                'float32')
            chunk['als_clicks_sim_-1'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[3])).astype(
                'float32')
            print(chunk[chunk['als_clicks_sim_-1'] != -10])
            print(chunk)

            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_MF_similarity(x, lmf_item_emb))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['lmf_clicks_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0])).astype('float32')
            chunk['lmf_clicks_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1])).astype('float32')
            chunk['lmf_clicks_sim_sum'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[2])).astype('float32')
            chunk['lmf_clicks_sim_-1'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[3])).astype('float32')
            chunk['lmf_clicks_sim_-2'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[4])).astype('float32')
            chunk['lmf_clicks_sim_-3'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[5])).astype('float32')
            print(chunk[chunk['lmf_clicks_sim_-3'] != -10])
            print(chunk)

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str'])
            print(chunk['als_clicks_sim_max'])
            print(chunk.columns)
            print(chunk)
            chunk.to_parquet(path)


def similarity_features(stage, candidate_type, start, end):
    # buys&clicks * 4
    deepwalk_i2i_similarity1(stage, candidate_type, start, end)
    # orders&carts * 4
    deepwalk_i2i_similarity2(stage, candidate_type, start, end)
    # buys&clicks * 6
    deepwalk_i2i_similarity_tail(stage, candidate_type, start, end)
    # buys&clicks * 3
    deepwalk_u2i_similarity(stage, candidate_type, start, end)

    # buys&clicks * 4
    prone_i2i_similarity(stage, candidate_type, start, end)
    # buys&clicks * 6
    prone_i2i_similarity_tail(stage, candidate_type, start, end)


if __name__ == '__main__':
    IS_TRAIN = True
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'
    candidate_type = ['clicks', 'carts', 'orders']
    similarity_features(stage, candidate_type, 0, 8)
