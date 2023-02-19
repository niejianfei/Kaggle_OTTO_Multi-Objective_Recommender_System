import glob
import pandas as pd


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


def calculate_cf_u2i_similarity(string, dic):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10 -10'
    aid = list[0]
    score = []
    for i in list[1:]:
        if aid + ' ' + i in dic:
            temp_score = float(dic[aid + ' ' + i])
        else:
            temp_score = 0
        score.append(temp_score)
    return str(max(score)) + ' ' + str(sum(score))


# 计算候选aid与user序列co_visitation矩阵的权重之和
def cf_u2i_similarity(stage, candidate_type, start, end):
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

    valid2 = valid
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

    print('开始读取字典！！')
    print('click')
    VER = 6
    print(VER)
    dic_click = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_20_clicks_v{VER}_0.pqt')
    DISK_PIECES = 4
    for k in range(1, DISK_PIECES):
        dic_click = dic_click.append(
            pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_20_clicks_v{VER}_{k}.pqt'))

    dic_click['aids1'] = dic_click['aid_x'].astype('str') + ' ' + dic_click['aid_y'].astype('str')
    dic_click['aids2'] = dic_click['aid_y'].astype('str') + ' ' + dic_click['aid_x'].astype('str')

    dic_click = dic_click.drop(columns=['aid_x', 'aid_y'])
    dic_click1 = dic_click[['aids1', 'wgt']]
    print(dic_click1)
    dic_click2 = dic_click[['aids2', 'wgt']]
    dic_click2.columns = ['aids1', 'wgt']
    print(dic_click2)
    dic_click = dic_click1.append(dic_click2)
    print(dic_click)
    dic_click.index = dic_click['aids1']
    print(dic_click)
    dic_click = dic_click['wgt'].to_dict()
    print('0 532042' in dic_click)
    print('532042 0' in dic_click)
    print('0 532022242' in dic_click)

    print('hot')
    dic_hot = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_15_carts_orders_v{VER}_0.pqt')
    DISK_PIECES = 4
    for k in range(1, DISK_PIECES):
        dic_hot = dic_hot.append(
            pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_15_carts_orders_v{VER}_{k}.pqt'))

    dic_hot['aids1'] = dic_hot['aid_x'].astype('str') + ' ' + dic_hot['aid_y'].astype('str')
    dic_hot['aids2'] = dic_hot['aid_y'].astype('str') + ' ' + dic_hot['aid_x'].astype('str')

    dic_hot = dic_hot.drop(columns=['aid_x', 'aid_y'])
    dic_click1 = dic_hot[['aids1', 'wgt']]
    print(dic_click1)
    dic_click2 = dic_hot[['aids2', 'wgt']]
    dic_click2.columns = ['aids1', 'wgt']
    print(dic_click2)
    dic_hot = dic_click1.append(dic_click2)
    print(dic_hot)
    dic_hot.index = dic_hot['aids1']
    print(dic_hot)
    dic_hot = dic_hot['wgt'].to_dict()
    print('0 532042' in dic_hot)
    print('532042 0' in dic_hot)
    print('0 532022242' in dic_hot)

    print('buys')
    dic_buys = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_15_buy2buy_v{VER}_0.pqt')
    print(dic_buys)

    dic_buys['aids1'] = dic_buys['aid_x'].astype('str') + ' ' + dic_buys['aid_y'].astype('str')
    dic_buys['aids2'] = dic_buys['aid_y'].astype('str') + ' ' + dic_buys['aid_x'].astype('str')

    dic_buys = dic_buys.drop(columns=['aid_x', 'aid_y'])
    dic_click1 = dic_buys[['aids1', 'wgt']]
    print(dic_click1)
    dic_click2 = dic_buys[['aids2', 'wgt']]
    dic_click2.columns = ['aids1', 'wgt']
    print(dic_click2)
    dic_buys = dic_click1.append(dic_click2)
    print(dic_buys)
    dic_buys.index = dic_buys['aids1']
    print(dic_buys)
    dic_buys = dic_buys['wgt'].to_dict()
    print('0 532042' in dic_buys)
    print('532042 0' in dic_buys)
    print('0 532022242' in dic_buys)

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

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(lambda x: calculate_cf_u2i_similarity(x, dic_buys))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['buys_CF_sim_max'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['buys_CF_sim_sum'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[(chunk['buys_CF_sim_max'] != -10) & (chunk['buys_CF_sim_max'] != 0)])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_cf_u2i_similarity(x, dic_click))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['clicks_CF_sim_max'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['clicks_CF_sim_sum'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[(chunk['clicks_CF_sim_max'] != -10) & (chunk['clicks_CF_sim_max'] != 0)])

            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str1'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_cf_u2i_similarity(x, dic_hot))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str1']])
            chunk['hot_CF_sim_max'] = chunk['clicks_sim_score_str1'].apply(lambda x: float(x.split(' ')[0]))
            chunk['hot_CF_sim_sum'] = chunk['clicks_sim_score_str1'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[(chunk['hot_CF_sim_max'] != -10) & (chunk['hot_CF_sim_max'] != 0)])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str', 'clicks_sim_score_str1'])
            print(chunk[['buys_CF_sim_max', 'buys_CF_sim_sum', 'hot_CF_sim_max', 'hot_CF_sim_sum', 'clicks_CF_sim_max',
                         'clicks_CF_sim_sum']])
            print(chunk.columns)
            print(chunk)
            chunk.to_parquet(path)


def calculate_cf_u2i_similarity_tail(string, dic):
    list = string.split(' ')
    if int(list[-1]) < 0:
        return '-10 -10'
    aid = list[0]
    score = []
    for i in list[1:]:
        if aid + ' ' + i in dic:
            temp_score = float(dic[aid + ' ' + i])
        else:
            temp_score = 0
        score.append(temp_score)
    return str(sum(score) / len(score)) + ' ' + str(score[-1])


# 计算候选aid与user序列co_visitation矩阵的权重之和
def cf_u2i_similarity_tail(stage, candidate_type, start, end):
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

    valid2 = valid
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

    print('开始读取字典！！')
    print('click')
    VER = 6
    print(VER)
    dic_click = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_20_clicks_v{VER}_0.pqt')
    DISK_PIECES = 4
    for k in range(1, DISK_PIECES):
        dic_click = dic_click.append(
            pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_20_clicks_v{VER}_{k}.pqt'))

    dic_click['aids1'] = dic_click['aid_x'].astype('str') + ' ' + dic_click['aid_y'].astype('str')
    dic_click['aids2'] = dic_click['aid_y'].astype('str') + ' ' + dic_click['aid_x'].astype('str')

    dic_click = dic_click.drop(columns=['aid_x', 'aid_y'])
    dic_click1 = dic_click[['aids1', 'wgt']]
    print(dic_click1)
    dic_click2 = dic_click[['aids2', 'wgt']]
    dic_click2.columns = ['aids1', 'wgt']
    print(dic_click2)
    dic_click = dic_click1.append(dic_click2)
    print(dic_click)
    dic_click.index = dic_click['aids1']
    print(dic_click)
    dic_click = dic_click['wgt'].to_dict()
    print('0 532042' in dic_click)
    print('532042 0' in dic_click)
    print('0 532022242' in dic_click)

    print('hot')
    dic_hot = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_15_carts_orders_v{VER}_0.pqt')
    DISK_PIECES = 4
    for k in range(1, DISK_PIECES):
        dic_hot = dic_hot.append(
            pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_15_carts_orders_v{VER}_{k}.pqt'))

    dic_hot['aids1'] = dic_hot['aid_x'].astype('str') + ' ' + dic_hot['aid_y'].astype('str')
    dic_hot['aids2'] = dic_hot['aid_y'].astype('str') + ' ' + dic_hot['aid_x'].astype('str')

    dic_hot = dic_hot.drop(columns=['aid_x', 'aid_y'])
    dic_click1 = dic_hot[['aids1', 'wgt']]
    print(dic_click1)
    dic_click2 = dic_hot[['aids2', 'wgt']]
    dic_click2.columns = ['aids1', 'wgt']
    print(dic_click2)
    dic_hot = dic_click1.append(dic_click2)
    print(dic_hot)
    dic_hot.index = dic_hot['aids1']
    print(dic_hot)
    dic_hot = dic_hot['wgt'].to_dict()
    print('0 532042' in dic_hot)
    print('532042 0' in dic_hot)
    print('0 532022242' in dic_hot)

    print('buys')
    dic_buys = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/top_15_buy2buy_v{VER}_0.pqt')
    print(dic_buys)

    dic_buys['aids1'] = dic_buys['aid_x'].astype('str') + ' ' + dic_buys['aid_y'].astype('str')
    dic_buys['aids2'] = dic_buys['aid_y'].astype('str') + ' ' + dic_buys['aid_x'].astype('str')

    dic_buys = dic_buys.drop(columns=['aid_x', 'aid_y'])
    dic_click1 = dic_buys[['aids1', 'wgt']]
    print(dic_click1)
    dic_click2 = dic_buys[['aids2', 'wgt']]
    dic_click2.columns = ['aids1', 'wgt']
    print(dic_click2)
    dic_buys = dic_click1.append(dic_click2)
    print(dic_buys)
    dic_buys.index = dic_buys['aids1']
    print(dic_buys)
    dic_buys = dic_buys['wgt'].to_dict()
    print('0 532042' in dic_buys)
    print('532042 0' in dic_buys)
    print('0 532022242' in dic_buys)

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

            chunk = chunk.merge(sentences_df, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['carts_and_orders_str'].astype('str')
            print('开始计算相似度！！！')
            chunk['sim_score_str'] = chunk['sim_list'].apply(lambda x: calculate_cf_u2i_similarity_tail(x, dic_buys))
            print(chunk[['carts_and_orders_str', 'sim_list', 'sim_score_str']])
            chunk['buys_CF_sim_mean'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['buys_CF_sim_-1'] = chunk['sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[(chunk['buys_CF_sim_mean'] != -10) & (chunk['buys_CF_sim_-1'] != 0)])

            chunk = chunk.merge(sentences_df1, left_on='session', right_index=True, how='left').fillna(value=-1)
            print(chunk)
            chunk['clicks_sim_list'] = chunk['aid'].astype('str') + ' ' + chunk['clicks_str'].astype('str')
            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_cf_u2i_similarity_tail(x, dic_click))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str']])
            chunk['clicks_CF_sim_mean'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[0]))
            chunk['clicks_CF_sim_-1'] = chunk['clicks_sim_score_str'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[(chunk['clicks_CF_sim_mean'] != -10) & (chunk['clicks_CF_sim_-1'] != 0)])

            print('click开始计算相似度！！！')
            chunk['clicks_sim_score_str1'] = chunk['clicks_sim_list'].apply(
                lambda x: calculate_cf_u2i_similarity_tail(x, dic_hot))
            print(chunk[['clicks_str', 'clicks_sim_list', 'clicks_sim_score_str1']])
            chunk['hot_CF_sim_mean'] = chunk['clicks_sim_score_str1'].apply(lambda x: float(x.split(' ')[0]))
            chunk['hot_CF_sim_-1'] = chunk['clicks_sim_score_str1'].apply(lambda x: float(x.split(' ')[1]))
            print(chunk[(chunk['hot_CF_sim_mean'] != -10) & (chunk['hot_CF_sim_-1'] != 0)])

            chunk = chunk.drop(
                columns=['carts_and_orders_str', 'sim_list', 'sim_score_str', 'clicks_str', 'clicks_sim_list',
                         'clicks_sim_score_str', 'clicks_sim_score_str1'])
            print(chunk[['buys_CF_sim_max', 'buys_CF_sim_sum', 'hot_CF_sim_max', 'hot_CF_sim_sum', 'clicks_CF_sim_max',
                         'clicks_CF_sim_sum']])
            print(chunk.columns)
            print(chunk)
            chunk.to_parquet(path)


# 三个矩阵的特征
def co_visitation_features(stage, candidate_type, start, end):
    cf_u2i_similarity(stage, candidate_type, start, end)
    cf_u2i_similarity_tail(stage, candidate_type, start, end)


if __name__ == '__main__':
    IS_TRAIN = True
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'
    candidate_type = ['clicks', 'carts', 'orders']
    co_visitation_features(stage, candidate_type, 0, 8)
