# 召回策略
# 1.基于历史序列召回    全部aids
# 2.基于co—visitation召回(I2I)  100aids
# 3.基于规则召回
#   点击最多
#   加购最多
#   购买最多
#   热门商品
# 4.基于embedding召回
#   deepwalk last week(I2I)     80aids
#   deepwalk last month(I2I)     80aids

# 开始计算recall@220！！！
# clicks recall = 0.628
# carts recall = 0.519
# orders recall = 0.716
# =============
# Overall Recall = 0.6481
# =============

import gensim
import pandas as pd, numpy as np
import glob
from collections import Counter
import itertools

type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}
VER = 6
DISK_PIECES = 4
IS_TRAIN = True


def load_data(path):
    dfs = []
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


def pqt_to_dict(df):
    df = df.loc[df.n < 20].drop('n', axis=1)
    # df['sim_aid_and_score'] = df['aid_y'].astype('str') + '#' + df['wgt'].astype('str')
    return df.groupby('aid_x').aid_y.apply(list).to_dict()


if IS_TRAIN:
    stage = 'CV'
    data = ''
else:
    stage = 'LB'
    data = 'all_data_'

print('加载原始数据！！')
test_df = load_data(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
train_df = load_data(f'/home/niejianfei/otto/{stage}/data/*_parquet/*')

print("开始读取co_visitation矩阵数据！！！")
# LOAD THREE CO-VISITATION MATRICES
top_20_clicks = pqt_to_dict(
    pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/{data}top_20_clicks_v{VER}_0.pqt'))
for k in range(1, DISK_PIECES):
    top_20_clicks.update(
        pqt_to_dict(
            pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/{data}top_20_clicks_v{VER}_{k}.pqt')))
top_20_buys = pqt_to_dict(
    pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/{data}top_15_carts_orders_v{VER}_0.pqt'))
for k in range(1, DISK_PIECES):
    top_20_buys.update(
        pqt_to_dict(
            pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/{data}top_15_carts_orders_v{VER}_{k}.pqt')))
top_20_buy2buy = pqt_to_dict(
    pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/{data}top_15_buy2buy_v{VER}_0.pqt'))

print('开始读取deepwalk词向量！！')
word2vec_last_week = gensim.models.KeyedVectors.load_word2vec_format(
    f'/home/niejianfei/otto/{stage}/preprocess/deepwalk_last_week.w2v',
    binary=False)
word2vec_last_month = gensim.models.KeyedVectors.load_word2vec_format(
    f'/home/niejianfei/otto/{stage}/preprocess/deepwalk_last_month.w2v',
    binary=False)

# 基于规则，热门商品
print("开始生成test热门商品！！！")
top_clicks = test_df.loc[test_df['type'] == 0, 'aid'].value_counts()[:200].to_dict()
top_carts = test_df.loc[test_df['type'] == 1, 'aid'].value_counts()[:200].to_dict()
top_orders = test_df.loc[test_df['type'] == 2, 'aid'].value_counts()[:200].to_dict()

# 修改权重
type_weight_multipliers = {0: 1, 1: 5, 2: 4}
print("开始生成test hot商品！！！")
test_df['score'] = test_df['type'].map(type_weight_multipliers)
top_hot_items = test_df.groupby('aid')['score'].apply(lambda x: x.sum()) \
                    .sort_values(ascending=False)[:200].to_dict()
print('开始生成train hot商品！！！')
train_df['score'] = train_df['type'].map(type_weight_multipliers)
top_hot_items_last_month = train_df.groupby('aid')['score'].apply(lambda x: x.sum()) \
                               .sort_values(ascending=False)[:200].to_dict()
print(top_hot_items_last_month)
print('开始生成train click hot商品！！！')
train_df['score'] = 1
top_clicks_items_last_month = train_df.groupby('aid')['score'].apply(lambda x: x.sum()) \
                                  .sort_values(ascending=False)[:200].to_dict()
print(top_clicks_items_last_month)


def suggest_clicks(df):
    # USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    # unique_aids = list(dict.fromkeys(aids[::-1]))

    # RERANK CANDIDATES USING WEIGHTS
    # 直接召回历史序列按权重划分的aids
    # 等比数列  2**0.1 - 2**1  权重差的过大？ 0.07-1？对于大序列，这点不好
    weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
    aids_temp = Counter()
    # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
    # 历史序列召回，考虑时间效应，优先召回时间近的
    for aid, w, t in zip(aids, weights, types):
        aids_temp[aid] += w * type_weight_multipliers[t]
    # session长度40已经可以涵盖90%的数据了
    history_aids = [k for k, v in aids_temp.most_common()]
    type_1 = [1] * len(history_aids)
    scores_1 = [v for k, v in aids_temp.most_common()]
    if len(set(scores_1)) == 1:
        scores_1 = [1] * len(scores_1)
    else:
        min_ = min(scores_1)
        max_ = max(scores_1)
        scores_1 = [(j - min_) / (max_ - min_) for j in scores_1]

    # 相似度矩阵召回
    # USE "CLICKS" CO-VISITATION MATRIX
    # click矩阵只考虑了时间的因素，cart-orders还考虑了相似商品的类别
    # 这里可以修改，通过sorted_aids召回相似物品 ---------------sort <= unique
    aids2 = list(itertools.chain(*[top_20_clicks[aid] for aid in history_aids if aid in top_20_clicks]))
    aids3 = list(itertools.chain(*[top_20_buys[aid] for aid in history_aids if aid in top_20_buys]))
    # RERANK CANDIDATES  Counter计数筛选，不管得分，历史序列优先
    # 融合aids2和aids3的信息，同时考虑了相似item的时间权重和类型权重
    sim_aids_100 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(100)]
    type_2 = [2] * len(sim_aids_100)
    scores_2 = [cnt for aid2, cnt in Counter(aids2 + aids3).most_common(100)]

    # 基于规则召回n个
    # 热门商品召回100个,类别加权
    top_hot_items_100 = list(top_hot_items.keys())[:100]
    type_3 = [3] * (len(top_hot_items_100))
    score_3 = list(top_hot_items.values())[:100]
    # 点击最多的商品召回100个
    top_clicks_100 = list(top_clicks.keys())[:100]
    type_4 = [4] * (len(top_clicks_100))
    score_4 = list(top_clicks.values())[:100]
    # 过去一个月点击最多的商品召回100个
    top_clicks_last_month_100 = list(top_clicks_items_last_month.keys())[:100]
    type_5 = [5] * (len(top_clicks_last_month_100))
    score_5 = list(top_clicks_items_last_month.values())[:100]
    # 过去一个月热度最高的100个商品
    top_hot_items_one_month_100 = list(top_hot_items_last_month.keys())[:100]
    type_6 = [6] * (len(top_hot_items_one_month_100))
    score_6 = list(top_hot_items_last_month.values())[:100]

    # 基于向量embedding召回160个
    # 基于最后一周deepwalk召回80个
    temp_counter = Counter()
    for i in history_aids:
        if f'item_{i}' in word2vec_last_week:
            for j in word2vec_last_week.similar_by_word(f'item_{i}', topn=20):
                temp_counter[j[0]] += j[1]
    item_emb_deepwalk_last_week_80 = [int(aid2.split('_')[1]) for aid2, cnt in temp_counter.most_common(80)]
    type_7 = [7] * len(item_emb_deepwalk_last_week_80)
    score_7 = [cnt for aid2, cnt in temp_counter.most_common(80)]

    # 基于全局deepwalk召回80个
    temp_counter1 = Counter()
    for i in history_aids:
        for j in word2vec_last_month.similar_by_word(f'item_{i}', topn=20):
            temp_counter1[j[0]] += j[1]
    item_emb_last_month_80 = [int(aid2.split('_')[1]) for aid2, cnt in temp_counter1.most_common(80)]
    type_8 = [8] * len(item_emb_last_month_80)
    score_8 = [cnt for aid2, cnt in temp_counter1.most_common(80)]
    # print(item_emb_deepwalk_last_week_80[0], score_7[0], item_emb_last_month_80[0], score_8[0])

    result = history_aids + sim_aids_100 + top_hot_items_100 + top_clicks_100 + top_clicks_last_month_100 + \
             top_hot_items_one_month_100 + item_emb_deepwalk_last_week_80 + item_emb_last_month_80

    type = type_1 + type_2 + type_3 + type_4 + type_5 + type_6 + type_7 + type_8
    score = scores_1 + scores_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8

    info = [str(result[i]) + "#" + str(type[i]) + "#" + str(score[i]) for i in range(len(result))]

    return info


def suggest_carts(df):
    # User history aids and types
    aids = df.aid.tolist()
    types = df.type.tolist()

    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df['type'] == 0) | (df['type'] == 1)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

    # Rerank candidates using weights，时间weight？
    # 等比数列 2**0.5[0.414] -- 2**1-1，要突出以往carts和orders的权重，时间权重不能过小
    weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
    aids_temp = Counter()

    # Rerank based on repeat items and types of items
    # 使用aids信息召回
    for aid, w, t in zip(aids, weights, types):  # w: 0.414-1  types:1,5,4  min 0.414 max 5
        aids_temp[aid] += w * type_weight_multipliers[t]
    # 不直接召回，下面利用矩阵信息再算一次
    # Rerank candidates using"top_20_carts" co-visitation matrix
    # 基于buy2buys召回carts 用unique_buys召回carts
    # aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in unique_buys if aid in top_20_buys]))
    # aids2 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
    # 将buy2buy矩阵输出 +0.1,
    # 还是以历史序列为主，尽量不要超过历史权重的量级，0.1算是合理
    # for aid in aids2: aids_temp[aid] += 0.1
    history_aids = [k for k, v in aids_temp.most_common()]
    type_1 = [1] * len(history_aids)
    scores_1 = [v for k, v in aids_temp.most_common()]
    if len(set(scores_1)) == 1:
        scores_1 = [1] * len(scores_1)
    else:
        min_ = min(scores_1)
        max_ = max(scores_1)
        scores_1 = [(j - min_) / (max_ - min_) for j in scores_1]
        # print(scores_1[1])

    # Use "cart order" and "clicks" co-visitation matrices
    # click时间序列召回  基于历史session，要考虑时间，召回最新的
    aids1 = list(itertools.chain(*[top_20_clicks[aid] for aid in history_aids if aid in top_20_clicks]))
    # carts-orders召回  这里通过aids召回，使用buys也情有可原
    # 使用点击召回carts
    aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in history_aids if aid in top_20_buys]))
    # 修改5：基于unique_buys召回carts，要考虑carts-orders，那么使用buy2buy
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))

    # RERANK CANDIDATES
    sim_aids_100 = [aid2 for aid2, cnt in Counter(aids1 + aids2 + aids3).most_common(100) if aid2 not in history_aids]
    type_2 = [2] * len(sim_aids_100)
    scores_2 = [cnt for aid2, cnt in Counter(aids1 + aids2 + aids3).most_common(100) if aid2 not in history_aids]

    # 基于规则召回200个
    # 热门商品召回100个,类别加权
    top_hot_items_100 = list(top_hot_items.keys())[:100]
    type_3 = [3] * (len(top_hot_items_100))
    score_3 = list(top_hot_items.values())[:100]
    # 购买最多的商品召回100个
    top_orders_100 = list(top_orders.keys())[:100]
    type_4 = [4] * (len(top_orders_100))
    score_4 = list(top_orders.values())[:100]
    # 加购最多的商品召回100个
    top_carts_100 = list(top_carts.keys())[:100]
    type_5 = [5] * (len(top_carts_100))
    score_5 = list(top_carts.values())[:100]
    # 过去一个月热度最高的100个商品
    top_hot_items_one_month_100 = list(top_hot_items_last_month.keys())[:150]
    type_6 = [6] * (len(top_hot_items_one_month_100))
    score_6 = list(top_hot_items_last_month.values())[:150]

    # 基于向量embedding召回160个
    # 基于最后一周deepwalk召回80个
    temp_counter = Counter()
    for i in history_aids:
        if f'item_{i}' in word2vec_last_week:
            for j in word2vec_last_week.similar_by_word(f'item_{i}', topn=20):
                temp_counter[j[0]] += j[1]
    item_emb_deepwalk_last_week_80 = [int(aid2.split('_')[1]) for aid2, cnt in temp_counter.most_common(80)]
    type_7 = [7] * len(item_emb_deepwalk_last_week_80)
    score_7 = [cnt for aid2, cnt in temp_counter.most_common(80)]

    # 基于全局deepwalk召回80个
    temp_counter1 = Counter()
    for i in history_aids:
        for j in word2vec_last_month.similar_by_word(f'item_{i}', topn=20):
            temp_counter1[j[0]] += j[1]
    item_emb_last_month_80 = [int(aid2.split('_')[1]) for aid2, cnt in temp_counter1.most_common(80)]
    type_8 = [8] * len(item_emb_last_month_80)
    score_8 = [cnt for aid2, cnt in temp_counter1.most_common(80)]
    print(item_emb_deepwalk_last_week_80[0], score_7[0], item_emb_last_month_80[0], score_8[0])

    result = history_aids + sim_aids_100 + top_hot_items_100 + top_orders_100 + top_carts_100 + \
             top_hot_items_one_month_100 + item_emb_deepwalk_last_week_80 + item_emb_last_month_80

    type = type_1 + type_2 + type_3 + type_4 + type_5 + type_6 + type_7 + type_8
    score = scores_1 + scores_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8

    info = [str(result[i]) + "#" + str(type[i]) + "#" + str(score[i]) for i in range(len(result))]

    return info


def suggest_buys(df):
    # USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    # UNIQUE AIDS AND UNIQUE BUYS
    # unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df['type'] == 1) | (df['type'] == 2)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

    # 基于历史序列召回40个
    # RERANK CANDIDATES USING WEIGHTS
    # 等比数列 0.414-1
    weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
    aids_temp = Counter()
    # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
    for aid, w, t in zip(aids, weights, types):
        aids_temp[aid] += w * type_weight_multipliers[t]
    # 直接取40，不管够不够，不够的话就这样
    history_aids = [k for k, v in aids_temp.most_common()]
    type_1 = [1] * len(history_aids)
    scores_1 = [v for k, v in aids_temp.most_common()]
    if len(set(scores_1)) == 1:
        scores_1 = [1] * len(scores_1)
    else:
        min_ = min(scores_1)
        max_ = max(scores_1)
        scores_1 = [(j - min_) / (max_ - min_) for j in scores_1]

    # 基于co—visitation召回100个
    # USE "CART ORDER" CO-VISITATION MATRIX  用aids召回orders，对的！
    aids2 = list(itertools.chain(*[top_20_buys[aid] for aid in history_aids if aid in top_20_buys]))
    # USE "BUY2BUY" CO-VISITATION MATRIX   用unique_buys召回orders，对的！！
    aids3 = list(itertools.chain(*[top_20_buy2buy[aid] for aid in unique_buys if aid in top_20_buy2buy]))
    # RERANK CANDIDATES

    sim_aids_100 = [aid2 for aid2, cnt in Counter(aids2 + aids3).most_common(100)]
    type_2 = [2] * len(sim_aids_100)
    scores_2 = [cnt for aid2, cnt in Counter(aids2 + aids3).most_common(100)]

    # 基于规则召回n个
    # 热门商品召回100个,类别加权
    top_hot_items_100 = list(top_hot_items.keys())[:100]
    type_3 = [3] * (len(top_hot_items_100))
    score_3 = list(top_hot_items.values())[:100]
    # 购买最多的商品召回100个
    top_orders_100 = list(top_orders.keys())[:100]
    type_4 = [4] * (len(top_orders_100))
    score_4 = list(top_orders.values())[:100]
    # 加购最多的商品召回100个
    top_carts_100 = list(top_carts.keys())[:100]
    type_5 = [5] * (len(top_carts_100))
    score_5 = list(top_carts.values())[:100]
    # 过去一个月热度最高的100个商品
    top_hot_items_one_month_100 = list(top_hot_items_last_month.keys())[:100]
    type_6 = [6] * (len(top_hot_items_one_month_100))
    score_6 = list(top_hot_items_last_month.values())[:100]

    # 基于向量embedding召回160个
    # 基于最后一周deepwalk召回80个
    temp_counter = Counter()
    for i in history_aids:
        if f'item_{i}' in word2vec_last_week:
            for j in word2vec_last_week.similar_by_word(f'item_{i}', topn=20):
                temp_counter[j[0]] += j[1]
    item_emb_deepwalk_last_week_80 = [int(aid2.split('_')[1]) for aid2, cnt in temp_counter.most_common(80)]
    type_7 = [7] * len(item_emb_deepwalk_last_week_80)
    score_7 = [cnt for aid2, cnt in temp_counter.most_common(80)]

    # 基于全局deepwalk召回80个
    temp_counter1 = Counter()
    for i in history_aids:
        for j in word2vec_last_month.similar_by_word(f'item_{i}', topn=20):
            temp_counter1[j[0]] += j[1]
    item_emb_last_month_80 = [int(aid2.split('_')[1]) for aid2, cnt in temp_counter1.most_common(80)]
    type_8 = [8] * len(item_emb_last_month_80)
    score_8 = [cnt for aid2, cnt in temp_counter1.most_common(80)]
    print(item_emb_deepwalk_last_week_80[0], score_7[0], item_emb_last_month_80[0], score_8[0])

    result = history_aids + sim_aids_100 + top_hot_items_100 + top_orders_100 + top_carts_100 + \
             top_hot_items_one_month_100 + item_emb_deepwalk_last_week_80 + item_emb_last_month_80

    type = type_1 + type_2 + type_3 + type_4 + type_5 + type_6 + type_7 + type_8
    score = scores_1 + scores_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8

    info = [str(result[i]) + "#" + str(type[i]) + "#" + str(score[i]) for i in range(len(result))]

    return info


print("开始进行clicks推荐！！！")

pred_df_clicks = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
    lambda x: suggest_clicks(x)
)
print("开始进行carts推荐！！！")
pred_df_carts = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
    lambda x: suggest_carts(x)
)
print("开始进行buys推荐！！！")
pred_df_buys = test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
    lambda x: suggest_buys(x)
)

print("开始进行推荐！！！")
clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
orders_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()
carts_pred_df = pd.DataFrame(pred_df_carts.add_suffix("_carts"), columns=["labels"]).reset_index()

pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
pred_df.columns = ["session_type", "labels"]
pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
pred_df.to_parquet(f'/home/niejianfei/otto/{stage}/candidates/candidates.pqt')
print(pred_df)


print("开始计算recall！！！")
score = 0
recall_score = {}
weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
for t in ['clicks', 'carts', 'orders']:
    sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
    sub['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
    # sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')])
    sub['labels'] = sub['labels'].apply(lambda x: [int(i.split('#')[0]) for i in x.split(' ')])
    test_labels = pd.read_parquet(f'/home/niejianfei/otto/CV/preprocess/test_labels.parquet')
    test_labels = test_labels.loc[test_labels['type'] == t]
    test_labels = test_labels.merge(sub, how='left', on=['session'])
    test_labels['hits'] = test_labels.apply(
        lambda df: min(20, len(set(df.ground_truth).intersection(set(df.labels)))), axis=1)
    # 设定阈值 长度多于20，定为20
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0, 20)
    recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
    recall_score[t] = recall
    score += weights[t] * recall
    print(f'{t} recall =', recall)

print('=============')
print('Overall Recall =', score)
print('=============')

# handcraft recall LB，0.577
# 开始计算recall！！！
# clicks recall = 0.5257653796508641
# carts recall = 0.41246734503419014
# orders recall = 0.6498501450672353
# =============
# Overall Recall = 0.5662268285156846
# =============

# 开始计算recall@170！！！
# clicks recall = 0.6012911171187798
# carts recall = 0.5011587525716328
# orders recall = 0.7053682856531855
# =============
# Overall Recall = 0.6336977088752791
# =============
