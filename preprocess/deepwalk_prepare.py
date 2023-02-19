import glob
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from gensim.models import Word2Vec
import numpy as np

type_transform = {"clicks": 0, "carts": 1, "orders": 2}
IS_TRAIN = True
IS_Last_Month = True


def load_data(path):
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        # if not IS_TRAIN:
        #     # 除去第一周的数据
        #     chunk = chunk[chunk['ts'] >= 1659909599]
        chunk['type'] = chunk['type'].map(type_transform).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


# 加载数据
print('加载数据')
if IS_TRAIN:
    if IS_Last_Month:
        train_sessions = load_data('/home/niejianfei/otto/CV/data/*_parquet/*')
        print(train_sessions)
    else:
        train_sessions = load_data('/home/niejianfei/otto/CV/data/test_parquet/*')
        print(train_sessions)
else:
    if IS_Last_Month:
        train_sessions = load_data('/home/niejianfei/otto/LB/data/*_parquet/*')
        print(train_sessions)
    else:
        train_sessions = load_data('/home/niejianfei/otto/LB/data/test_parquet/*')
        print(train_sessions)

print('开始排序')
# 分别对session_id聚合，对时间进行排序
df = train_sessions.sort_values(by=["session", "ts"], ascending=True)
print(df.head(10))

print('开始构图')
# 开始构图
dic = defaultdict(list)  # defaultdict为了给key不在字典的情况赋予一个default值
# 加文字是区分item和user
for x in tqdm(df[["session", "aid"]].values):
    dic[f"user_{x[0]}"].append(f"item_{x[1]}")  # list中元素是有顺序的
    dic[f"item_{x[1]}"].append(f"user_{x[0]}")

# 随机游走
print('开始随机游走')
# 中心点item，先选定一个session，再走到session中item后面的元素中
# 计算user item对应长度
dic_count = {}
for key in dic:
    dic_count[key] = len(dic[key])

item_list = df["aid"].unique()
user_list = df["session"].unique()
print('item数量', len(item_list))
print('user数量', len(user_list))

path_length = 20
sentences = []
num_sentences = 20000000  # 实际跑的时候建议50w+ (有2w个item)
'''
badcase:
    item_a : session_1
    session_1 : [item_b,item_a]
需要加一个max_repeat_time 避免死循环
'''

max_repeat_nums = path_length * 2
for _ in tqdm(range(num_sentences)):
    start_item = "item_{}".format(item_list[np.random.randint(0, len(item_list))])
    sentence = [start_item]
    repeat_time = 0
    while len(sentence) < path_length:
        last_item = sentence[-1]
        random_user = dic[last_item][np.random.randint(0, dic_count[last_item])]  # 递归，选最后一个得到user列表，再选一个user
        # 若两个相同的item紧挨着，则+1后跳到下一个，继续session随机可能跳出来，其实图也有这种情况，闭环的产生
        next_item_index = np.where(np.array(dic[random_user]) == last_item)[0][
                              0] + 1  # 在random_user的items里面找到last_item的索引+1
        # user内item不是最后一个，把后面这个加过去
        # 若是最后一个，不做操作继续循环，可能有bad case
        if next_item_index <= dic_count[random_user] - 1:
            next_item = dic[random_user][next_item_index]
            sentence.append(next_item)
        repeat_time += 1
        if repeat_time > max_repeat_nums:
            break
    sentences.append(sentence)

# embedding_dimensions = number_of_categories**0.25
model = Word2Vec(sentences, vector_size=64, sg=1, window=5, min_count=1, hs=1, negative=5, sample=0.001, workers=4)
# 保存模型
if IS_TRAIN:
    if IS_Last_Month:
        model.wv.save_word2vec_format('/home/niejianfei/otto/CV/preprocess/deepwalk_last_month.w2v', binary=False)
    else:
        model.wv.save_word2vec_format('/home/niejianfei/otto/CV/preprocess/deepwalk_last_week.w2v', binary=False)
else:
    if IS_Last_Month:
        model.wv.save_word2vec_format('/home/niejianfei/otto/LB/preprocess/deepwalk_last_month.w2v', binary=False)
    else:
        model.wv.save_word2vec_format('/home/niejianfei/otto/LB/preprocess/deepwalk_last_week.w2v', binary=False)
