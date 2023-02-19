import glob
import pickle
import pandas as pd
import numpy as np
IS_TRAIN = True


def load_data(path):
    type_transform = {"clicks": 0, "carts": 1, "orders": 2}
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_transform).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


# 加载数据
print('加载数据')

if IS_TRAIN:
    train_sessions = load_data('/home/niejianfei/otto/CV/data/*_parquet/*')
else:
    train_sessions = load_data('/home/niejianfei/otto/LB/data/*_parquet/*')
print(train_sessions)

dic = pd.DataFrame(train_sessions.drop_duplicates(['aid']).sort_values(by='aid', ascending=True)['aid'])
dic['num'] = range(len(dic))
dic.index = dic['aid']
dic = dic.drop(columns='aid').to_dict()['num']
# print(dic)

# 保存矩阵到本地
if IS_TRAIN:
    f_save = open('/home/niejianfei/otto/CV/preprocess/aid_num_dict.pkl', 'wb')
    pickle.dump(dic, f_save)
    f_save.close()
else:
    f_save = open('/home/niejianfei/otto/LB/preprocess/aid_num_dict.pkl', 'wb')
    pickle.dump(dic, f_save)
    f_save.close()
print("aid_num映射保存完毕！！！")


def generate_pairs(df):
    df = df.sort_values(by=['session', 'ts'])
    print(df)
    df['aid'] = df['aid'].map(dic)
    print(df)

    print('count 1')
    df['session_count'] = df['session'].map(df['session'].value_counts())
    print(df)
    df1 = df[df['session_count'] == 1]
    df = df.append(df1)
    print('count 2')
    df['session_count'] = df['session'].map(df['session'].value_counts())
    print(df['session_count'].min())
    print(df)

    df = df.sort_values(by=['session', 'ts'])
    df['ranking'] = df.groupby(['session'])['ts'].rank(method='first', ascending=True)
    print(df)
    df['aid_next'] = df['aid'].shift(-1)
    print(df)
    df = df.query('session_count!=ranking').reset_index(drop=True)

    df['aid_next'] = df['aid_next'].astype('int32')
    print(df)
    df = df[['aid', 'aid_next']]
    print(df)
    pairs_list = np.array(df)
    return pairs_list


pairs_list = generate_pairs(train_sessions).tolist()
print(pairs_list[:10])

if IS_TRAIN:
    f = open('/home/niejianfei/otto/CV/preprocess/session_pairs.ungraph', "w")
    for line in pairs_list:
        f.write(str(line[0]) + ' ' + str(line[1]) + '\n')
    f.close()
else:
    f = open('/home/niejianfei/otto/LB/preprocess/session_pairs.ungraph', "w")
    for line in pairs_list:
        f.write(str(line[0]) + ' ' + str(line[1]) + '\n')
    f.close()
