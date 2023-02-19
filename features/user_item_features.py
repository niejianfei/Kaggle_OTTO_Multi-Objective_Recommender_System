import glob
import pandas as pd
type_transform = {"clicks": 0, "carts": 1, "orders": 2}
IS_TRAIN = True


def load_data(path):
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_transform).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


def user_item_features(stage, candidate_type):
    valid = load_data(f'/home/niejianfei/otto/{stage}/data/test_parquet/*')
    for t in candidate_type:
        print('读取candidates！！！')
        candidates = pd.read_parquet(f'/home/niejianfei/otto/{stage}/candidates/candidates_{t}.pqt').reset_index(
            drop=True)
        candidates = candidates.sort_values('session', ascending=True)
        print(candidates)

        print("开始构造user_item interaction features！！！")
        # 构造user_item interaction features
        print("click！！！")
        # item是否被点击
        item_clicked = valid[valid["type"] == 0].drop(columns="ts").drop_duplicates(["session", "aid"])
        item_clicked["type"] = 1
        item_clicked.columns = ["session", "aid", "item_clicked"]
        # item_clicked 特征
        item_clicked_features = valid[valid["type"] == 0].groupby(['session', 'aid']).agg(
            {'aid': 'count'})
        item_clicked_features.columns = ['item_clicked_num']
        item_clicked_features = item_clicked_features.astype('float32')

        print("cart！！！")
        # item是否被加购
        item_carted = valid[valid["type"] == 1].drop(columns="ts").drop_duplicates(["session", "aid"])
        item_carted["type"] = 1
        item_carted.columns = ["session", "aid", "item_carted"]
        # item_carted 特征
        item_carted_features = valid[valid["type"] == 1].groupby(['session', 'aid']).agg(
            {'aid': 'count'})
        item_carted_features.columns = ['item_carted_num']
        item_carted_features = item_carted_features.astype('float32')
        print("order！！！")

        # item是否被购买
        item_ordered = valid[valid["type"] == 2].drop(columns="ts").drop_duplicates(["session", "aid"])
        item_ordered["type"] = 1
        item_ordered.columns = ["session", "aid", "item_ordered"]
        # item_ordered 特征
        item_ordered_features = valid[valid["type"] == 2].groupby(['session', 'aid']).agg(
            {'aid': 'count'})
        item_ordered_features.columns = ['item_ordered_num']
        item_ordered_features = item_ordered_features.astype('float32')

        print("开始聚合数据！！！")
        item_features = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/item_features.pqt')

        chunk = 8
        size = candidates.shape[0] + 200
        print(f"candidates有{candidates.shape[0]}条数据！！！")
        # 距离session结束的时间sec, 需要ts merge到candidate上然后减去min_ts
        # 去重，保留最后一个ts，merge 相减 加两列特征距离session结束的时间sec，和最后一次和aid交互的类型
        valid = valid.drop_duplicates(['session', 'aid'], keep='last').drop(columns='type')
        # valid['user_item_within'] = 1
        print(valid)

        user_features = pd.read_parquet(f'/home/niejianfei/otto/{stage}/preprocess/user_features.pqt')
        valid = valid.merge(user_features, left_on='session', right_index=True, how='left').fillna(-1000)

        valid['sec_to_session_start'] = valid['ts'] - valid['user_min_ts']
        valid['sec_to_session_end'] = valid['user_max_ts'] - valid['ts']
        valid = valid.drop(columns=['user_min_ts', 'user_max_ts', 'ts'])

        val_session = valid[['sec_to_session_start', 'sec_to_session_end', 'user_long_ts']]
        print(val_session)
        print((val_session['sec_to_session_start'] + val_session['sec_to_session_end'] - val_session['user_long_ts']).max())

        k = size // chunk
        t = 0
        for i in range(chunk):
            print(f"第{i + 1}块！！！")
            print("1！！！")
            temp_candidates = candidates.iloc[k * i:k * (i + 1), :]
            print(temp_candidates)
            # merge user_item interaction features
            temp_candidates = temp_candidates.merge(item_clicked, how="left", on=["session", "aid"]).fillna(value=-1)
            temp_candidates = temp_candidates.merge(item_clicked_features, how="left", on=["session", "aid"]).fillna(
                value=-1)
            print(temp_candidates)
            print("2！！！")
            temp_candidates = temp_candidates.merge(item_carted, how="left", on=["session", "aid"]).fillna(value=-1)
            temp_candidates = temp_candidates.merge(item_carted_features, how="left", on=["session", "aid"]).fillna(
                value=-1)
            print("3！！！")
            temp_candidates = temp_candidates.merge(item_ordered, how="left", on=["session", "aid"]).fillna(value=-1)
            temp_candidates = temp_candidates.merge(item_ordered_features, how="left", on=["session", "aid"]).fillna(
                value=-1)
            print(temp_candidates)
            print("开始读取聚合item_features！！！")
            # Step 5：add features to our candidate dataframe
            temp_candidates = temp_candidates.merge(item_features, left_on='aid', right_index=True, how='left').fillna(
                -1000)

            # 加入交互特征
            temp_candidates = temp_candidates.merge(valid, on=["session", "aid"], how='left').fillna(-1)
            print(temp_candidates)

            temp_candidates.to_parquet(
                f"/home/niejianfei/otto/{stage}/candidates/candidates_{candidate_type[0:-1]}_features_data/candidate_{candidate_type[0:-1]}_{i}.pqt")
            print(temp_candidates)
            t += len(temp_candidates)
            print(f'第{i+1}块数据量:', len(temp_candidates))
        print('数据总量:', t)


if __name__ == '__main__':
    IS_TRAIN = True
    candidate_type = ['clicks', 'carts', 'orders']
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'

    user_item_features(stage, candidate_type)