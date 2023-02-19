from features import recall_features
from features import user_item_features
from features import similarity_features
from features import co_visitation_features
import pandas as pd


def add_labels(candidate_type):
    targets = pd.read_parquet('/home/niejianfei/otto/CV/preprocess/test_labels.parquet')
    for t in candidate_type:
        print("给data加标签！！！")
        # 加标签
        temp_target = targets[targets['type'] == t].drop(columns="type")
        temp_target = temp_target.explode("ground_truth").astype("int32")
        temp_target.columns = ['session', 'aid']
        temp_target[t[0:-1]] = 1

        # 只导入CV数据
        print('开始导入数据')
        for i in range(0, 8):
            path = f"/home/niejianfei/otto/CV/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
            print(f'第{i + 1}块数据')
            chunk = pd.read_parquet(path)
            print(path)
            print(chunk.columns)
            # 加标签，负类标0
            chunk = chunk.merge(temp_target, ['session', 'aid'], how='left').fillna(value=0)
            print(chunk)
            chunk.to_parquet(path)


if __name__ == '__main__':
    IS_TRAIN = True
    candidate_type = ['clicks', 'carts', 'orders']
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'

    recall_features(stage, candidate_type)
    user_item_features(stage, candidate_type)
    similarity_features(stage, candidate_type, 0, 8)
    co_visitation_features(stage, candidate_type, 0, 8)
    # 给CV加标签
    if IS_TRAIN:
        add_labels(candidate_type)