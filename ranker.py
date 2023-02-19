import glob
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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


def load_train_data(t, semi_sessions):
    dfs = []
    # 只导入训练数据
    k = 0
    print('开始导入数据')
    for i in range(0, 8):
        path = f"/home/niejianfei/otto/CV/candidates/candidates_{t[0:-1]}_features_data/candidate_{t[0:-1]}_{i}.pqt"
        print(f'第{i + 1}块数据')
        chunk = pd.read_parquet(path)
        print(path)
        print(chunk)

        train = chunk[chunk.session.isin(semi_sessions)]

        chunk = train.astype("float32")
        chunk['session'] = chunk['session'].astype('int32')
        chunk['aid'] = chunk['aid'].astype('int32')

        chunk_pos = chunk[chunk[t[0:-1]] == 1].sort_values(by='session', ascending=True)
        print('正类', len(chunk_pos))
        chunk_neg = chunk[chunk[t[0:-1]] == 0].sample(len(chunk_pos) * 30, random_state=random_state)
        chunk = chunk_neg.append(chunk_pos).sort_values(by='session', ascending=True)
        dfs.append(chunk)
        print(len(chunk))
        k += len(chunk_pos)
    print(f'正类一共有：', k)
    return pd.concat(dfs).reset_index(drop=True)


# 训练
def train_xgb(candidate_type, semi_sessions, describe):
    for t in candidate_type:
        candidates = load_train_data(t, semi_sessions)
        print(candidates)

        # 训练
        candidates = candidates.sort_values(by='session', ascending=True)
        FEATURES = candidates.columns[0:-1]
        print(FEATURES)

        skf = GroupKFold(n_splits=5)
        for fold, (train_idx, valid_idx) in enumerate(
                skf.split(candidates, candidates[t[0:-1]], groups=candidates['session'])):
            # loc: 标签索引
            X_train_ = candidates.loc[train_idx, FEATURES]
            X_train = X_train_.drop(columns=['session', 'aid'])
            y_train = candidates.loc[train_idx, t[0:-1]]

            X_valid_ = candidates.loc[valid_idx, FEATURES]
            X_valid = X_valid_.drop(columns=['session', 'aid'])
            y_valid = candidates.loc[valid_idx, t[0:-1]]

            groups1 = X_train_.groupby('session').aid.agg('count').values
            groups2 = X_valid_.groupby('session').aid.agg('count').values
            # 读取数据,每个user一起训练
            # DMatrix是XGBoost使用的内部数据结构，它针对内存效率和训练速度进行了优化
            dtrain = xgb.DMatrix(X_train, y_train, group=groups1)
            dvalid = xgb.DMatrix(X_valid, y_valid, group=groups2)
            # 就当成是一组
            # dtrain = xgb.DMatrix(X_train, y_train)
            # dvalid = xgb.DMatrix(X_valid, y_valid)

            xgb_parms = {'booster': 'gbtree',
                         'tree_method': 'gpu_hist',
                         'objective': 'binary:logistic',
                         'eta': 0.01,
                         'eval_metric': 'logloss',
                         'seed': 0,
                         # 'early_stopping_rounds': 300,
                         # 'subsample': 0.5,
                         # 'colsample_bytree': 0.5,
                         # 'max_depth': 3,
                         # 'reg_alpha': 1,
                         'reg_lambda': 20,
                         'scale_pos_weight': 30}

            model = xgb.train(xgb_parms,
                              dtrain=dtrain,
                              evals=[(dtrain, 'train'), (dvalid, 'valid')],
                              num_boost_round=3000,
                              verbose_eval=100,
                              )

            print(f"第{fold + 1}次开始输出模型指标!!!")
            name = 'XGB'
            dtrain1 = xgb.DMatrix(X_train)
            dtest1 = xgb.DMatrix(X_valid)

            def sigmoid(x):
                return 1. / (1 + np.exp(-x))

            y_train_pred_pre = np.array(model.predict(dtrain1))
            # y_train_pred_pre = sigmoid(y_train_pred_pre)
            print(y_train_pred_pre[:10])
            y_train_pred = np.array(y_train_pred_pre)

            y_train_pred[y_train_pred >= 0.5] = int(1)
            y_train_pred[y_train_pred < 0.5] = int(0)
            print(y_train_pred[:10])

            y_test_pred_pre = np.array(model.predict(dtest1))
            # y_test_pred_pre = sigmoid(y_test_pred_pre)
            y_test_pred = np.array(y_test_pred_pre)

            y_test_pred[y_test_pred >= 0.5] = int(1)
            y_test_pred[y_test_pred < 0.5] = int(0)

            # accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_valid, y_test_pred)

            # precision
            train_precision = precision_score(y_train, y_train_pred)
            test_precision = precision_score(y_valid, y_test_pred)
            # recall
            train_recall = recall_score(y_train, y_train_pred)
            test_recall = recall_score(y_valid, y_test_pred)
            # f1
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_valid, y_test_pred)
            # auc 计算时，计算的应该是不同的概率画出来的曲线下的面积,而不是预测值对应的曲线下的面积

            train_auc = roc_auc_score(y_train, y_train_pred_pre)
            test_auc = roc_auc_score(y_valid, y_test_pred_pre)

            print('{}  训练集： accuracy:{:.3},precision:{:.3}, recall:{:.3}, f1:{:.3}, auc:{:.3}'.format(name,
                                                                                                      train_accuracy,
                                                                                                      train_precision,
                                                                                                      train_recall,
                                                                                                      train_f1,
                                                                                                      train_auc))
            print(
                '{}  验证集： accuracy:{:.3},precision:{:.3}, recall:{:.3}, f1:{:.3}, auc:{:.3}'.format(name, test_accuracy,
                                                                                                    test_precision,
                                                                                                    test_recall,
                                                                                                    test_f1,
                                                                                                    test_auc))
            importance_weight = model.get_score(fmap='', importance_type='weight')
            print('weight', importance_weight)
            importance_gain = model.get_score(fmap='', importance_type='gain')
            print('gain', importance_gain)

            model.save_model(f'/home/niejianfei/otto/CV/models/xgb_fold{fold}_{t[0:-1]}_{describe}.xgb')


# 预测
def xgb_inference(key, stage, t, semi_sessions, describe):
    fold_num = 5
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(
            glob.glob(f"/home/niejianfei/otto/{stage}/candidates/candidates_{t[0:-1]}_features_data/*")):
        print(f"第{e + 1}块数据！！！")

        chunk = pd.read_parquet(chunk_file)
        print(chunk)
        print(chunk.columns)

        if stage == 'CV':
            x_train = chunk[chunk.session.isin(semi_sessions)].astype("float32")
            x_test = chunk[~chunk.session.isin(semi_sessions)].astype("float32")

            if key == 'train':
                chunk = x_train.astype("float32")
                if len(chunk) == 0:
                    continue
            else:
                chunk = x_test.astype("float32")
                if len(chunk) == 0:
                    continue
        print(f'{key}长度为', len(chunk))
        FEATURES = chunk.columns[2:-1]
        chunk['session'] = chunk['session'].astype('int32')
        chunk['aid'] = chunk['aid'].astype('int32')

        preds = np.zeros(len(chunk))
        for fold in range(fold_num):
            print(f"第{fold + 1}次预测！！！")

            model = xgb.Booster()
            model.load_model(
                f'/home/niejianfei/otto/CV/models/xgb_fold{fold}_{t[0:-1]}_{describe}.xgb')
            model.set_param({'predictor': 'gpu_predictor'})
            print("开始构建test数据集！！！")
            dtest = xgb.DMatrix(data=chunk[FEATURES])
            print("开始预测！！！")
            preds += model.predict(dtest) / fold_num
            print(preds.max())
        print(f"第{e + 1}次构建predictions！！！")
        predictions = chunk[['session', 'aid']].copy()
        predictions['pred'] = preds
        print(predictions[:10])
        dfs.append(predictions)
    return pd.concat(dfs, axis=0).reset_index(drop=True)


def generate_submission(key, stage, candidate_type, semi_sessions, describe):
    for t in candidate_type:
        predictions = xgb_inference(key, stage, t, semi_sessions, describe)

        print("开始构造submission！！！")
        predictions = predictions.sort_values(['session', 'pred'], ascending=[True, False]).reset_index(
            drop=True).drop_duplicates(['session', 'aid'], keep='first')
        predictions['n'] = predictions.groupby('session').aid.cumcount().astype('int32')
        print(predictions[:200])
        print(f"开始筛选<20！！！")
        predictions1 = predictions[predictions['n'] < 20]
        print(predictions1[:20])

        sub = predictions1.groupby('session').aid.apply(list)
        sub = sub.to_frame().reset_index()
        sub.aid = sub.aid.apply(lambda x: " ".join(map(str, x)))
        sub.columns = ['session_type', 'labels']
        sub.session_type = sub.session_type.astype('str') + f'_{t}'
        print(len(sub))
        print("开始写入本地！！！")
        sub.to_parquet(f'/home/niejianfei/otto/{stage}/submission/sub_{t}.pqt')


def get_recall(key, candidate_type):
    for t in candidate_type:
        print("开始读取数据！！！")
        pred_df = pd.read_parquet(f'/home/niejianfei/otto/CV/submission/sub_{t}.pqt')
        print(len(pred_df))

        sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
        sub['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
        sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')])
        print("开始读取labels！！！")
        test_labels = pd.read_parquet(f'/home/niejianfei/otto/CV/preprocess/test_labels.parquet')
        print(len(test_labels))
        print(len(pred_df) - len(pred_df))
        test_labels = test_labels.loc[test_labels['type'] == t]
        test_labels = test_labels.merge(sub, how='left', on=['session'])
        test_labels['hits'] = test_labels.apply(
            lambda df: min(20, len(set(df.ground_truth).intersection(set(df.labels)))), axis=1)
        # 设定阈值 长度多于20，定为20
        test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0, 20)
        print(f"开始计算{key}recall！！！")
        recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
        print(f'{key} {t} recall@20 =', recall)


def user_sample(frac):
    return valid.drop_duplicates(['session']).sample(frac=frac, random_state=random_state)['session']


if __name__ == '__main__':
    # 抽取一半session计算recall
    random_state = 33
    valid = load_data(f'/home/niejianfei/otto/CV/data/test_parquet/*')

    candidate_type = ['clicks', 'carts', 'orders']
    describe = 'final'

    train_xgb(candidate_type, user_sample(0.5), describe)
    generate_submission('test', 'CV', candidate_type, user_sample(0.5), describe)
    get_recall('test', candidate_type)

    # 使用全量数据训练模型做最终预测
    train_xgb(candidate_type, user_sample(1), 'final_all_data')
