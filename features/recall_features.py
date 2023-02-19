import pandas as pd


# 导入candidates数据
def recall_features(stage, candidate_type):
    type_transform = {"clicks": 0, "carts": 1, "orders": 2}
    print("开始导入数据！！！")
    candidates = pd.read_parquet(f'/home/niejianfei/otto/{stage}/candidates/candidates.pqt')
    print('candidate的长度为', len(candidates))

    print("开始处理candidates数据！！！")
    # 标记类型
    print("转换！！！")
    candidates["type"] = candidates.session_type.apply(lambda x: x.split("_")[1])
    candidates["type"] = candidates["type"].map(type_transform).astype('int8')
    for t in candidate_type:
        print(f"只要{t}！！！")
        candidates = candidates[candidates['type'] == type_transform[t]]
        print("推荐长度：", len(candidates))
        # 裂开  session_type, labels, type
        print("裂开！！！")
        candidates["labels"] = candidates["labels"].apply(lambda x: x.split(" "))
        candidates = candidates.explode("labels")
        # 开始计算类型 session_type, labels, type, candidate_type
        print("candidate_type")
        candidates["candidate_type"] = candidates["labels"].apply(lambda x: x.split('#')[1]).astype('float32').astype(
            'int32')
        # 开始计算得分 session_type, labels, type, candidate_type, candidate_type_scores
        print("candidate_type_scores")
        candidates["candidate_type_scores"] = candidates["labels"].apply(lambda x: x.split('#')[2]).astype('float32')
        # 开始标签 session_type, labels, type, candidate_type, candidate_type_scores
        print("labels")
        candidates["labels"] = candidates["labels"].apply(lambda x: x.split('#')[0]).astype('int32')
        candidates["session_type"] = candidates.session_type.apply(lambda x: x.split("_")[0]).astype("int32")
        candidates.rename(columns={'session_type': 'session', 'labels': 'aid'}, inplace=True)
        print(candidates)

        # 'session', 'aid', 'type', 'candidate_type', 'candidate_type_scores'
        # history_aid, sim_aid, top_hot_aid, top_orders_aid
        candidate_type_dic = {1: 'history_aid', 2: 'sim_aid', 3: 'top_hot_aid', 4: 'top_orders_aid',
                              5: 'top_carts_aid', 6: 'top_hot_aid_last_month', 7: 'deepwalk', 8: 'word2vec'}
        candidate_type_scores_dic = {1: 'history_aid_score', 2: 'sim_aid_score', 3: 'top_hot_aid_score',
                                     4: 'top_orders_aid_score', 5: 'top_carts_aid_score',
                                     6: 'top_hot_aid_last_month_score', 7: 'deepwalk_score',
                                     8: 'word2vec_score'}
        print('开始merge！！！')
        candidates1 = candidates[candidates['candidate_type'] == 1]
        candidates1.columns = ['session', 'aid', 'type', 'history_aid', 'history_aid_score']
        candidates1 = candidates1.sort_values(['session', 'history_aid_score'], ascending=[True, False])
        candidates1['history_aid_rank'] = candidates1.groupby('session')['aid'].cumcount()
        print(candidates1)
        for i in range(7):
            temp_df = candidates[candidates['candidate_type'] == i + 2]
            temp_df['candidate_type'] = 1
            temp_df.rename(columns={'candidate_type': f'{candidate_type_dic[i + 2]}',
                                    'candidate_type_scores': f'{candidate_type_scores_dic[i + 2]}'}, inplace=True)
            temp_df = temp_df.sort_values(['session', f'{candidate_type_scores_dic[i + 2]}'],
                                          ascending=[True, False])
            temp_df[f'{candidate_type_dic[i + 2]}_rank'] = temp_df.groupby('session')['aid'].cumcount()
            print(temp_df)
            candidates1 = candidates1.merge(temp_df, on=['session', 'aid', 'type'], how='outer').fillna(value=-1)
            print(candidates1)
        candidates1.to_parquet(f'/home/niejianfei/otto/{stage}/candidates/candidates_{t}.pqt')
        print('保存完毕')


if __name__ == '__main__':
    IS_TRAIN = True
    candidate_type = ['clicks', 'carts', 'orders']
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'

    recall_features(stage, candidate_type)
