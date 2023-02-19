import glob
import math
import pandas as pd
import numpy as np


def load_data(path):
    type_transform = {"clicks": 0, "carts": 1, "orders": 2}
    dfs = []
    # 只导入训练数据
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts / 1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_transform).astype('int8')
        if not IS_TRAIN:
            # 除去第一周的数据
            chunk = chunk[chunk['ts'] >= 1659909599]
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True)


def item_features(input_path, output_path):
    print("开始导入数据！！！")
    train = load_data(input_path)

    print("开始构造item_feature!!!")
    # Step 2:构造item_features
    # item_features，使用train data 和valid data
    print("开始聚合aid：agg中！！！")
    item_features = train.groupby('aid').agg({'aid': 'count', 'session': 'nunique', 'type': ['mean', 'skew'],
                                              'ts': ['min', 'max', 'skew']})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    item_features.columns = ['item_item_count', 'item_user_count', 'item_buy_ratio', 'item_buy_skew', 'item_min_ts',
                             'item_max_ts', 'item_skew_ts']
    print("开始构造ts偏态峰态中！！！")
    # 计算时间偏态系数，计算时间峰态系数,Pandas Series.kurt()函数使用Fisher对峰度的定义（正常的峰度==0.0）
    item_features['item_skew_ts'] = item_features['item_skew_ts'].fillna(value=0)
    item_features['item_kurt_ts'] = train.groupby('aid')['ts'].apply(lambda x: pd.DataFrame.kurt(x)).fillna(value=0)

    print("开始构造type偏态峰态中！！！")
    # 计算类型偏态系数，计算类型峰态系数,Pandas Series.kurt()函数使用Fisher对峰度的定义（正常的峰度==0.0）
    item_features['item_buy_skew'] = item_features['item_buy_skew'].fillna(value=0)
    item_features['item_buy_kurt'] = train.groupby('aid')['type'].apply(lambda x: pd.DataFrame.kurt(x)).fillna(value=0)
    # aids序列持续的时间(天)
    print("开始计算ts时间s！！！")
    item_features['item_long_ts'] = item_features['item_max_ts'] - item_features['item_min_ts']
    print(item_features)
    item_features = item_features.drop(columns=['item_min_ts', 'item_max_ts'])

    print("开始计算aid三个比例特征！！！")
    # aid平均每天被观看几次
    item_features["item_avg_visit_per_day"] = item_features['item_item_count'] / (item_features['item_long_ts'] / (60 *
                                                                                                                   60 * 24)).clip(
        1, 60).apply(lambda x: math.ceil(x))
    item_features["item_repeat_visit_num"] = item_features['item_item_count'] - item_features['item_user_count']
    # 平均每个商品被每个用户观看的次数
    item_features["item_ave_visit_num"] = item_features['item_item_count'] / item_features['item_user_count']
    # aids的re_watch比例
    item_features["item_re_visit_rate"] = item_features['item_repeat_visit_num'] / item_features['item_item_count']

    # train 的ts是毫秒，没有除以1000
    print("开始导入数据！！！")
    # 前三周的训练数据

    time = (train['ts'].max() - train['ts'].min()) / (60 * 60 * 24)
    print('天', time)
    # 只要后几周的数据
    train['ts_minus'] = (train['ts'] - train['ts'].min()) / (60 * 60 * 24)
    # 最后一周
    print('最后一周')
    train1 = train[train['ts_minus'] >= 21].drop(columns='ts_minus')
    print(train1)
    item_item_count_last_week = train1.groupby('aid').agg({'aid': 'count', 'type': 'mean'})
    item_item_count_last_week.columns = ['item_item_count_last_week', 'item_buy_ratio_last_week']
    print(item_item_count_last_week)
    # 最后两周
    print('最后两周')
    train2 = train[train['ts_minus'] >= 14].drop(columns='ts_minus')
    print(train2)
    item_item_count_last_two_week = train2.groupby('aid').agg({'aid': 'count', 'type': 'mean'})
    item_item_count_last_two_week.columns = ['item_item_count_last_two_week', 'item_buy_ratio_last_two_week']
    print(item_item_count_last_two_week)
    # 最后三周
    print('最后三周')
    train3 = train[train['ts_minus'] >= 7].drop(columns='ts_minus')
    print(train3)
    item_item_count_last_three_week = train3.groupby('aid').agg({'aid': 'count', 'type': 'mean'})
    item_item_count_last_three_week.columns = ['item_item_count_last_three_week', 'item_buy_ratio_last_three_week']
    print(item_item_count_last_three_week)

    item_features = item_features.merge(item_item_count_last_week, left_index=True, right_index=True,
                                        how='left').fillna(value=-1000)
    item_features = item_features.merge(item_item_count_last_two_week, left_index=True, right_index=True,
                                        how='left').fillna(value=-1000)
    item_features = item_features.merge(item_item_count_last_three_week, left_index=True, right_index=True,
                                        how='left').fillna(value=-1000)

    print(item_features)
    print(item_features.columns)

    # 规定保存格式
    item_features = item_features.astype('float32')
    print("开始保存特征到文件！！！")
    item_features.to_parquet(output_path)


def add_item_features(input_path1, input_path2, output_path):
    # item feature
    # item_feature:点击购买率 item_item 总count / cart/order count
    # 点击加购率
    # 加购购买率
    # 点击占比（点击占全部点击之比）
    # 加购占比
    # 购买占比
    # last_week和last_month      趋势 斜率变化
    # 复购率   集中度
    # 复加购率
    # 复点击率 item_item - item_user
    print("开始导入数据！！！")
    train = load_data(input_path1)

    train_click = train[train['type'] == 0]
    train_cart = train[train['type'] == 1]
    train_order = train[train['type'] == 2]

    print("开始构造item_feature!!!")
    # 最后一个月
    print("开始聚合aid：agg中！！！")
    click_item_features = train_click.groupby('aid').agg({'aid': 'count', 'session': 'nunique'})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    click_item_features.columns = ['click_item_item_count', 'click_item_user_count']

    cart_item_features = train_cart.groupby('aid').agg({'aid': 'count', 'session': 'nunique'})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    cart_item_features.columns = ['cart_item_item_count', 'cart_item_user_count']

    order_item_features = train_order.groupby('aid').agg({'aid': 'count', 'session': 'nunique'})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    order_item_features.columns = ['order_item_item_count', 'order_item_user_count']

    click_item_features = click_item_features.merge(cart_item_features, left_index=True, right_index=True,
                                                    how='left').fillna(value=0)
    click_item_features = click_item_features.merge(order_item_features, left_index=True, right_index=True,
                                                    how='left').fillna(value=0)

    # click_item_item_count, click_item_user_count
    # 点击购买率 * 3
    click_item_features['click_cart_rate'] = click_item_features['cart_item_item_count'] / click_item_features[
        'click_item_item_count']
    click_item_features['click_order_rate'] = click_item_features['order_item_item_count'] / click_item_features[
        'click_item_item_count']
    click_item_features['cart_order_rate'] = (
            click_item_features['order_item_item_count'] / click_item_features['cart_item_item_count'])
    print(click_item_features['cart_order_rate'].max())
    print(click_item_features['cart_order_rate'].min())
    features = click_item_features[
        (click_item_features['order_item_item_count'] == 0) & (click_item_features['cart_item_item_count'] == 0)]
    print(features[['cart_item_item_count', 'order_item_item_count', 'cart_order_rate']])
    # 点击占比
    click_item_features['click_percentage'] = click_item_features['click_item_item_count'] / click_item_features[
        'click_item_item_count'].sum()
    click_item_features['cart_percentage'] = click_item_features['cart_item_item_count'] / click_item_features[
        'cart_item_item_count'].sum()
    click_item_features['order_percentage'] = click_item_features['order_item_item_count'] / click_item_features[
        'order_item_item_count'].sum()
    # 复购率
    click_item_features['re_click_rate'] = (click_item_features['click_item_item_count'] - click_item_features[
        'click_item_user_count']) / click_item_features['click_item_item_count']
    click_item_features['re_cart_rate'] = (click_item_features['cart_item_item_count'] - click_item_features[
        'cart_item_user_count']) / click_item_features['cart_item_item_count']
    click_item_features['re_order_rate'] = (click_item_features['order_item_item_count'] - click_item_features[
        'order_item_user_count']) / click_item_features['order_item_item_count']

    click_item_features = click_item_features.replace(np.inf, 100)

    print("开始导入valid数据！！！")
    valid = load_data(input_path2)

    valid_click = valid[valid['type'] == 0]
    valid_cart = valid[valid['type'] == 1]
    valid_order = valid[valid['type'] == 2]

    print("开始构造item_feature!!!")
    # 最后一个月
    print("开始聚合aid：agg中！！！")
    valid_click_item_features = valid_click.groupby('aid').agg({'aid': 'count', 'session': 'nunique'})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    valid_click_item_features.columns = ['click_item_item_count1', 'click_item_user_count1']

    valid_cart_item_features = valid_cart.groupby('aid').agg({'aid': 'count', 'session': 'nunique'})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    valid_cart_item_features.columns = ['cart_item_item_count1', 'cart_item_user_count1']

    valid_order_item_features = valid_order.groupby('aid').agg({'aid': 'count', 'session': 'nunique'})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    valid_order_item_features.columns = ['order_item_item_count1', 'order_item_user_count1']

    valid_click_item_features = valid_click_item_features.merge(valid_cart_item_features, left_index=True,
                                                                right_index=True,
                                                                how='left').fillna(value=0)
    valid_click_item_features = valid_click_item_features.merge(valid_order_item_features, left_index=True,
                                                                right_index=True,
                                                                how='left').fillna(value=0)
    # click_item_item_count, click_item_user_count
    # 点击购买率 * 3
    valid_click_item_features['click_cart_rate1'] = valid_click_item_features['cart_item_item_count1'] / \
                                                    valid_click_item_features[
                                                        'click_item_item_count1']
    valid_click_item_features['click_order_rate1'] = valid_click_item_features['order_item_item_count1'] / \
                                                     valid_click_item_features[
                                                         'click_item_item_count1']
    valid_click_item_features['cart_order_rate1'] = valid_click_item_features['order_item_item_count1'] / \
                                                    valid_click_item_features[
                                                        'cart_item_item_count1']
    # 点击占比
    valid_click_item_features['click_percentage1'] = valid_click_item_features['click_item_item_count1'] / \
                                                     valid_click_item_features[
                                                         'click_item_item_count1'].sum()
    valid_click_item_features['cart_percentage1'] = valid_click_item_features['cart_item_item_count1'] / \
                                                    valid_click_item_features[
                                                        'cart_item_item_count1'].sum()
    valid_click_item_features['order_percentage1'] = valid_click_item_features['order_item_item_count1'] / \
                                                     valid_click_item_features[
                                                         'order_item_item_count1'].sum()
    # 复购率
    valid_click_item_features['re_click_rate1'] = (valid_click_item_features['click_item_item_count1'] -
                                                   valid_click_item_features[
                                                       'click_item_user_count1']) / valid_click_item_features[
                                                      'click_item_item_count1']
    valid_click_item_features['re_cart_rate1'] = (valid_click_item_features['cart_item_item_count1'] -
                                                  valid_click_item_features[
                                                      'cart_item_user_count1']) / valid_click_item_features[
                                                     'cart_item_item_count1']
    valid_click_item_features['re_order_rate1'] = (valid_click_item_features['order_item_item_count1'] -
                                                   valid_click_item_features[
                                                       'order_item_user_count1']) / valid_click_item_features[
                                                      'order_item_item_count1']
    valid_click_item_features = valid_click_item_features.replace(np.inf, 100)

    # 缺失值用-1填补，相减后也是负数，小于等于-1
    click_item_features = click_item_features.merge(valid_click_item_features, left_index=True, right_index=True,
                                                    how='left').fillna(value=-10)
    # 点击加购率
    click_item_features['click_cart_rate_trend'] = (
            click_item_features['click_cart_rate1'] - click_item_features['click_cart_rate']).clip(-10)
    click_item_features['click_order_rate_trend'] = (
            click_item_features['click_order_rate1'] - click_item_features['click_order_rate']).clip(-10)
    click_item_features['cart_order_rate_trend'] = (
            click_item_features['cart_order_rate1'] - click_item_features['cart_order_rate']).clip(-10)
    # 点击占比
    click_item_features['click_percentage_trend'] = (
            click_item_features['click_percentage1'] - click_item_features['click_percentage']).clip(-10)
    click_item_features['cart_percentage_trend'] = (
            click_item_features['cart_percentage1'] - click_item_features['cart_percentage']).clip(-10)
    click_item_features['order_percentage_trend'] = (
            click_item_features['order_percentage1'] - click_item_features['order_percentage']).clip(-10)
    # 复购率
    click_item_features['re_click_rate_trend'] = (
            click_item_features['re_click_rate1'] - click_item_features['re_click_rate']).clip(-10)
    click_item_features['re_cart_rate_trend'] = (
            click_item_features['re_cart_rate1'] - click_item_features['re_cart_rate']).clip(-10)
    click_item_features['re_order_rate_trend'] = (
            click_item_features['re_order_rate1'] - click_item_features['re_order_rate']).clip(-10)

    print(click_item_features)
    print(click_item_features.describe())

    print("开始保存特征到文件！！！")
    click_item_features.to_parquet(output_path)


def trans_time_span_item_features(input_path, output_path1, output_path2, output_path3):
    train = load_data(input_path)

    train_clicks = train[train['type'] == 0].drop(columns='type')
    train_clicks = train_clicks.rename(columns={'ts': 'ts_click'})
    train_carts = train[train['type'] == 1].drop(columns='type')
    train_carts = train_carts.rename(columns={'ts': 'ts_cart'})
    train_orders = train[train['type'] == 2].drop(columns='type')
    train_orders = train_orders.rename(columns={'ts': 'ts_order'})

    print('click_cart_span')
    click_cart_span = train_clicks.merge(train_carts, on=['session', 'aid'], how='inner')
    print(click_cart_span)
    click_cart_span['min'] = click_cart_span['ts_click'] - click_cart_span['ts_cart']
    click_cart_span = click_cart_span[click_cart_span['min'] <= 0].drop(columns='min')
    print(click_cart_span)
    click_cart_span_feature = click_cart_span.groupby(['session', 'aid']).agg({'ts_click': 'min', 'ts_cart': 'min'})
    click_cart_span_feature.columns = ['ts_click_min', 'ts_cart_min']
    print(click_cart_span_feature)
    click_cart_span_feature['click_cart_span'] = click_cart_span_feature['ts_cart_min'] - click_cart_span_feature[
        'ts_click_min']
    print(click_cart_span_feature)
    click_cart_span_feature['aids'] = click_cart_span_feature.index.get_level_values('aid')
    print(click_cart_span_feature)
    print(click_cart_span_feature.index.get_level_values('aid')[:10])
    click_cart_span_feature = click_cart_span_feature.groupby('aids').agg({'aids': 'count', 'click_cart_span': 'mean'})
    click_cart_span_feature.columns = ['trans_click_cart_count', 'trans_click_cart_span_avg']
    print(click_cart_span_feature.describe())
    print(click_cart_span_feature)
    click_cart_span_feature.to_parquet(output_path1)

    print('click_order_span')
    click_order_span = train_clicks.merge(train_orders, on=['session', 'aid'], how='inner')
    print(click_order_span)
    click_order_span['min'] = click_order_span['ts_click'] - click_order_span['ts_order']
    click_order_span = click_order_span[click_order_span['min'] <= 0].drop(columns='min')
    print(click_order_span)
    click_order_span_feature = click_order_span.groupby(['session', 'aid']).agg({'ts_click': 'min', 'ts_order': 'min'})
    click_order_span_feature.columns = ['ts_click_min', 'ts_order_min']
    print(click_order_span_feature)
    click_order_span_feature['click_order_span'] = click_order_span_feature['ts_order_min'] - click_order_span_feature[
        'ts_click_min']
    print(click_order_span_feature)
    click_order_span_feature['aids'] = click_order_span_feature.index.get_level_values('aid')
    print(click_order_span_feature)
    print(click_order_span_feature.index.get_level_values('aid')[:10])
    click_order_span_feature = click_order_span_feature.groupby('aids').agg(
        {'aids': 'count', 'click_order_span': 'mean'})
    click_order_span_feature.columns = ['trans_click_order_count', 'trans_click_order_span_avg']
    print(click_order_span_feature.describe())
    print(click_order_span_feature)
    click_order_span_feature.to_parquet(output_path2)

    print('cart_order_span')
    carts_order_span = train_carts.merge(train_orders, on=['session', 'aid'], how='inner')
    print(carts_order_span)
    carts_order_span['min'] = carts_order_span['ts_cart'] - carts_order_span['ts_order']
    carts_order_span = carts_order_span[carts_order_span['min'] <= 0].drop(columns='min')
    print(carts_order_span)
    cart_order_span_feature = carts_order_span.groupby(['session', 'aid']).agg({'ts_cart': 'min', 'ts_order': 'min'})
    cart_order_span_feature.columns = ['ts_cart_min', 'ts_order_min']
    print(cart_order_span_feature)
    cart_order_span_feature['cart_order_span'] = cart_order_span_feature['ts_order_min'] - cart_order_span_feature[
        'ts_cart_min']
    print(cart_order_span_feature)
    cart_order_span_feature['aids'] = cart_order_span_feature.index.get_level_values('aid')
    print(cart_order_span_feature)
    print(cart_order_span_feature.index.get_level_values('aid')[:10])
    cart_order_span_feature = cart_order_span_feature.groupby('aids').agg({'aids': 'count', 'cart_order_span': 'mean'})
    cart_order_span_feature.columns = ['trans_cart_order_count', 'trans_cart_order_span_avg']
    print(cart_order_span_feature.describe())
    print(cart_order_span_feature)
    cart_order_span_feature.to_parquet(output_path3)


if __name__ == '__main__':
    IS_TRAIN = True
    if IS_TRAIN:
        stage = 'CV'
    else:
        stage = 'LB'
    input_path = f'/home/niejianfei/otto/{stage}/data/*_parquet/*'
    input_path2 = f'/home/niejianfei/otto/{stage}/data/test_parquet/*'
    output_path = f'/home/niejianfei/otto/{stage}/preprocess/item_features.pqt'
    output_path1 = f'/home/niejianfei/otto/{stage}/preprocess/add_item_features.pqt'
    item_features(input_path, output_path)
    add_item_features(input_path, input_path2, output_path1)

    input_path3 = f'/home/niejianfei/otto/{stage}/data/train_parquet/*'
    output_path2 = f'/home/niejianfei/otto/{stage}/preprocess/click_cart_item_features.pqt'
    output_path3 = f'/home/niejianfei/otto/{stage}/preprocess/click_order_item_features.pqt'
    output_path4 = f'/home/niejianfei/otto/{stage}/preprocess/cart_order_item_features.pqt'
    trans_time_span_item_features(input_path3, output_path2, output_path3, output_path4)
