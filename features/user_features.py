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


def user_features(input_path, output_path):
    print('开始读取数据！！！')
    valid = load_data(input_path)
    print("开始构造user_feature!!!")
    # 类别型变量分析：计数，分布
    # 连续性变量分析：最小值，最大值，离差，平均数，中位数，众数，标准差，变异系数，偏度，峰度
    print("开始聚合user：agg中！！！")
    user_features = valid.groupby('session').agg({'session': 'count', 'aid': 'nunique', 'type': ['mean', 'skew'],
                                                  'ts': ['min', 'max', 'skew']})
    user_features.columns = ['user_user_count', 'user_item_count', 'user_buy_ratio', 'user_buy_skew',
                             'user_min_ts', 'user_max_ts', 'user_skew_ts']
    print("开始计算ts偏态峰态系数!!!")
    # 计算时间偏态系数，计算时间峰态系数,Pandas Series.kurt()函数使用Fisher对峰度的定义（正常的峰度==0.0）
    user_features['user_skew_ts'] = user_features['user_skew_ts'].fillna(value=0)
    user_features['user_kurt_ts'] = valid.groupby('session')['ts'].apply(lambda x: pd.DataFrame.kurt(x)).fillna(value=0)

    print("开始计算type偏态峰态系数!!!")
    # 计算类型偏态系数，计算类型峰态系数,Pandas Series.kurt()函数使用Fisher对峰度的定义（正常的峰度==0.0）
    user_features['user_buy_skew'] = user_features['user_buy_skew'].fillna(value=0)
    user_features['user_buy_kurt'] = valid.groupby('session')['type'].apply(lambda x: pd.DataFrame.kurt(x)).fillna(
        value=0)

    print("开始计算ts天数!!!")
    # 序列持续的时间(天)
    user_features['user_long_ts'] = user_features['user_max_ts'] - user_features['user_min_ts']

    print("开始计算user三个比例特征！！！")
    # 平均每天观看几个商品
    user_features["user_avg_visit_per_day"] = user_features['user_user_count'] / (
            user_features['user_long_ts'] / (60 * 60 * 24)).clip(1, 60).apply(
        lambda x: math.ceil(x))
    # user重复观看的商品次数
    user_features["user_repeat_visit_num"] = user_features['user_user_count'] - user_features['user_item_count']
    # 平均每个商品观看的次数
    user_features["user_ave_visit_num"] = user_features['user_user_count'] / user_features['user_item_count']
    # session里面aids的re_watch比例
    user_features["user_re_visit_rate"] = user_features['user_repeat_visit_num'] / user_features['user_user_count']
    print(user_features.head())
    print(user_features.columns)
    print(user_features.shape)
    # 规定保存格式
    user_features = user_features.astype('float32')
    print("开始保存特征到文件！！！")
    user_features.to_parquet(output_path)


def add_user_features(input_path, output_path):
    # user feature  7
    # 平均购买/加购/点击间隔 max - min / num
    # 点击购买率
    # 点击加购率
    # 加购购买率
    # 点击占比
    # 加购占比
    # 购买占比  user特征比较稀疏，加上可能效果不好
    # 复购率
    # 复加购率
    # 复点击率 item_item - item_user
    print('开始读取数据！！！')
    train = load_data(input_path)
    print("开始构造user_feature!!!")
    # 类别型变量分析：计数，分布
    # 连续性变量分析：最小值，最大值，离差，平均数，中位数，众数，标准差，变异系数，偏度，峰度
    print("开始聚合user：agg中！！！")
    train_click = train[train['type'] == 0]
    train_cart = train[train['type'] == 1]
    train_order = train[train['type'] == 2]

    print("开始构造item_feature!!!")
    click_user_features = train_click.groupby('session').agg({'aid': ['count', 'nunique'], 'ts': ['min', 'max']})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    click_user_features.columns = ['click_user_user_count', 'click_user_item_count', 'ts_min', 'ts_max']
    click_user_features['click_time'] = click_user_features['ts_max'] - click_user_features['ts_min']
    click_user_features['avg_click_span'] = click_user_features['click_time'] / click_user_features['click_user_user_count']
    click_user_features = click_user_features.drop(columns=['ts_min', 'ts_max', 'click_time'])
    print(click_user_features)

    cart_user_features = train_cart.groupby('session').agg({'aid': ['count', 'nunique'], 'ts': ['min', 'max']})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    cart_user_features.columns = ['cart_user_user_count', 'cart_user_item_count', 'ts_min', 'ts_max']
    cart_user_features['cart_time'] = cart_user_features['ts_max'] - cart_user_features['ts_min']
    cart_user_features['avg_cart_span'] = cart_user_features['cart_time'] / cart_user_features['cart_user_user_count']
    cart_user_features = cart_user_features.drop(columns=['ts_min', 'ts_max', 'cart_time'])
    print(cart_user_features)

    order_user_features = train_order.groupby('session').agg({'aid': ['count', 'nunique'], 'ts': ['min', 'max']})
    # aid出现的次数,也就是aid发生的events数量：定义热门商品；操作aid的用户数量：简介定义热门商品；类型均值：这个商品易购程度
    order_user_features.columns = ['order_user_user_count', 'order_user_item_count', 'ts_min', 'ts_max']
    order_user_features['order_time'] = order_user_features['ts_max'] - order_user_features['ts_min']
    order_user_features['avg_order_span'] = order_user_features['order_time'] / order_user_features['order_user_user_count']
    order_user_features = order_user_features.drop(columns=['ts_min', 'ts_max', 'order_time'])
    print(order_user_features)

    click_user_features = click_user_features.merge(cart_user_features, left_index=True, right_index=True,
                                                    how='left').fillna(value=0)
    click_user_features = click_user_features.merge(order_user_features, left_index=True, right_index=True,
                                                    how='left').fillna(value=0)

    # click_item_item_count, click_item_user_count
    # 点击购买率 * 3
    click_user_features['user_click_cart_rate'] = click_user_features['cart_user_user_count'] / click_user_features[
        'click_user_user_count']
    click_user_features['user_click_order_rate'] = click_user_features['order_user_user_count'] / click_user_features[
        'click_user_user_count']
    click_user_features['user_cart_order_rate'] = click_user_features['order_user_user_count'] / click_user_features['cart_user_user_count']

    # 点击占比
    click_user_features['user_click_percentage'] = click_user_features['click_user_user_count'] / click_user_features[
        'click_user_user_count'].sum()
    click_user_features['user_cart_percentage'] = click_user_features['cart_user_user_count'] / click_user_features[
        'cart_user_user_count'].sum()
    click_user_features['user_order_percentage'] = click_user_features['order_user_user_count'] / click_user_features[
        'order_user_user_count'].sum()
    # 复购率
    click_user_features['user_re_click_rate'] = (click_user_features['click_user_user_count'] - click_user_features[
        'click_user_item_count']) / click_user_features['click_user_user_count']
    click_user_features['user_re_cart_rate'] = (click_user_features['cart_user_user_count'] - click_user_features[
        'cart_user_item_count']) / click_user_features['cart_user_user_count']
    click_user_features['user_re_order_rate'] = (click_user_features['order_user_user_count'] - click_user_features[
        'order_user_item_count']) / click_user_features['order_user_user_count']

    click_user_features = click_user_features.replace(np.inf, 100)
    click_user_features = click_user_features.fillna(value=-10)
    print(click_user_features)

    print("开始保存特征到文件！！！")
    click_user_features.to_parquet(output_path)


def trans_time_span_features(input_path, output_path1, output_path2, output_path3):
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
    click_cart_span_feature['click_cart_span'] = click_cart_span_feature['ts_cart_min'] - click_cart_span_feature['ts_click_min']
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
    click_order_span_feature['click_order_span'] = click_order_span_feature['ts_order_min'] - click_order_span_feature['ts_click_min']
    print(click_order_span_feature)
    click_order_span_feature['aids'] = click_order_span_feature.index.get_level_values('aid')
    print(click_order_span_feature)
    print(click_order_span_feature.index.get_level_values('aid')[:10])
    click_order_span_feature = click_order_span_feature.groupby('aids').agg({'aids': 'count', 'click_order_span': 'mean'})
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
    cart_order_span_feature['cart_order_span'] = cart_order_span_feature['ts_order_min'] - cart_order_span_feature['ts_cart_min']
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
    input_path = f'/home/niejianfei/otto/{stage}/data/test_parquet/*'
    output_path = f'/home/niejianfei/otto/{stage}/preprocess/user_features.pqt'
    output_path1 = f'/home/niejianfei/otto/{stage}/preprocess/add_user_features.pqt'
    user_features(input_path, output_path)
    add_user_features(input_path, output_path1)

    input_path1 = f'/home/niejianfei/otto/{stage}/data/train_parquet/*'
    output_path2 = f'/home/niejianfei/otto/{stage}/preprocess/click_cart_span_features.pqt'
    output_path3 = f'/home/niejianfei/otto/{stage}/preprocess/click_order_span_features.pqt'
    output_path4 = f'/home/niejianfei/otto/{stage}/preprocess/cart_order_span_features.pqt'
    trans_time_span_features(input_path1, output_path2, output_path3, output_path4)