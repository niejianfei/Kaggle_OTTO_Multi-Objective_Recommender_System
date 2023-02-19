import gc
import glob
import cudf
import numpy as np
import pandas as pd

print('We will use RAPIDS version', cudf.__version__)
VER = 6
type_weight = {0: 1, 1: 5, 2: 4}
IS_TRAIN = True
use_all_data = True


# CACHE FUNCTIONS
# 读取文件路径，将cpu RAM上的对应df读取到GPU上
def read_file(f):
    return cudf.DataFrame(data_cache[f])


def read_file_to_cache(f):
    df = pd.read_parquet(f)
    df.ts = (df.ts / 1000).astype('int32')
    if not use_all_data:
        # 除去第一周的数据
        df = df[df['ts'] >= 1659909599]
    df['type'] = df['type'].map(type_labels).astype('int8')
    return df


# CACHE THE DATA ON CPU BEFORE PROCESSING ON GPU
# 存储在cpu上的字典
data_cache = {}
type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}
if IS_TRAIN:
    # glob模块用来查找文件目录和文件，并将搜索的到的结果返回到一个列表中
    files = glob.glob('/home/niejianfei/otto/CV/data/*_parquet/*')
    # 存到字典里面存到cpu的RAM里面，字典的键是文件路径，值是对应路径文件生成的dataframe
    for f in files: data_cache[f] = read_file_to_cache(f)
else:
    # glob模块用来查找文件目录和文件，并将搜索的到的结果返回到一个列表中
    files = glob.glob('/home/niejianfei/otto/LB/data/*_parquet/*')
    # 存到字典里面存到cpu的RAM里面，字典的键是文件路径，值是对应路径文件生成的dataframe
    for f in files: data_cache[f] = read_file_to_cache(f)

# CHUNK PARAMETERS
# 分成5组
READ_CT = 5
# ceil向上取整将文件分成chunk=len/6块，将文件分成6块
CHUNK = int(np.ceil(len(files) / 6))
print(f'We will process {len(files)} files, in groups of {READ_CT} and chunks of {CHUNK}.')

# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 4
# item的数量/分区数量
SIZE = 1.86e6 / DISK_PIECES

# "Carts Orders" Co-visitation Matrix - Type Weighted
# COMPUTE IN PARTS FOR MEMORY MANGEMENT
# for循环分块计算
for PART in range(DISK_PIECES):  # 一次循环计算150个文件中的1/4 个items（item[0，180w/4]），
    print()
    print('### DISK PART', PART + 1)

    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    # 150个文件分成6大块，每一块25个小文件
    for j in range(6):  # 6 * 5 *5 = 30 * 5 = 150
        a = j * CHUNK
        b = min((j + 1) * CHUNK, len(files))
        print(f'Processing files {a} thru {b - 1} in groups of {READ_CT}...')

        # => INNER CHUNKS
        # 25个小文件分成5份，每份5个文件，读取最开始的那1份文件
        for k in range(a, b, READ_CT):
            # READ FILE
            # 读到GPU里面去,df为list
            df = [read_file(files[k])]
            for i in range(1, READ_CT):  # 在上述一份文件的基础上，在添加4份文件到GPU
                if k + i < b: df.append(read_file(files[k + i]))
            # 融合5个dataframe信息
            df = cudf.concat(df, ignore_index=True, axis=0)
            # 升序排列session，降序排列ts
            df = df.sort_values(['session', 'ts'], ascending=[True, False])

            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            # session分组排序标序号，[0-count-1]，df顺序不变，0-n-1的顺序是降序排列，留下session最近的30个item
            df['n'] = df.groupby('session').cumcount()
            # 过滤数据，筛选出n小于30的session，类似于baseline中的ranking和session_day count
            df = df.loc[df.n < 30].drop('n', axis=1)

            # CREATE PAIRS
            df = df.merge(df, on='session')
            # 构造item对，这两个item被user查看的时间相差不到一天
            df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]

            # MEMORY MANAGEMENT COMPUTE IN PARTS
            # 内存管理，这里对df的计算继续分区（采用过滤的方式），分part计算，一共有1800000个item，size=sum（items）/ DISK_PIECES
            df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]

            # ASSIGN WEIGHTS
            # 只留下 session ，item pair，type信息并去重
            df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y', 'type_y'])
            # 根据merge的aid_y的类型赋予权重，，x的类型不知道？？
            df['wgt'] = df.type_y.map(type_weight)
            # 去掉session和type信息
            df = df[['aid_x', 'aid_y', 'wgt']]
            df.wgt = df.wgt.astype('float32')
            # items pair groupby分组计算权重 click/carts/orders 1/5/4
            df = df.groupby(['aid_x', 'aid_y']).wgt.sum()
            # print(df)
            # COMBINE INNER CHUNKS
            if k == a:
                tmp2 = df
            else:
                tmp2 = tmp2.add(df, fill_value=0)
            print(k, ', ', end='')

        print()

        # COMBINE OUTER CHUNKS
        if a == 0:
            tmp = tmp2
        else:
            tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()

    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])
    print(tmp)
    # SAVE TOP 40  15
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    print(tmp)
    tmp = tmp.loc[tmp.n < 50]
    print(tmp)
    # SAVE PART TO DISK (convert to pandas first uses less memory)
    if IS_TRAIN:
        tmp.to_pandas().to_parquet(f'/home/niejianfei/otto/CV/preprocess/top_15_carts_orders_v{VER}_{PART}.pqt')
    else:
        if use_all_data:
            tmp.to_pandas().to_parquet(
                f'/home/niejianfei/otto/LB/preprocess/all_data_top_15_carts_orders_v{VER}_{PART}.pqt')
        else:
            tmp.to_pandas().to_parquet(
                f'/home/niejianfei/otto/LB/preprocess/top_15_carts_orders_v{VER}_{PART}.pqt')

# 2."Buy2Buy" Co-visitation Matrix
# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 1
SIZE = 1.86e6 / DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(DISK_PIECES):
    print()
    print('### DISK PART', PART + 1)

    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    for j in range(6):
        a = j * CHUNK
        b = min((j + 1) * CHUNK, len(files))
        print(f'Processing files {a} thru {b - 1} in groups of {READ_CT}...')

        # => INNER CHUNKS
        for k in range(a, b, READ_CT):

            # READ FILE
            df = [read_file(files[k])]
            for i in range(1, READ_CT):
                if k + i < b: df.append(read_file(files[k + i]))
            df = cudf.concat(df, ignore_index=True, axis=0)
            df = df.loc[df['type'].isin([1, 2])]  # ONLY WANT CARTS AND ORDERS
            df = df.sort_values(['session', 'ts'], ascending=[True, False])

            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            df = df.loc[df.n < 30].drop('n', axis=1)

            # CREATE PAIRS
            df = df.merge(df, on='session')
            df = df.loc[((df.ts_x - df.ts_y).abs() < 14 * 24 * 60 * 60) & (df.aid_x != df.aid_y)]  # 14 DAYS

            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]

            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y', 'type_y'])
            df['wgt'] = 1
            df = df[['aid_x', 'aid_y', 'wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x', 'aid_y']).wgt.sum()

            # COMBINE INNER CHUNKS
            if k == a:
                tmp2 = df
            else:
                tmp2 = tmp2.add(df, fill_value=0)
            print(k, ', ', end='')

        print()

        # COMBINE OUTER CHUNKS
        if a == 0:
            tmp = tmp2
        else:
            tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()

    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])

    # SAVE TOP 15
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n < 50]
    # SAVE PART TO DISK (convert to pandas first uses less memory)
    if IS_TRAIN:
        tmp.to_pandas().to_parquet(f'/home/niejianfei/otto/CV/preprocess/top_15_buy2buy_v{VER}_{PART}.pqt')
    else:
        if use_all_data:
            tmp.to_pandas().to_parquet(
                f'/home/niejianfei/otto/LB/preprocess/all_data_top_15_buy2buy_v{VER}_{PART}.pqt')
        else:
            tmp.to_pandas().to_parquet(f'/home/niejianfei/otto/LB/preprocess/top_15_buy2buy_v{VER}_{PART}.pqt')

# 3."Clicks" Co-visitation Matrix - Time Weighted
# USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
DISK_PIECES = 4
SIZE = 1.86e6 / DISK_PIECES

# COMPUTE IN PARTS FOR MEMORY MANGEMENT
for PART in range(DISK_PIECES):
    print()
    print('### DISK PART', PART + 1)

    # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
    # => OUTER CHUNKS
    for j in range(6):
        a = j * CHUNK
        b = min((j + 1) * CHUNK, len(files))
        print(f'Processing files {a} thru {b - 1} in groups of {READ_CT}...')

        # => INNER CHUNKS
        for k in range(a, b, READ_CT):
            # READ FILE
            df = [read_file(files[k])]
            for i in range(1, READ_CT):
                if k + i < b: df.append(read_file(files[k + i]))
            df = cudf.concat(df, ignore_index=True, axis=0)
            df = df.sort_values(['session', 'ts'], ascending=[True, False])

            # USE TAIL OF SESSION
            df = df.reset_index(drop=True)
            df['n'] = df.groupby('session').cumcount()
            df = df.loc[df.n < 30].drop('n', axis=1)

            # CREATE PAIRS
            df = df.merge(df, on='session')
            df = df.loc[((df.ts_x - df.ts_y).abs() < 24 * 60 * 60) & (df.aid_x != df.aid_y)]

            # MEMORY MANAGEMENT COMPUTE IN PARTS
            df = df.loc[(df.aid_x >= PART * SIZE) & (df.aid_x < (PART + 1) * SIZE)]

            # ASSIGN WEIGHTS
            df = df[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1 + 3 * (df.ts_x - 1659304800) / (1662328791 - 1659304800)  # 归一化数据，离得时间越近，权重越大
            # 1659304800 : minimum timestamp
            # 1662328791 : maximum timestamp
            df = df[['aid_x', 'aid_y', 'wgt']]
            df.wgt = df.wgt.astype('float32')
            df = df.groupby(['aid_x', 'aid_y']).wgt.sum()

            # COMBINE INNER CHUNKS
            if k == a:
                tmp2 = df
            else:
                tmp2 = tmp2.add(df, fill_value=0)
            print(k, ', ', end='')
        print()

        # COMBINE OUTER CHUNKS
        if a == 0:
            tmp = tmp2
        else:
            tmp = tmp.add(tmp2, fill_value=0)
        del tmp2, df
        gc.collect()

    # CONVERT MATRIX TO DICTIONARY
    tmp = tmp.reset_index()
    tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])

    # SAVE TOP 20
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n < 50]

    # SAVE PART TO DISK (convert to pandas first uses less memory)
    if IS_TRAIN:
        tmp.to_pandas().to_parquet(f'/home/niejianfei/otto/CV/preprocess/top_20_clicks_v{VER}_{PART}.pqt')
    else:
        if use_all_data:
            tmp.to_pandas().to_parquet(
                f'/home/niejianfei/otto/LB/preprocess/all_data_top_20_clicks_v{VER}_{PART}.pqt')
        else:
            tmp.to_pandas().to_parquet(f'/home/niejianfei/otto/LB/preprocess/top_20_clicks_v{VER}_{PART}.pqt')
