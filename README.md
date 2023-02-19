# Kaggle_OTTO_Multi-Objective_Recommender_System
kaggle比赛—otto多目标推荐系统源代码，单模型分数0.594，LB排名30左右

## 召回阶段
1.基于历史序列召回

2.基于协同过滤co-visitation召回(I2I)

3.基于规则召回

  点击最多/加购最多/购买最多/热门商品，综合指数最高
  
4.基于embedding召回

  deepwalk last week(I2I)
  
  deepwalk last month(I2I)

## 排序阶段

构造candidates特征，使用xgboost作为排序模型，做出预测。

特征构造具体如下：

初次召回特征（多重召回策略所带的特征）

item特征

user特征

user和item的交互特征

similarity特征（包括deepwalk,ProNE等相似度特征）

co-visitation特征

## 模型提升历程

1.利用手工规则recall@20后LB分数为0.577

2.采用rank模型，增加召回数量（平均每个user召回170个item），candidates加入相似度特征（mean和max）后，LB提升到0.585

3.尝试向量召回，继续增加召回数量（平均每个user召回220个item），并加入co-visitation权重特征（mean和max），LB提升到0.590

4.继续尝试增加相似度特征（candidate与user序列最后三个aid分别的相似度特征）和co-visitation权重特征（candidate与user序列最后三个aid分别的权重特征），LB提升到0.594

## 尝试但不work的方法

1.ProNE基于图的user-item相似度特征

2.BPRMF，ALSMF，LMF基于矩阵分解的user-item相似度特征

3.许多item特征，例如时间趋势类特征、物品点击购买率类特征等

4.许多user特征，例如点击购买时间间隔类特征、用户点击购买率类特征等

5.使用网格搜索对xgboost简单进行调参，模型几乎没有提升

6.使用简单的特征交叉，模型没有提升
