import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import community_detection
# 创建一个示例图

G = nx.read_edgelist('data/facebook_combined.txt',nodetype=int)
# 初始化 Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, workers=4)

# 训练嵌入
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# 获取节点的嵌入向量
node_embeddings = model.wv

# 查看某个节点的嵌入向量
print("Embedding of node 0:", node_embeddings['0'])

# 保存和加载模型
model.save("node2vec.model")
# model = Node2Vec.load("node2vec.model")


import numpy as np
import pandas as pd

# 从模型中提取节点嵌入向量
node_embeddings = model.wv

# 获取节点列表并按节点 ID 排序
nodes = sorted(G.nodes())

# 生成特征矩阵
feature_matrix = np.array([node_embeddings[str(node)] for node in nodes])

# 将特征矩阵转化为 Pandas DataFrame（可视化友好）
feature_df = pd.DataFrame(feature_matrix, index=nodes)
feature_df.index.name = "Node"

# 打印特征矩阵信息
print("Feature matrix shape:", feature_df.shape)
print(feature_df.head())


feature_df_with_labels = feature_df.copy()
feature_df_with_labels["Label"] = feature_df.index.map(community_detection.numeric_labels)


# 保存到文件（可选）
feature_df_with_labels.to_csv("feature_matrix_with_labels.csv")
