import community
import matplotlib.cm as cm
import networkx as nx
import random
import numpy as np

def louvain_with_numeric_labels(G, pos):
    # 使用 Louvain 算法进行社区划分
    partition = nx.algorithms.community.louvain_communities(G, seed=2024)

    print("Partition:", partition)

    # 为每个社区分配从0到16的标签，并为每个节点标记其社区标签
    node_labels = {}
    for idx, community_nodes in enumerate(partition):
        for node in community_nodes:
            node_labels[node] = idx

    print("Node Labels:", dict(list(node_labels.items())[:20]))  # 输出前20个标签

    # 返回节点的社区标签
    return node_labels, partition

def create_datasets(G, node_labels, partition):
    training_set = []
    validation_set = []
    num_nodes = len(G.nodes)

    # 初始化全局向量
    training_vector = np.zeros(num_nodes, dtype=np.float32)
    validation_vector = np.zeros(num_nodes, dtype=np.float32)

    for community_nodes in partition:
        # 确保 community_nodes 是排序列表
        sorted_nodes = sorted(community_nodes)
        # 从每个社区中随机选择10个用于训练，20个用于验证
        sampled_nodes = random.sample(sorted_nodes, min(len(sorted_nodes), 30))
        training_nodes = sampled_nodes[:10]
        validation_nodes = sampled_nodes[10:30]

        # 更新训练集和验证集列表
        training_set.extend(training_nodes)
        validation_set.extend(validation_nodes)

        # 更新全局训练和验证向量
        for node in training_nodes:
            training_vector[node] = 1.0
        for node in validation_nodes:
            validation_vector[node] = 1.0

    return training_set, validation_set, training_vector, validation_vector



G = nx.read_edgelist('data/facebook_combined.txt',nodetype=int)
pos = nx.spring_layout(G)

print("Louvain test with numeric labels")
numeric_labels, partition = louvain_with_numeric_labels(G, pos)

training_set, validation_set, training_vector, validation_vector = create_datasets(G, numeric_labels, partition)
print("Training Set:", training_set)
print("Validation Set:", validation_set)
print("Training Vector:", training_vector)
print("Validation Vector:", validation_vector)
import pandas as pd

df = pd.DataFrame({
    'Training Vector': training_vector,
    'Validation Vector': validation_vector
})

# 保存为 CSV 文件
df.to_csv('data/vectors.csv', index=False)


