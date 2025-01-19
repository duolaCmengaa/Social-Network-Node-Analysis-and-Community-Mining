import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
import scipy.sparse as sp
import numpy as np
import os
import time
import json

from input_data import load_data
from preprocessing import *
import args
import model

import networkx as nx
from node2vec import Node2Vec


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

adj, features = load_data(args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
# asj_orig对角线元素为0
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()  # 去除稀疏矩阵中多余的0以节省内存

adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train  # 没有自环边的邻接矩阵

# Some preprocessing
adj_norm = preprocess_graph(adj)


num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()  # 非边数/边数，作为正样本的权重，以解决正负样本不均衡的问题
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)



adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), 
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), 
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1  # 是一个布尔向量，其中值为 True 的位置表示边存在的正样本
weight_tensor = torch.ones(weight_mask.size(0))  # 生成一个全为 1 的张量
weight_tensor[weight_mask] = pos_weight  # 将正样本的权重设置为 pos_weight

# 将掩蔽的边列表（train_edges 和 val_edges）转换为一维索引
def get_masked_edge_indices(edges, num_nodes):
    indices = []
    for edge in edges:
        i, j = edge
        index = i * num_nodes + j  # 转换为一维索引
        indices.append(index)
    return indices
masked_edge_indices = get_masked_edge_indices(np.concatenate([test_edges, val_edges, np.array(test_edges_false), np.array(val_edges_false)], axis=0), adj_label.shape[0])
weight_tensor[masked_edge_indices] = 0  # 掩蔽边的权重设置为 0



# init model and optimizer
model = getattr(model,args.model)(adj_norm)
optimizer = Adam(model.parameters(), lr=args.learning_rate)


def get_scores(edges_pos, edges_neg, adj_rec):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:

        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    preds_binary = np.array(preds_all > sigmoid(0.5))
    TP = np.sum(np.logical_and(preds_binary == 1, labels_all == 1))
    TN = np.sum(np.logical_and(preds_binary == 0, labels_all == 0))
    FP = np.sum(np.logical_and(preds_binary == 1, labels_all == 0))
    FN = np.sum(np.logical_and(preds_binary == 0, labels_all == 1))
    weight = pos_weight * len(edges_pos) / len(edges_neg)
    acc = (TP + weight * TN) / (TP + weight * TN + weight * FP + FN)
    return roc_score, ap_score, acc

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# train model
train_loss_list = []
train_acc_list = []
val_acc_list = []
val_roc_list = []
val_ap_list = []
for epoch in range(args.num_epoch):
    t = time.time()

    A_pred = model(features)
    optimizer.zero_grad()\
    # 计算loss时，要给正负样本带不同的权重，以解决正负样本不均衡的问题
    loss = log_lik = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
    if args.model == 'VGAE':
        kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        loss -= kl_divergence

    loss.backward()
    optimizer.step()

    _, _, train_acc = get_scores(train_edges, train_edges_false, A_pred)
    # train_acc = get_acc(A_pred, adj_label)

    val_roc, val_ap, val_acc = get_scores(val_edges, val_edges_false, A_pred)

    train_loss_list.append(loss.item())
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    val_roc_list.append(val_roc)
    val_ap_list.append(val_ap)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_auc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap), "val_acc=", "{:.5f}".format(val_acc),
          "time=", "{:.5f}".format(time.time() - t))
    

test_roc, test_ap, test_acc = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap), "test_acc=", "{:.5f}".format(test_acc))

filepath = "result/" + args.dataset + "_" + args.model + ".txt"
with open(filepath, 'w') as f:
    json.dump({"train_loss": train_loss_list, "train_acc": train_acc_list, "val_acc": val_acc_list,
               "val_roc": val_roc_list, "val_ap": val_ap_list, "test_ap": test_ap, "test_roc": test_roc, "test_acc": test_acc}, f)