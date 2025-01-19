'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from node2vec import Node2Vec

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    if dataset == 'facebook':
        # 加载数据
        edge_list = np.loadtxt("data/facebook_combined.txt", dtype=int)
        num_nodes = edge_list.max() + 1
        adj_matrix = sp.csr_matrix((np.ones(edge_list.shape[0]), (edge_list[:, 0], edge_list[:, 1])),
                                shape=(num_nodes, num_nodes))
        adj = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix)
        # 用Node2Vec生成节点嵌入
        # G = nx.from_scipy_sparse_array(adj)
        # node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, workers=4)
        # model = node2vec.fit(window=10, min_count=1, batch_words=4)
        # feature_mat = np.array([model.wv[str(node)] for node in range(num_nodes)])
        return adj

    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
