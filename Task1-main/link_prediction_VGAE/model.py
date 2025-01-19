import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import args

class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)  # 第一层图卷积
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)  # 均值
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)  # 对数标准差

	def encode(self, X):
		hidden = self.base_gcn(X)  # mu和sigma的第一层卷积层共享
		self.mean = self.gcn_mean(hidden)  # hidden过另一层得到mu
		self.logstd = self.gcn_logstddev(hidden)  # hidden过另一层得到sigma
		gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)  # 编码得到隐变量
		A_pred = dot_product_decode(Z)  # 这里的A_pred取值范围未必是[0,1]
		return A_pred

class GraphConvSparse(nn.Module):
	"""
	将输入特征𝑋转换为新的嵌入表示𝐻
	H = \sigma(\hat{A}XW_0)
	"""
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Module):
	"""
	GAE: Graph Auto-Encoder
	包括一个编码器和一个解码器
	"""
	def __init__(self,adj):
		super(GAE,self).__init__()
		# 两层卷积层
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		# 第一层卷积是编码器
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		# 第二层卷积是解码器
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred
		

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out