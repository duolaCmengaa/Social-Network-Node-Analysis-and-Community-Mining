import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 1. 数据加载和预处理
# 加载你提供的 feature_matrix_with_labels 数据
file_path = 'data/feature_matrix_with_labels.csv'
df = pd.read_csv(file_path)

# 假设：第一列是节点标识符，最后一列是标签，中间是特征
features = df.iloc[:, 1:-1].values  # 获取特征矩阵
labels = df.iloc[:, -1].values      # 获取标签

# 标准化特征（有助于提高某些模型的表现）
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 假设每个节点是一个独立的节点，我们将其转换为 PyTorch Geometric 的 Data 对象
# 注意：此处没有图结构，所以边信息将是完全连接的图
# 创建完全连接的图（所有节点之间都有边）
num_nodes = len(df)

# 使用 pandas 读取边列表
edges = pd.read_csv('data/facebook_combined.txt',sep=" ", header=None, names=["node1", "node2"])

# 将节点信息转换为 PyTorch 张量
edge_index = torch.tensor(edges.values.T, dtype=torch.long)
# 转换数据为 PyTorch 张量
x = torch.tensor(features_scaled, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

# 创建 Data 对象
data = Data(x=x, edge_index=edge_index, y=y)

vector_file_path = 'data/vectors.csv'
vector_df = pd.read_csv(vector_file_path)

# 根据 vector.csv 文件设置 train_mask 和 test_mask
data.train_mask = torch.tensor(vector_df['Training Vector'].values, dtype=torch.bool)
data.test_mask = torch.tensor(vector_df['Validation Vector'].values, dtype=torch.bool)

class Net(torch.nn.Module):
    def __init__(self,dp=0.6,p=0.5):
        super(Net, self).__init__()
        self.conv1 = GATConv(64, 64, heads=8, dropout=dp)
        self.conv2 = GATConv(8*64, 17, heads=1, concat=False, dropout=dp)
        self.p=p

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=self.p, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x ,p=self.p, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cpu')
GCN = Net(dp=0.7,p=0.7).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.005, weight_decay=5e-4)

def train_one_epoch():
    GCN.train()
    optimizer.zero_grad()
    out = GCN(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_one_epoch():
    GCN.eval()
    _, pred = GCN(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    return accuracy.item()

accL=[]
GCN.train()
for epoch in range(50):
    loss = train_one_epoch()
    acc = test_one_epoch()
    accL.append(acc)
    if (1+epoch) % 10 == 0:
        print('epoch',epoch+1,'loss',loss,'accuracy',acc)

print('best acc on test:',max(accL),',Epoch:',accL.index(max(accL)))

#可视化
def visualize(out, color):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Perform TSNE for dimensionality reduction
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    
    # Create the figure
    plt.figure(figsize=(10, 10))
    
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # Remove spines (axes borders)
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
   
    # Scatter plot
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
       
    # Display the plot
    plt.show()
    
GCN.eval()
out = GCN(data)
visualize(out, color=data.y.detach().cpu().numpy())