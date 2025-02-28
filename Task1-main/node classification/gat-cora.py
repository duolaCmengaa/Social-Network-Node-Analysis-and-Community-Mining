import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GATConv

device = torch.device('cpu')

class Net(torch.nn.Module):
    def __init__(self,dp=0.6,p=0.5):
        super(Net, self).__init__()
        self.conv1 = GATConv(1433, 8, heads=8, dropout=dp)
        self.conv2 = GATConv(8*8, 7, heads=1, concat=False, dropout=dp)
        self.p=p

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x,p=self.p, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x ,p=self.p, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='./', name='Cora')

GCN = Net(dp=0.7,p=0.7).to(device)
data = dataset[0].to(device)
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
for epoch in range(600):
    loss = train_one_epoch()
    acc = test_one_epoch()
    accL.append(acc)
    if (1+epoch) % 100 == 0:
        print('epoch',epoch+1,'loss',loss,'accuracy',acc)

print('best acc on test:',max(accL),',Epoch:',accL.index(max(accL)))


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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