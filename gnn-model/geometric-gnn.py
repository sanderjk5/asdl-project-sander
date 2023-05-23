import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import DataLoader

from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

from pathlib import Path
from docopt import docopt

from utils.datapreparation import prepareData

import random
import sys

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(
            input_size, hidden_channels)
        self.conv2 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv3 = GCNConv(
            hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)
    
    def forward(self, x, edge_index, batch,  edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
    
        return x
    
def train(data_loader, model, optimizer, criterion): 
    model.train()
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
def test(data_loader, model):
    model.eval()
    correct = 0
    for data in data_loader:
        out = model(data.x, data.edge_index, data.batch, data.edge_attr)  
        pred = torch.argmax(out, dim=1)
        #print(f'out: {F.softmax(out)}, pred: {pred}, y: {data.y}')
        correct += int((pred == data.y).sum())
    return correct / len(data_loader.dataset)
    
if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])
    
    data_list = prepareData(dataset_dir)
    random.shuffle(data_list)
    data_list_train = data_list[:int(0.7*len(data_list))]
    data_list_test = data_list[int(0.7*len(data_list)):]
    print(f'Total Number of Graphs: {len(data_list)}, Number of Graphs for Training: {len(data_list_train)}, Number of Graphs for Tests: {len(data_list_test)}')

    train_epoch = 200
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(1, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=True)

    for epoch in range (1, train_epoch+1):
        train(train_loader, model, optimizer, criterion)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')