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

import matplotlib.pyplot as plt

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GATConv(
            input_size, hidden_channels)
        self.conv2 = GATConv(
            hidden_channels, hidden_channels)
        self.conv3 = GATConv(
            hidden_channels, hidden_channels)
        self.conv4 = GATConv(
            hidden_channels, hidden_channels)
        self.conv5 = GATConv(
            hidden_channels, hidden_channels)
        self.conv6 = GATConv(
            hidden_channels, hidden_channels)
        self.conv7 = GATConv(
            hidden_channels, hidden_channels)
        self.conv8 = GATConv(
            hidden_channels, 2)
    
    def forward(self, x, edge_index,  edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p= 0.2, training=self.training)
        x = self.conv4(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv5(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv6(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p= 0.2, training=self.training)
        x = x.relu()
        x = self.conv7(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv8(x, edge_index, edge_attr)
    
        return x
    
def train(data_loader, model, optimizer, criterion, useScaler): 
    running_loss = 0
    model.train()
    if useScaler:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        # print(out)
        # print(data.y)
        loss = criterion(out, data.y)
        if useScaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss/len(data_loader)
        
def test(data_loader, model):
    model.eval()
    correct = 0
    for data in data_loader:
        out = model(data.x, data.edge_index, data.edge_attr)  
        
        y_index = (data.y == 1).nonzero(as_tuple=True)[0]

        pred_max_index = torch.argmax(out, dim=0)[1]
        if y_index == pred_max_index:
            correct += 1

        # pred_max_indices = (out[1] == torch.max(out, dim=0)[0][1]).nonzero(as_tuple=True)
        # if y_index in pred_max_indices:
        #     correct += 1


        # if pred_index[1] > 0 and pred_index[1] < 15:
        #     print(f'index: {pred_index}, out: {out}')
        # print(f'index: {y_index}, y: {data.y}')

        # pred = torch.argmax(out, dim=1)
        # print(f'out: {F.softmax(out)}, pred: {pred}, y: {data.y}')
        
    return correct / len(data_loader.dataset)
    
if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])
    
    data_list = prepareData(dataset_dir, True)
    random.shuffle(data_list)
    data_list_train = data_list[:int(0.7*len(data_list))]
    data_list_test = data_list[int(0.7*len(data_list)):]
    print(f'Total Number of Graphs: {len(data_list)}, Number of Graphs for Training: {len(data_list_train)}, Number of Graphs for Testing: {len(data_list_test)}')

    train_epoch = 50
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(1, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=True)

    #train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range (1, train_epoch+1):
        loss = train(train_loader, model, optimizer, criterion, False)
        losses.append(loss)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}')
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Loss: {loss:.4f}')
    
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('losses-loc.png')
    plt.show()
    

    