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

import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(
            input_size, hidden_channels)
        self.conv2 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv3 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv4 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv5 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv6 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv7 = GCNConv(
            hidden_channels, hidden_channels)
        self.conv8 = GCNConv(
            hidden_channels, 2)
    
    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv3(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = F.dropout(x, p= 0.2, training=self.training)
        x = self.conv4(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv5(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv6(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = F.dropout(x, p= 0.2, training=self.training)
        x = x.relu()
        x = self.conv7(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv8(x, data.edge_index, data.edge_attr)

        return x        
    
def train(data_loader, model, optimizer, criterion, useScaler, criterion_loc): 
    running_loss = 0
    model.train()
    if useScaler:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data)
        reference_indices = (data.mask == 1).nonzero(as_tuple=True)[0]
        reference_out = torch.index_select(out, 0, reference_indices)
        reference_y = torch.index_select(data.y, 0, reference_indices)

        y_index = (reference_y == 1).nonzero(as_tuple=True)[0].item()
        pred_max_index = torch.argmax(reference_out, dim=0)[1]
        correct_prediction = 0
        if y_index == pred_max_index:
            correct_prediction = 1

        loss_norm = criterion(reference_out, reference_y)
        loss_loc = criterion_loc(torch.tensor([correct_prediction], dtype=torch.float), torch.tensor([1], dtype=torch.float))/100
        #print(f'loss norm: {loss_norm}, loss loc: {loss_loc}, pred ind: {pred_max_index}, y_index: {y_index}, correct_prediction: {correct_prediction}') 
        loss = loss_norm + loss_loc

        if useScaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss/len(data_loader)
        
def test(data_loader, model, k):
    model.eval()
    correct = 0
    correct_topk = 0
    for data in data_loader:
        out = model(data)  

        reference_indices = (data.mask == 1).nonzero(as_tuple=True)[0]
        reference_y = torch.index_select(data.y, 0, reference_indices)
        y_index = (reference_y == 1).nonzero(as_tuple=True)[0].item()
        reference_out = torch.index_select(out, 0, reference_indices)

        pred_max_index = torch.argmax(reference_out, dim=0)[1]
        if y_index == pred_max_index:
            correct += 1

        pred_max_indices = torch.topk(reference_out, k, dim=0)[1]
        pred_max_indices = torch.transpose(pred_max_indices, 0, 1)[1]
        # print(f'reference_out: {reference_out}, pred_max_indices: {pred_max_indices}, pred_max_indices[1]: {pred_max_indices[1]}, y_index: {y_index}, y_index in pred_max_indices: {y_index in pred_max_indices[1]}') 
        if y_index in pred_max_indices:
            correct_topk += 1
        
    return (correct / len(data_loader.dataset), correct_topk / len(data_loader.dataset))
    
if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])
    
    data_list, weights = prepareData(dataset_dir, True)
    random.shuffle(data_list)
    data_list_train = data_list[:int(0.7*len(data_list))]
    data_list_test = data_list[int(0.7*len(data_list)):]
    print(f'Total Number of Graphs: {len(data_list)}, Number of Graphs for Training: {len(data_list_train)}, Number of Graphs for Testing: {len(data_list_test)}') 

    train_epoch = 25
    batch_size = 1
    k = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(1, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion_loc = torch.nn.BCELoss()
    train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=True)

    losses, train_accs, test_accs, train_accs_topk, test_accs_topk  = [], [], [], [], []

    for epoch in range (1, train_epoch+1):
        loss = train(train_loader, model, optimizer, criterion, False, criterion_loc)
        losses.append(loss)
        train_acc = test(train_loader, model, k)
        train_accs.append(train_acc[0])
        train_accs_topk.append(train_acc[1])
        test_acc = test(test_loader, model, k)
        test_accs.append(test_acc[0])
        test_accs_topk.append(test_acc[1])
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc[0]:.4f}, Test Acc: {test_acc[0]:.4f}, Train Acc Tok k: {train_acc[1]:.4f}, Test Acc: {test_acc[1]:.4f}, Loss: {loss:.4f}')

    plt.figure(0)
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim((min(losses)-0.1, losses[5]+0.1))
    plt.savefig('losses_loc_detail.png')

    plt.figure(1)
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('losses_loc_total.png')
    
    plt.figure(2)
    plt.plot(train_accs, label='Training')
    plt.plot(test_accs, label='Testing')
    plt.plot(train_accs_topk, label='Training Top k')
    plt.plot(test_accs_topk, label='Testing Top k')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim((0, max(max(train_accs_topk), max(test_accs_topk))+0.05))
    plt.savefig('accuracies_loc.png')
    plt.close()