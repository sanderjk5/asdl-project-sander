import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from pathlib import Path
from utils.datapreparation import prepareDataWithoutVocabularies, prepareDataWithVocablularies
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

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
        x = self.conv7(x, data.edge_index, data.edge_attr)
        x = x.relu()
        x = self.conv8(x, data.edge_index, data.edge_attr)

        return x     
    
def train(data_loader, model, optimizer, criterion, useScaler): 
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
        pred_max_index = torch.argmax(F.softmax(reference_out, dim=1), dim=0)[1]
        correct_prediction = 0
        if y_index == pred_max_index:
            correct_prediction = 1

        loss_norm = criterion(reference_out, reference_y)
        loss = loss_norm

        if useScaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss/len(data_loader)
        
def test(data_loader, model, k_s):
    model.eval()
    correct_per_k = [0] * len(k_s)
    localization_per_bugtype = defaultdict(lambda: np.array([0, 0], dtype=np.int32))
    localization_per_refnodes_number = defaultdict(lambda: np.array([0, 0], dtype=np.int32))
    for data in data_loader:
        out = model(data)  

        reference_indices = (data.mask == 1).nonzero(as_tuple=True)[0]
        reference_y = torch.index_select(data.y, 0, reference_indices)
        y_index = (reference_y == 1).nonzero(as_tuple=True)[0].item()
        reference_out = F.softmax(torch.index_select(out, 0, reference_indices), dim=1)
        num_ref_nodes = reference_out.size()[0]

        for i in range(len(k_s)):   
            k = k_s[i]     
            if(num_ref_nodes >= k):
                pred_max_indices = torch.topk(reference_out, k, dim=0)[1]
            else:
                pred_max_indices = torch.topk(reference_out, num_ref_nodes, dim=0)[1]
            pred_max_indices = torch.transpose(pred_max_indices, 0, 1)[1]
            if y_index in pred_max_indices:
                correct_per_k[i] += 1
        
        pred_max_index = torch.argmax(reference_out, dim=0)[1]
        if y_index == pred_max_index:
            localization_per_bugtype[data.bugtype[0]] += [1, 1]
            localization_per_refnodes_number[num_ref_nodes] += [1, 1]
        else:
            localization_per_bugtype[data.bugtype[0]] += [0, 1]
            localization_per_refnodes_number[num_ref_nodes] += [0, 1]

    return ([correct/len(data_loader) for correct in correct_per_k], localization_per_bugtype, localization_per_refnodes_number)

def baseline(data_loader, k_s):
    correct_per_k = [0] * len(k_s)
    localization_per_bugtype = defaultdict(lambda: np.array([0, 0], dtype=np.int32))
    localization_per_refnodes_number = defaultdict(lambda: np.array([0, 0], dtype=np.int32))
    for data in data_loader:
        reference_indices = (data.mask == 1).nonzero(as_tuple=True)[0]
        num_ref_nodes = len(reference_indices)
        reference_y = torch.index_select(data.y, 0, reference_indices)
        y_index = (reference_y == 1).nonzero(as_tuple=True)[0].item()
        random_reference_out = torch.tensor([random.random() for i in range(0, num_ref_nodes)], dtype=torch.float)

        for i in range(len(k_s)):   
            k = k_s[i]     
            if(num_ref_nodes >= k):
                pred_max_indices = torch.topk(random_reference_out, k, dim=0)[1]
            else:
                pred_max_indices = torch.topk(random_reference_out, num_ref_nodes, dim=0)[1]
            if y_index in pred_max_indices:
                correct_per_k[i] += 1
        
        pred_max_index = torch.argmax(random_reference_out, dim=0)
        if y_index == pred_max_index:
            localization_per_bugtype[data.bugtype[0]] += [1, 1]
            localization_per_refnodes_number[num_ref_nodes] += [1, 1]
        else:
            localization_per_bugtype[data.bugtype[0]] += [0, 1]
            localization_per_refnodes_number[num_ref_nodes] += [0, 1]

    return ([correct/len(data_loader) for correct in correct_per_k], localization_per_bugtype, localization_per_refnodes_number)

def run_training():
    print('\nTrain...')   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    losses = []
    train_accs, valid_accs = np.empty([len(k_s), num_epochs]), np.empty([len(k_s), num_epochs])
    num_epochs_not_improved, best_acc = 0, 0

    for epoch in range (1, num_epochs+1):
        if num_epochs_not_improved > patience:
            print(f'Accuracy has not improved for {num_epochs_not_improved} epochs. Stopping.')
            break
        result = f'Epoch: {epoch:02d}'
        loss = train(train_loader, model, optimizer, criterion, useScaler)
        losses.append(loss)
        train_acc, _, _ = test(train_loader, model, k_s)
        valid_acc, _, _ = test(valid_loader, model, k_s)
        for i in range(len(k_s)):
            train_accs[i][epoch-1] = train_acc[i]
            valid_accs[i][epoch-1] = valid_acc[i]
            result += f', Train Acc Top {k_s[i]}: {train_acc[i]:.4f}'
            result += f', Valid Acc Top {k_s[i]}: {valid_acc[i]:.4f}'
        result += f', Loss: {loss:.4f}'
        print(result)

        if valid_acc[0] > best_acc:
            best_acc = valid_acc[0]
            num_epochs_not_improved = 0
        else:
            num_epochs_not_improved += 1
    
    print('\nGenerating  plots...')
    x_values = [epoch for epoch in range (1, len(losses)+1)]
    plt.figure(0)
    plt.plot(x_values, losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim((min(losses)-0.1, losses[5]+0.1))
    plt.savefig('losses_loc_detail.png')

    plt.figure(1)
    plt.plot(x_values, losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('losses_loc_total.png')
    
    plt.figure(2)
    for i in range(len(k_s)):
        label_training = 'Train Top ' + str(k_s[i])
        label_valid = 'Valid Top ' + str(k_s[i])
        plt.plot(x_values, train_accs[i][:len(losses)], label=label_training)
        plt.plot(x_values, valid_accs[i][:len(losses)], label=label_valid)
    plt.legend(loc='upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim((0, min(max(max(train_accs[-1]), max(valid_accs[-1]))+0.05, 1)))
    plt.savefig('accuracies_loc.png')
    plt.close()

def run_evaluation():
    print('\nEvaluate...') 
    test_acc, test_localization_per_bugtype, test_localization_per_refnodes_number = test(test_loader, model, k_s)
    baseline_acc, baseline_localization_per_bugtype, baseline_localization_per_refnodes_number = baseline(test_loader, k_s)
    eval_file_name = 'evaluation.txt'
    eval_headline = 'Evaluation results:\n'
    mode = 'w+'
    if os.path.isfile(eval_file_name):
        mode = 'a+'
        eval_headline = '\n \nEvaluation results:\n'
    eval_file = open(eval_file_name, mode)
    eval_file.write(eval_headline)

    accs_headline = '(gnn) Accuracies of the localization task: '
    accs = f'Top {k_s[0]}: {test_acc[0]:.4f}'
    for i in range(1, len(k_s)):
        accs += f', Top {k_s[i]}: {test_acc[i]:.4f}'

    eval_file.write(accs_headline)
    eval_file.write(accs)
    print(accs_headline)
    print(accs)

    accs_headline = '\n(gnn) Accuracies of the localization task per bug type: '
    print(accs_headline)
    eval_file.write(accs_headline)
    for bugtype, (num_correct, total) in sorted(test_localization_per_bugtype.items(), key=lambda item: item[0]):
        acc = f'{bugtype}: {num_correct/total:.4f}  ({num_correct}/{total})'
        print(acc)
        eval_file.write('\n' + acc)

    accs_headline = '\n(gnn) Accuracies of the localization task per number of reference nodes: '
    print(accs_headline)
    eval_file.write(accs_headline)
    for refnodes_number, (num_correct, total) in sorted(test_localization_per_refnodes_number.items(), key=lambda item: item[0]):
            acc = f'{refnodes_number}: {num_correct/total:.4f}  ({num_correct}/{total})'
            print(acc)
            eval_file.write('\n' + acc)

    accs_headline = '\n(baseline) Accuracies of the localization task: '
    accs = f'Top {k_s[0]}: {baseline_acc[0]:.4f}'
    for i in range(1, len(k_s)):
        accs += f', Top {k_s[i]}: {baseline_acc[i]:.4f}'

    eval_file.write(accs_headline)
    eval_file.write(accs)
    print(accs_headline)
    print(accs)

    accs_headline = '\n(baseline) Accuracies of the localization task per bug type: '
    print(accs_headline)
    eval_file.write(accs_headline)
    for bugtype, (num_correct, total) in sorted(baseline_localization_per_bugtype.items(), key=lambda item: item[0]):
        acc = f'{bugtype}: {num_correct/total:.4f}  ({num_correct}/{total})'
        print(acc)
        eval_file.write('\n' + acc)

    accs_headline = '\n(baseline) Accuracies of the localization task per number of reference nodes: '
    print(accs_headline)
    eval_file.write(accs_headline)
    for refnodes_number, (num_correct, total) in sorted(baseline_localization_per_refnodes_number.items(), key=lambda item: item[0]):
            acc = f'{refnodes_number}: {num_correct/total:.4f}  ({num_correct}/{total})'
            print(acc)
            eval_file.write('\n' + acc)

    param_headline = '\nTraining parameters: '
    params = f'#Layers: {8}, #Hidden channels: {hidden_channels}, Learning rate: {learning_rate}, #Training graphs: {len(data_list_train)}, #Avg. Reference nodes in evaluation graphs: {avg_ref_nodes_eval_graph}, Use Scaler: {useScaler}'
    bugtypes = f'\nRemoved bugtypes: '
    if len(delBugtypes) > 0:
        bugtypes += f'{delBugtypes[0]}'
        for i in range(1, len(delBugtypes)):
            bugtypes += f', {delBugtypes[i]}'
    else:
        bugtypes += f'-'
    eval_file.write(param_headline)
    eval_file.write(params)
    eval_file.write(bugtypes)
    eval_file.close()
    
if __name__ == "__main__":
    dataset_dir_train = Path(sys.argv[1])
    dataset_dir_valid = Path(sys.argv[2])
    dataset_dir_test = Path(sys.argv[3])

    delBugtypes = []
    # delBugtypes = ['VariableMisuseRewriteScout', 'ArgSwapRewriteScout', 'LiteralRewriteScout']
    # delBugtypes = ['ArgSwapRewriteScout', 'LiteralRewriteScout', 'BooleanOperatorRewriteScout', 'AssignRewriteScout', 'BinaryOperatorRewriteScout', 'ComparisonOperatorRewriteScout', 'VariableMisuseRewriteScout']

    data_list_train, weights, node_label_vocab, edge_attr_vocab, num_nodes_train, num_reference_nodes_train = prepareDataWithoutVocabularies(dataset_dir_train, delBugtypes)
    data_list_valid, num_nodes_valid, num_reference_nodes_valid = prepareDataWithVocablularies(dataset_dir_valid, node_label_vocab, edge_attr_vocab, delBugtypes)
    data_list_test, num_nodes_test, num_reference_nodes_test = prepareDataWithVocablularies(dataset_dir_test, node_label_vocab, edge_attr_vocab, delBugtypes)

    num_epochs = 200
    batch_size = 1
    k_s = [1, 3, 5]
    patience = 50
    hidden_channels = 128
    learning_rate = 0.00001
    useScaler = False

    train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_list_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=True)

    num_nodes = np.concatenate([num_nodes_train, num_nodes_valid, num_nodes_test])
    num_reference_nodes = np.concatenate([num_reference_nodes_test, num_reference_nodes_valid, num_reference_nodes_test])
    print(f'# of Graphs: {len(data_list_train)+len(data_list_valid)+len(data_list_test)}, # Graphs for Training: {len(data_list_train)}, # Graphs for Validation: {len(data_list_valid)}, # Graphs for Evaluation: {len(data_list_test)}') 
    print(f"# Nodes per training graph: Avg: {np.mean(num_nodes_train)} Median: {np.median(num_nodes_train)} Max: {np.max(num_nodes_train)} Min: {np.min(num_nodes_train)}")
    print(f"# Reference nodes per training graph: Avg: {np.mean(num_reference_nodes_train)} Median: {np.median(num_reference_nodes_train)} Max: {np.max(num_reference_nodes_train)} Min: {np.min(num_reference_nodes_train)}")
    print(f"# Nodes per validation graph: Avg: {np.mean(num_nodes_valid)} Median: {np.median(num_nodes_valid)} Max: {np.max(num_nodes_valid)} Min: {np.min(num_nodes_valid)}")
    print(f"# Reference nodes per validation graph: Avg: {np.mean(num_reference_nodes_valid)} Median: {np.median(num_reference_nodes_valid)} Max: {np.max(num_reference_nodes_valid)} Min: {np.min(num_reference_nodes_valid)}")
    print(f"# Nodes per evaluation graph: Avg: {np.mean(num_nodes_test)} Median: {np.median(num_nodes_test)} Max: {np.max(num_nodes_test)} Min: {np.min(num_nodes_test)}")
    print(f"# Reference nodes per evaluation graph: Avg: {np.mean(num_reference_nodes_test)} Median: {np.median(num_reference_nodes_test)} Max: {np.max(num_reference_nodes_test)} Min: {np.min(num_reference_nodes_test)}")
    print(f"# Nodes per graph: Avg: {np.mean(num_nodes)} Median: {np.median(num_nodes)} Max: {np.max(num_nodes)} Min: {np.min(num_nodes)}")
    print(f"# Reference nodes per graph: Avg: {np.mean(num_reference_nodes)} Median: {np.median(num_reference_nodes)} Max: {np.max(num_reference_nodes)} Min: {np.min(num_reference_nodes)}")

    avg_ref_nodes_eval_graph = np.mean(num_reference_nodes_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(1, hidden_channels).to(device)

    run_training()
    run_evaluation()