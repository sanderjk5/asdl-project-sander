import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from pathlib import Path
from docopt import docopt

from utils.datapreparation import prepareData

class GCN(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def train(data, model, optimizer): 
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss
    
if __name__ == "__main__":
    # args = docopt(__doc__)
    # dataset_dir = Path(args["ALL_DATA_FOLDER"])

    dataset_dir = Path("C:\\Users\\Jurek\\git\\asdl-project-sander\\target")

    
    prepareData(dataset_dir)
    
