from utils.msgpackutils import load_msgpack_l_gz
from dpu_utils.utils import RichPath
import torch
from torch_geometric.data import Data

def prepareData(dataset_dir: RichPath):
    assert dataset_dir.exists()

    datalist = []
    num_diff_nodes, num_diff_edge_attr = 0, 0
    node_idx, edge_attr_idx = {}, {}

    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                nodes = []
                for node in graph["graph"]["nodes"]:
                    if node in node_idx.keys():
                        nodes.append([node_idx[node]])
                    else:
                        nodes.append([num_diff_nodes])
                        node_idx[node] = num_diff_nodes
                        num_diff_nodes += 1
                
                x = torch.tensor(nodes, dtype=torch.float)
                outgoing_edges, incoming_edges, edge_attributes = [], [], []
                for key in graph["graph"]["edges"].keys():
                    for edge in graph["graph"]["edges"][key]:
                        outgoing_edges.append(edge[0])
                        incoming_edges.append(edge[1])
                        if key in edge_attr_idx.keys():
                            edge_attributes.append([edge_attr_idx[key]])
                        else:
                            edge_attributes.append([num_diff_edge_attr])
                            edge_attr_idx[key] = num_diff_edge_attr
                            num_diff_edge_attr += 1
                edge_index = torch.tensor([outgoing_edges,
                           incoming_edges], dtype=torch.long)
                edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
                if graph["target_fix_action_idx"] is None:
                    y_val = 0
                else:
                    y_val = 1
                y = torch.tensor([y_val], dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                datalist.append(data)
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")
    return datalist