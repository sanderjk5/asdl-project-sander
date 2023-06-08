from utils.msgpackutils import load_msgpack_l_gz
from dpu_utils.utils import RichPath
from dpu_utils.codeutils import split_identifier_into_parts
import torch
from torch_geometric.data import Data
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import re
from dpu_utils.mlutils import Vocabulary

def prepareData(dataset_dir: RichPath, prepareForLocalization: bool):
    assert dataset_dir.exists()

    datalist = []
    node_label_vocab, edge_attr_vocab = create_vocabs(dataset_dir)
    num_bug = 0
    num_no_bug = 0

    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                if prepareForLocalization and graph["target_fix_action_idx"] is None:
                    continue

                nodes = []
                for node in graph["graph"]["nodes"]:
                    nodes.append([node_label_vocab.get_id_or_unk(node)])

                x = torch.tensor(nodes, dtype=torch.float)

                outgoing_edges, incoming_edges, edge_attributes = [], [], []
                for key in graph["graph"]["edges"].keys():
                    for edge in graph["graph"]["edges"][key]:
                        outgoing_edges.append(edge[0])
                        incoming_edges.append(edge[1])
                        edge_attributes.append([edge_attr_vocab.get_id_or_unk(key)])

                edge_index = torch.tensor([outgoing_edges,
                           incoming_edges], dtype=torch.long)
                edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

                if prepareForLocalization:
                    y_val = [0] * len(graph["graph"]["nodes"])
                    y_val[graph["graph"]["reference_nodes"][graph["target_fix_action_idx"]]] = 1

                    mask_val = [0] * len(graph["graph"]["nodes"])
                    for node_id in graph["graph"]["reference_nodes"]:
                        mask_val[node_id] = 1
                    mask = torch.tensor(mask_val, dtype=torch.int)
                    # y_val = []
                    # for i in range(len(graph["graph"]["nodes"])):
                    #     if i == graph["target_fix_action_idx"]:
                    #         y_val.append(1)
                    #     else:
                    #         y_val.append(0)
                    # if graph["target_fix_action_idx"] < 5:
                        # print(graph["target_fix_action_idx"])
                        # print(y_val)
                else:
                    if graph["target_fix_action_idx"] is None:
                        y_val = [0]
                        num_no_bug += 1
                    else:
                        # if num_bug > 209:
                        #     continue
                        # if num_bug > 12605:
                        #     continue
                        y_val = [1]
                        num_bug += 1

                y = torch.tensor(y_val, dtype=torch.long)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
                datalist.append(data)
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")
    # print(num_no_bug)
    return datalist

def create_vocabs(dataset_dir: RichPath) -> Tuple[Vocabulary, Vocabulary]:
    node_labels, edge_types = set(), set()
    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                for node in graph["graph"]["nodes"]:
                    node_labels.add(node)

                for key in graph["graph"]["edges"].keys():
                    edge_types.add(key)
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")
    node_vocab = Vocabulary.create_vocabulary(node_labels, max_size=len(node_labels)+1, count_threshold=0, add_unk=True)
    edge_type_vocab = Vocabulary.create_vocabulary(edge_types, max_size=len(edge_types)+1, count_threshold=0, add_unk=True)
    return node_vocab, edge_type_vocab