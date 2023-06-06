from utils.msgpackutils import load_msgpack_l_gz
from dpu_utils.utils import RichPath
from dpu_utils.codeutils import split_identifier_into_parts
import torch
from torch_geometric.data import Data
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import re
from dpu_utils.mlutils import Vocabulary

IS_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

def prepareData(dataset_dir: RichPath):
    assert dataset_dir.exists()

    datalist = []
    num_diff_nodes, num_diff_edge_attr = 0, 0
    node_label_vocab, edge_attr_vocab = create_vocabs(dataset_dir)
    num_bug = 0
    num_no_bug = 0

    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                nodes = []
                for node in graph["graph"]["nodes"]:
                    nodes.append([node_label_vocab.get_id_or_unk(node)])

                # token_nodes = set()
                # if "NextToken" in graph["graph"]["edges"]:
                #     for n1, n2 in graph["graph"]["edges"]["NextToken"]:
                #         token_nodes.add(n1)
                #         token_nodes.add(n2)

                #     subtoken_idxs: Dict[int, int] = {}
                #     vocab_edges: List[Tuple[int, int]] = []

                #     all_nodes = graph["graph"]["nodes"]
                #     for node_idx in token_nodes:
                #         token_str = all_nodes[node_idx]
                #         if not IS_IDENTIFIER.match(token_str):
                #             continue
                #         for subtoken in split_identifier_into_parts(token_str):
                #             subtoken_node_value = node_label_vocab.get(subtoken)
                #             if subtoken_node_value is None:
                #                 subtoken_node_value = num_diff_nodes
                #                 node_label_vocab[subtoken] = subtoken_node_value
                #                 num_diff_nodes += 1
                #             subtoken_idx = subtoken_idxs.get(subtoken_node_value)
                #             if subtoken_idx is None:
                #                 subtoken_idx = len(all_nodes)
                #                 all_nodes.append(subtoken)
                #                 nodes.append([subtoken_node_value])
                #                 subtoken_idxs[subtoken_node_value] = subtoken_idx
                #             vocab_edges.append((node_idx, subtoken_idx))

                #     graph["graph"]["edges"]["HasSubtoken"] = vocab_edges
  
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
                if graph["target_fix_action_idx"] is None:
                    y_val = 0
                    num_no_bug += 1
                else:
                    y_val = 1
                    num_bug += 1
                y = torch.tensor([y_val], dtype=torch.long)
                # if y_val == 1 and num_bug > 209:
                #     continue
                # if y_val == 1 and num_bug > 12605:
                #     continue
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                datalist.append(data)
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")
    print(num_no_bug)
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
    node_vocab = Vocabulary.create_vocabulary(node_labels, max_size=10000, count_threshold=0, add_unk=True)
    edge_type_vocab = Vocabulary.create_vocabulary(edge_types, max_size=10000, count_threshold=0, add_unk=True)
    return node_vocab, edge_type_vocab