from utils.msgpackutils import load_msgpack_l_gz
from dpu_utils.utils import RichPath
import torch
from torch_geometric.data import Data
from typing import List, Tuple
from dpu_utils.mlutils import Vocabulary
import numpy as np

def prepareDataWithoutVocabularies(dataset_dir: RichPath, delBugtypes: List[str], restrict_num_ref_nodes: bool) -> Tuple[List, torch.Tensor, Vocabulary, Vocabulary, List, List]:
    """ 
    Creates vocabularies to abstract the node labels and edge types and creates the dataset afterwards. 

        Parameters:
            dataset_dir (RichPath): The path to the folder that contains the data
            delBugtypes: (List[str]): The bug types that should be removed
            restrict_num_ref_nodes: (bool): Flag, if only graphs with maximal 25 reference nodes should be used

        Returns:
            data (Tuple[List, torch.Tensor, Vocabulary, Vocabulary, List, List]): The datalist, the weights, the vocabualaries and number of nodes.
    """
    assert dataset_dir.exists()
    node_label_vocab, edge_attr_vocab = create_vocabs(dataset_dir)
    data_list, weights, num_nodes, num_reference_nodes = prepareData(dataset_dir, node_label_vocab, edge_attr_vocab, delBugtypes, restrict_num_ref_nodes)
    return (data_list, weights, node_label_vocab, edge_attr_vocab, num_nodes, num_reference_nodes)

def prepareDataWithVocablularies(dataset_dir: RichPath, node_label_vocab: Vocabulary, edge_attr_vocab: Vocabulary, delBugtypes: List[str], restrict_num_ref_nodes: bool) -> Tuple[List, List, List]:
    """ 
    Creates the dataset using the given vocabularies. 

        Parameters:
            dataset_dir (RichPath): The path to the folder that contains the data
            node_label_vocab (Vocabulary): The vocabulary of the node labels
            edge_attr_vocab (Vocabulary): The vocabulary of the edge attributes
            delBugtypes: (List[str]): The bug types that should be removed
            restrict_num_ref_nodes: (bool): Flag, if only graphs with maximal 25 reference nodes should be used

        Returns:
            data (Tuple[List, List, List]): The datalist, the weights and number of nodes.
    """
    assert dataset_dir.exists()
    data_list, _, num_nodes, num_reference_nodes = prepareData(dataset_dir, node_label_vocab, edge_attr_vocab, delBugtypes, restrict_num_ref_nodes)
    return (data_list, num_nodes, num_reference_nodes)

def prepareData(dataset_dir: RichPath, node_label_vocab: Vocabulary, edge_attr_vocab: Vocabulary, delBugtypes: List[str], restrict_num_ref_nodes: bool) -> Tuple[List, torch.Tensor, List, List]:
    """ 
    Converts the given graphs into Data objects such that they can be used by the graph neural network.

        Parameters:
            dataset_dir (RichPath): The path to the folder that contains the data
            node_label_vocab (Vocabulary): The vocabulary of the node labels
            edge_attr_vocab (Vocabulary): The vocabulary of the edge attributes
            delBugtypes: (List[str]): The bug types that should be removed
            restrict_num_ref_nodes: (bool): Flag, if only graphs with maximal 25 reference nodes should be used

        Returns:
            data (Tuple[List, torch.Tensor, List, List]): The datalist, the weights and number of nodes.
    """
    print(restrict_num_ref_nodes)
    datalist, num_nodes, num_reference_nodes = [], [], []
    num_unique_ref_nodes = 0

    # Create for each graph in every file a data object
    for pkg_file in dataset_dir.rglob("*.msgpack.l.gz"):
        try:
            for graph in load_msgpack_l_gz(pkg_file):
                if graph is None:
                    continue
                if graph["target_fix_action_idx"] is None:
                    continue

                # get the bugtype of the graph
                bugtype = graph["candidate_rewrite_metadata"][graph["target_fix_action_idx"]][0]

                if bugtype in delBugtypes:
                    continue

                # get the unique reference nodes
                unique_reference_nodes = np.unique(graph["graph"]["reference_nodes"])
                if restrict_num_ref_nodes and len(unique_reference_nodes) > 25:
                    print('test')
                    continue
                num_unique_ref_nodes += len(unique_reference_nodes)
                num_reference_nodes.append(len(unique_reference_nodes))

                # tensorize the nodes
                nodes = []
                for node in graph["graph"]["nodes"]:
                    nodes.append([node_label_vocab.get_id_or_unk(node)])
                num_nodes.append(len(graph["graph"]["nodes"]))
                x = torch.tensor(nodes, dtype=torch.float)

                # tensorize the edges
                outgoing_edges, incoming_edges, edge_attributes = [], [], []
                for key in graph["graph"]["edges"].keys():
                    for edge in graph["graph"]["edges"][key]:
                        outgoing_edges.append(edge[0])
                        incoming_edges.append(edge[1])
                        edge_attributes.append([edge_attr_vocab.get_id_or_unk(key)])
                edge_index = torch.tensor([outgoing_edges,
                           incoming_edges], dtype=torch.long)
                edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

                # create the target vector
                y_val = [0] * len(graph["graph"]["nodes"])
                y_val[graph["graph"]["reference_nodes"][graph["target_fix_action_idx"]]] = 1
                y = torch.tensor(y_val, dtype=torch.long)

                # add the mask to identify the reference during training and evaluation
                mask_val = [0] * len(graph["graph"]["nodes"])
                for node_id in unique_reference_nodes:
                    mask_val[node_id] = 1
                mask = torch.tensor(mask_val, dtype=torch.int)

                # create the data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask, bugtype=bugtype)
                datalist.append(data)
        except Exception as e:
            print(f"Error loading {pkg_file}: {e} Skipping...")

    # weight the classes, because they are not equally distributed
    weight_buggy_class = 1 - (len(datalist)/num_unique_ref_nodes)
    weights = torch.tensor([1 - weight_buggy_class, weight_buggy_class], dtype=torch.float)
    return (datalist, weights, num_nodes, num_reference_nodes)

def create_vocabs(dataset_dir: RichPath) -> Tuple[Vocabulary, Vocabulary]:
    """ 
    Create the vocabularies to abstract the node labels and edge types such that they could be tensorized.

        Parameters:
            dataset_dir (RichPath): The path to the folder that contains the data

        Returns:
            data (Tuple[Vocabulary, Vocabulary]): The vocabularies
    """
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