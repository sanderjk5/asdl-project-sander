from typing_extensions import TypedDict
from typing import Any, Counter, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
from ptgnn.neuralmodels.gnn import (
    GnnOutput,
    GraphData,
    GraphNeuralNetwork,
    GraphNeuralNetworkModel,
    TensorizedGraphData,
)
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
import numpy as np
from dpu_utils.mlutils import Vocabulary
import re
from dpu_utils.codeutils import split_identifier_into_parts

class BugLabGraph(TypedDict):
    nodes: List[str]
    edges: Dict[str, List[Union[Tuple[int, int], Tuple[int, int, str]]]]
    path: str
    text: str
    reference_nodes: List[int]
    code_range: Tuple[Tuple[int, int], Tuple[int, int]]

class BugLabData(TypedDict):
    graph: BugLabGraph
    candidate_rewrites: List[Tuple[str, Any]]
    candidate_rewrite_metadata: List[Tuple[str, Any]]
    candidate_rewrite_ranges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    target_fix_action_idx: Optional[int]
    package_name: str
    candidate_rewrite_logprobs: Optional[List[float]]

class BaseTensorizedBugLabGnn(NamedTuple):
    graph_data: TensorizedGraphData

    # Bug localization
    target_location_node_idx: Optional[int]

    # Bug repair
    target_rewrites: List[int]
    target_rewrite_to_location_group: List[int]
    correct_rewrite_target: Optional[int]
    text_rewrite_original_idx: List[int]

    num_rewrite_locations_considered: int

    # Bug selection
    rewrite_logprobs: Optional[List[float]]

class BugFixModule(ModuleWithMetrics):
    pass

IS_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

class BugFixGnn(AbstractNeuralModel[BugLabData, BaseTensorizedBugLabGnn, BugFixModule]):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel[str, Any],
        max_num_classes: int = 100
    ):
        super().__init__()
        self.__gnn_model = gnn_model
        self.max_num_classes = max_num_classes

    def __convert(self, buglab_data: BugLabData) -> Tuple[GraphData[str, None], Optional[int]]:
        token_nodes = set()
        if "NextToken" not in buglab_data["graph"]["edges"]:
            return
        for n1, n2 in buglab_data["graph"]["edges"]["NextToken"]:
            token_nodes.add(n1)
            token_nodes.add(n2)

        vocab_nodes: Dict[str, int] = {}
        vocab_edges: List[Tuple[int, int]] = []

        all_nodes = buglab_data["graph"]["nodes"]
        for node_idx in token_nodes:
            token_str = all_nodes[node_idx]
            if not IS_IDENTIFIER.match(token_str):
                continue
            for subtoken in split_identifier_into_parts(token_str):
                subtoken_node_idx = vocab_nodes.get(subtoken)
                if subtoken_node_idx is None:
                    subtoken_node_idx = len(all_nodes)
                    all_nodes.append(subtoken)
                    vocab_nodes[subtoken] = subtoken_node_idx
                vocab_edges.append((node_idx, subtoken_node_idx))

        buglab_data["graph"]["edges"]["HasSubtoken"] = vocab_edges

        edges = {}
        edge_features = {}
        for edge_type, adj_list in buglab_data["graph"]["edges"].items():
            edges_of_type: List[Tuple[int, int]] = []
            edge_features_of_type: List[str] = []
            for edge in adj_list:
                edges_of_type.append((edge[0], edge[1]))
                if len(edge) >= 3:
                    edge_features_of_type.append(edge[2])
                else:
                    edge_features_of_type.append(Vocabulary.get_pad())
            edges[edge_type] = np.array(edges_of_type, dtype=np.int32)
            edge_features[edge_type] = edge_features_of_type
        
        candidate_node_idxs, inv = np.unique(buglab_data["graph"]["reference_nodes"], return_inverse=True)
        if buglab_data["target_fix_action_idx"] is not None:
            target_node_idx = inv[buglab_data["target_fix_action_idx"]]
        else:
            target_node_idx = None
        
        return (GraphData[str](
            node_information=buglab_data["graph"]["nodes"],
            edges=edges,
            edge_features=edge_features,
            reference_nodes=candidate_node_idxs
        ), target_node_idx)
    
    def initialize_metadata(self) -> None:
        return super().initialize_metadata()