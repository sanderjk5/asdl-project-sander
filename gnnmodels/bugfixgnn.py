from typing_extensions import TypedDict
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
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
import torch

from gnnmodels.bugfixgnn import BugLabData

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

Prediction = Tuple[BugLabData, Dict[int, float], List[float]]

class BugFixModule(ModuleWithMetrics):
    def __init__(self, gnn: GraphNeuralNetwork):
        super().__init__()
        self.__gnn = gnn

    def forward(self):
        pass


IS_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

class BugFixGnn(AbstractNeuralModel[BugLabData, BaseTensorizedBugLabGnn, BugFixModule]):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel[str, Any],
    ):
        super().__init__()
        self.__gnn_model = gnn_model

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
    
    def update_metadata_from(self, datapoint: BugLabData) -> None:
        graph_data, _ = self.__convert(datapoint)
        self.__gnn_model.update_metadata_from(graph_data)
    
    def build_neural_module(self) -> BugFixModule:
        return BugFixModule(gnn=self.__gnn_model.build_neural_module())

    def tensorize(self, datapoint: BugLabData) -> Optional[TensorizedGraphData]:
        graph_data, target_node_idx = self.__convert(datapoint)
        graph_tensorized_data = self.__gnn_model.tensorize(graph_data)

        if graph_tensorized_data is None:
            return None
        
        return BaseTensorizedBugLabGnn(graph_data=graph_tensorized_data)
    
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {"graph_mb_data": self.__gnn_model.initialize_minibatch()}
    
    def extend_minibatch_with(self, tensorized_datapoint: BaseTensorizedBugLabGnn, partial_minibatch: Dict[str, Any]) -> bool:
        continue_extending = self.__gnn_model.extend_minibatch_with(tensorized_datapoint.graph_data, partial_minibatch["graph_mb_data"])
        return continue_extending
    
    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any], device: Any) -> Dict[str, Any]:
        return {"graph_mb_data": self.__gnn_model.finalize_minibatch(accumulated_minibatch_data["graph_mb_data"], device)}
    
    def predict(self, data: Iterator[BugLabData], trained_network: BugFixModule, device: Union[str, torch.device]) -> Iterator[Prediction]:
        trained_network.eval()
        with torch.no_grad():
            
            for mb_data, original_datapoints in self.minibatch_iterator(
                self.tensorize_dataset(data, return_input_data=True, parallelize=False),
                device,
                max_minibatch_size=50,
                parallelize=False
            ):
                current_graph_idx = 0
                graph_preds: Dict[int, Tuple[str, float]] = {}

                probs, predictions, graph_idxs = trained_network.predict(mb_data["graph_mb_data"])


