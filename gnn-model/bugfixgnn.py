from typing_extensions import TypedDict
from typing import Any, Counter, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

# BuglabGraph = TypedDict(
#     "BuglabGraph", 
#     {
#         "nodes": List[str],
#         "edges": Dict[str, List[Union[Tuple[int, int], Tuple[int, int, str]]]],
#         "path": str,
#         "reference_nodes": List[int],
#         "code_range": Tuple[Tuple[int, int], Tuple[int, int]]
#     }
# )

# Prediction = Tuple[BuglabGraph, Dict[int, Tuple[str, float]]]

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