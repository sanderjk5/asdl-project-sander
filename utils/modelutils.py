from typing import Optional
import torch
import torch_scatter
from torch.cuda.amp import autocast

def scatter_log_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError("`scatter_log_softmax` can only be computed over tensors with floating point data types.")

    with autocast(enabled=False):
        max_value_per_index = torch_scatter.scatter_max(src.float(), index, dim=dim)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = torch.zeros_like(max_value_per_index).scatter_add_(dim=-1, index=index, src=recentered_scores.exp())
    normalizing_constants = sum_per_index.add_(eps).log_().gather(dim, index)

    return recentered_scores.sub_(normalizing_constants)

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None):
    with autocast(enabled=False):
        return torch_scatter.scatter_max(src.float(), index, dim, dim_size=dim_size)