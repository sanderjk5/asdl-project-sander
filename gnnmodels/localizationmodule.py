import math
from typing import Any, Callable, Dict, Union

import torch
import torch.nn as nn
from ptgnn.baseneuralmodel import ModuleWithMetrics

from utils.modelutils import scatter_log_softmax, scatter_max

class LocalizationModule(ModuleWithMetrics):
    def __init__(
            self,
            representation_size: int,
            buggy_samples_weight_schedule: Callable[[int], float],
            abstain_weight: float = 0.0
        ):
        super().__init__()