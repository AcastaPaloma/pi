import torch
import torch.nn as nn

import numpy as np

class TemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, o4):
        return o4

TT = TemporalTransformer()
