import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange, einsum

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Create Q, K, V
        self.raw_to_queries = nn.Linear(1152, 1152)
        self.raw_to_keys = nn.Linear(1152, 1152)
        self.raw_to_values = nn.Linear(1152, 1152)

        # Consolidation layer after attention, output projection
        self.consolidation_after_attention = nn.Linear(1152, 1152)

        # RMSNorm learnable weights
        self.v1 = nn.Parameter(torch.ones(1152))

        # FFN 1152 -> 4304 after RMSNorm
        self.after_rms_to_4304 = nn.Linear(1152, 4304)

        # FFN 4304 -> 1152
        self.after_activation_to_1152 = nn.Linear(4304, 1152)

    def RMSNorm(self, tensor):
        pass

    def forward(self, o4):
        ## o4: [BV*6, 1024, 1152]
        queries = self.raw_to_queries(o4)
        keys = self.raw_to_keys(o4)
        values = self.raw_to_values(o4)

        # Split into 16 heads
        self.split_queries = rearrange(queries, 'a s (h d) -> a h s d', h=16)
        self.split_keys = rearrange(keys, 'a s (h d) -> a h s d', h=16)
        self.split_values = rearrange(values, 'a s (h d) -> a h s d', h=16)

        # Attention scores + softmax 
        self.attention_scores = einsum(self.split_queries, self.split_keys, 'b h q d, b h k d -> b h q k')
        self.attention_scores = self.attention_scores / (72 ** 0.5)
        self.attention_scores = torch.softmax(self.attention_scores, dim=-1)

        # Weight the values
        self.weighted_values = einsum(self.attention_scores, self.split_values, 'b h q k, b h k d -> b h q d')
        self.weighted_reconcatenated_values = rearrange(self.weighted_values, "b h p d -> b p (h d)")

        # MLP after attention, output projection
        self.output_projection = self.consolidation_after_attention(self.weighted_reconcatenated_values)

        # Residual 1
        self.after_residual_1 = values + self.output_projection

        ## RMSNorm across all BV * 6 * 1024 "patch" vectors
        rms = torch.sqrt(self.after_residual_1.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        self.after_rms = (self.after_residual_1 / rms) * self.v1

        # Linear, 1152 -> 4304
        self.before_activation = self.after_rms_to_4304(self.after_rms)

        # Activation? GeLU!
        self.activated = F.gelu(self.before_activation)

        # Linear, 4304 -> 1152
        self.after_activation = self.after_activation_to_1152(self.activated)

        # Residual 2
        self.after_residual_2 = values + self.after_activation

        # Reshape
        self.final = rearrange(self.after_residual_2, '(b v k) p d -> b v k p d', b=2, v=4, k=6)

        return self.final.shape

ST = SpatialTransformer()

o4_shape = ([48, 1024, 1152])
o4 = torch.zeros(o4_shape)

print(ST.forward(o4))
