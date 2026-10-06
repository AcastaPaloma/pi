### CONGLOMERATES VISUAL SUBGOALS, FULL OBSERVATION, TEXT PROMPT AND PROPRIOCEPTION

import torch 
import torch.nn as nn
import numpy as np

from einops import rearrange, reduce

from spatial_transformer import SpatialTransformer
from temporal_transformer import TemporalTransformer

BATCH_SIZE = 2

# Image captures, for 6 past frames, by 4 cameras
# [B, V, K, C, width, height]
full_observation__shape = (BATCH_SIZE, 4, 6, 3, 448, 448)
full_observation = torch.zeros(full_observation__shape)

raw_text = "Pick up the red cube and put it in the cup."

# [B, V, K, C, width, height]
visual_subgoals_shape = (BATCH_SIZE, 3, 1, 3, 448, 488)
visual_subgoals = torch.zeros(visual_subgoals_shape)

# [B, 6 frames into the past, 2560]
proprioception_shape = (BATCH_SIZE, 6, 12)
proprioception = torch.zeros(proprioception_shape)

class SpatialSemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Should accept visual subgoals and full observation frames into [B, S, 2560]
        """

        ### Full observation
        #. patchification
        self.patchified_obs_to_o1 = nn.Linear(588, 1152)

        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, 1, 1024, 1152) * 0.02)
        self.temporal_embedding = nn.Parameter(torch.randn(1, 1, 6, 1, 1152) * 0.02)

        self.o6_to_o7 = nn.Linear(1152, 2560)
        self.spatial_transformers = nn.ModuleList([SpatialTransformer() for _ in range(27)])
        self.temporal_transformers = nn.ModuleList([TemporalTransformer() for _ in range(6)])

    def patchify(self, full_observation):
        """
        [B, V, K, C, width, height] -> [B, V, 6, 1024, 1152]
        """

        patchified_obs = rearrange(
            full_observation,
            "b v k c (h p1) (w p2) -> b v k (h w) (c p1 p2)",
            p1=14,
            p2=14,
        )
        return patchified_obs

    
    def forward(self, full_observation, raw_text, visual_subgoals, proprioception):
        full_observation = full_observation
        raw_text = raw_text
        visual_subgoals = visual_subgoals
        proprioception = proprioception

        ### Full observation: [B, V, K, C, width, height] -> [B, 256V, 2560]
        patchified_obs = self.patchify(full_observation)

        ### Linear projection (588, 1152)
        o1 = self.patchified_obs_to_o1(patchified_obs)

        ### Spatial & Temporal Embeddings
        o2 = o1 + self.spatial_embedding
        o3 = o2 + self.temporal_embedding

        ### Rearrange for spatial transformer
        o4 = rearrange(o3, "b v k p d -> (b v k) p d")

        ### Spatial & Temporal transformer (bad naming o4 lol)
        for layer_index, spatial_transformer in enumerate(self.spatial_transformers, start=1):
            o4 = spatial_transformer(o4)
            if layer_index % 4 == 0:
                o4 = self.temporal_transformers[layer_index // 4 - 1](o4)

        o4 = rearrange(o4, '(b v k) p d -> b v k p d',
                       b=full_observation.shape[0], v=full_observation.shape[1],
                       k=full_observation.shape[2])

        # Drop previous 4 dimensions, as attention has mixed everything up already lol
        o5 = o4[..., -1:, :, :]
        
        # o5 -> o6, average downsizing pooling with 2x2 kernel
        o6 = reduce(o5, 'b v k (h p1 w p2) d -> b v k (h w) d', 'mean', h=16, w=16, p1=2, p2=2)
        o7 = self.o6_to_o7(o6)

        return o7.shape

SSEnc = SpatialSemanticEncoder()
print(SSEnc.forward(full_observation=full_observation, raw_text="", visual_subgoals=None, proprioception=None))
