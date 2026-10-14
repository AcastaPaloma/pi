// Each supplied drawing is segmented into named stops and optional components.
// Coordinates are logical; source layers retain their native SVG coordinates.
export const artwork = {
  "src": "./assets/full-backbone.svg",
  "width": 2048,
  "height": 1591.4394510756329,
  "nativeWidth": 4118.032888760241,
  "nativeHeight": 3200,
  "layers": [
    {
      "id": "base",
      "src": "./assets/full-backbone.svg",
      "x": 0,
      "y": 0,
      "scale": 1,
      "clip": "M0 0H4118.033V571.427H2670.249V2257.114H0Z"
    },
    {
      "id": "decision",
      "src": "./assets/decision-model.svg",
      "x": 2670.249,
      "y": 571.427,
      "scale": 1
    },
    {
      "id": "vlm-loss",
      "src": "./assets/loss-vlm.svg",
      "x": 2670,
      "y": 2320,
      "scale": 0.64
    },
    {
      "id": "eam-loss",
      "src": "./assets/loss-eam.svg",
      "x": 3420,
      "y": 2320,
      "scale": 0.63,
      "clip": "M0 0H1105V630H0Z"
    }
  ]
};

export const macros = [
  {id:'ingestion',title:'Ingestion',entry:'context'},
  {id:'decision',title:'Decision model',entry:'decision'},
];

export const steps = [
  {
    "id": "context",
    "title": "Spatial & semantic context ingestion",
    "macro": "ingestion",
    "bounds": [
      2.487,
      285.962,
      1312.938,
      830.035
    ],
    "details": []
  },
  {
    "id": "observation",
    "title": "Full observation",
    "macro": "ingestion",
    "bounds": [
      91,
      350,
      686,
      497
    ],
    "details": [
      "observation-input",
      "patches",
      "embeddings",
      "vision-backbone",
      "pooling",
      "observation-output"
    ]
  },
  {
    "id": "subgoals",
    "title": "Visual subgoals",
    "macro": "ingestion",
    "bounds": [
      801,
      350,
      410,
      431
    ],
    "details": [
      "goal-input",
      "goal-patches",
      "goal-backbone",
      "goal-embeddings",
      "goal-output"
    ]
  },
  {
    "id": "text",
    "title": "Text prompt",
    "macro": "ingestion",
    "bounds": [
      91,
      864,
      686,
      190
    ],
    "details": [
      "prompt-parts",
      "tokenizer",
      "text-output"
    ]
  },
  {
    "id": "proprioception",
    "title": "Proprioception",
    "macro": "ingestion",
    "bounds": [
      801,
      797,
      405,
      158
    ],
    "details": [
      "state-history",
      "state-output"
    ]
  },
  {
    "id": "together",
    "title": "Together",
    "macro": "ingestion",
    "bounds": [
      801,
      969,
      458,
      94
    ],
    "details": [
      "context-output"
    ]
  },
  {
    "id": "decision",
    "title": "Decision model",
    "macro": "decision",
    "bounds": [
      1327.857,
      284.47,
      720.126,
      837.992
    ],
    "details": []
  },
  {
    "id": "vlm",
    "title": "VLM backbone · X context",
    "macro": "decision",
    "bounds": [
      1333.825,
      290.935,
      319.283,
      181.026
    ],
    "frame": [
      1317.911,
      280.989,
      377.967,
      201.417
    ],
    "details": [
      "vlm-context"
    ]
  },
  {
    "id": "flow",
    "title": "Flow-matching model",
    "macro": "decision",
    "bounds": [
      1733.177,
      291.432,
      309.833,
      178.042
    ],
    "frame": [
      1670.017,
      280.989,
      377.967,
      201.417
    ],
    "details": [
      "action-chunk",
      "noise-time",
      "velocity"
    ]
  },
  {
    "id": "l1",
    "title": "L1",
    "macro": "decision",
    "bounds": [
      1650.124,
      347.133,
      88.026,
      16.909
    ],
    "frame": [
      1610.338,
      312.32,
      167.598,
      86.535
    ],
    "maxScale": 4.5,
    "details": []
  },
  {
    "id": "l2",
    "title": "L2",
    "macro": "decision",
    "bounds": [
      1650.124,
      368.518,
      88.026,
      16.909
    ],
    "frame": [
      1610.338,
      333.705,
      167.598,
      86.535
    ],
    "maxScale": 4.5,
    "details": []
  },
  {
    "id": "l33",
    "title": "L33",
    "macro": "decision",
    "bounds": [
      1650.124,
      398.357,
      88.026,
      16.909
    ],
    "frame": [
      1610.338,
      363.544,
      167.598,
      86.535
    ],
    "maxScale": 4.5,
    "details": []
  },
  {
    "id": "l34",
    "title": "L34",
    "macro": "decision",
    "bounds": [
      1650.124,
      419.245,
      88.026,
      16.909
    ],
    "frame": [
      1610.338,
      384.432,
      167.598,
      86.535
    ],
    "maxScale": 4.5,
    "details": []
  },
  {
    "id": "transformer",
    "title": "Transformer layer",
    "macro": "decision",
    "bounds": [
      1329.847,
      476.437,
      718.137,
      646.025
    ],
    "details": [
      "transformer-fast-input",
      "context-qkv",
      "action-qkv",
      "rope",
      "concatenate",
      "attention-mask",
      "context-attention",
      "fast-attention",
      "action-attention",
      "attention-output",
      "context-residual",
      "action-residual",
      "adaptive-rmsnorm",
      "ff-activation",
      "transformer-vlm-output",
      "transformer-eam-output"
    ]
  },
  {
    "id": "eam-loss",
    "title": "EAM Loss",
    "macro": "decision",
    "bounds": [
      1700.851,
      1153.794,
      346.213,
      197.388
    ],
    "details": []
  },
  {
    "id": "eam-projection",
    "title": "Linear(D, d)",
    "macro": "decision",
    "bounds": [
      1823.044,
      1191.391,
      104.96,
      61.41
    ],
    "details": [
      "eam-output",
      "eam-linear",
      "eam-prediction"
    ]
  },
  {
    "id": "flow-loss",
    "title": "Flow loss",
    "macro": "decision",
    "bounds": [
      1713.383,
      1255.308,
      327.414,
      76.762
    ],
    "details": [
      "eam-target",
      "eam-squared-error",
      "eam-mean"
    ]
  },
  {
    "id": "vlm-loss",
    "title": "VLM Loss",
    "macro": "decision",
    "bounds": [
      1327.857,
      1153.794,
      353.936,
      417.594
    ],
    "details": []
  },
  {
    "id": "fast-input",
    "title": "FAST tokens → Gemma input",
    "macro": "decision",
    "bounds": [
      1484.455,
      1215.223,
      148.004,
      93.258
    ],
    "details": [
      "loss-context",
      "fast-tokenizer",
      "fast-embedding",
      "fast-concatenate"
    ]
  },
  {
    "id": "fast-transformer",
    "title": "34 transformer layers",
    "macro": "decision",
    "bounds": [
      1483.818,
      1310.71,
      188.745,
      42.014
    ],
    "details": [
      "fast-mask",
      "fast-positions",
      "fast-meaning"
    ]
  },
  {
    "id": "token-probabilities",
    "title": "Logits → probabilities",
    "macro": "decision",
    "bounds": [
      1495.277,
      1352.724,
      127.633,
      53.154
    ],
    "details": [
      "vocabulary-projection",
      "fast-logits",
      "fast-log-softmax",
      "fast-probabilities"
    ]
  },
  {
    "id": "cross-entropy",
    "title": "Cross-entropy",
    "macro": "decision",
    "bounds": [
      1494.004,
      1410.015,
      183.016,
      133.363
    ],
    "details": [
      "target-probability",
      "negative-log-likelihood",
      "zero-loss",
      "token-confidence",
      "mean-cross-entropy"
    ]
  }
];

export const details = [
  {
    "id": "observation-input",
    "step": "observation",
    "title": "Full observation input",
    "bounds": [
      109,
      386,
      438,
      143
    ]
  },
  {
    "id": "patches",
    "step": "observation",
    "title": "Patchification",
    "bounds": [
      111,
      534,
      187,
      227
    ]
  },
  {
    "id": "embeddings",
    "step": "observation",
    "title": "Positional / temporal embeddings",
    "bounds": [
      305,
      534,
      258,
      115
    ]
  },
  {
    "id": "vision-backbone",
    "step": "observation",
    "title": "SigLIP / MEM backbone",
    "bounds": [
      305,
      685,
      449,
      146
    ]
  },
  {
    "id": "pooling",
    "step": "observation",
    "title": "Frame selection, pooling & projection",
    "bounds": [
      596,
      469,
      152,
      217
    ]
  },
  {
    "id": "observation-output",
    "step": "observation",
    "title": "Observation tensor",
    "bounds": [
      606,
      437,
      122,
      29
    ]
  },
  {
    "id": "goal-input",
    "step": "subgoals",
    "title": "Visual subgoal input",
    "bounds": [
      817,
      389,
      190,
      143
    ]
  },
  {
    "id": "goal-patches",
    "step": "subgoals",
    "title": "Patchification",
    "bounds": [
      815,
      537,
      187,
      225
    ]
  },
  {
    "id": "goal-backbone",
    "step": "subgoals",
    "title": "SigLIP / MEM backbone",
    "bounds": [
      1024,
      548,
      168,
      72
    ]
  },
  {
    "id": "goal-embeddings",
    "step": "subgoals",
    "title": "Positional embeddings",
    "bounds": [
      1023,
      643,
      169,
      119
    ]
  },
  {
    "id": "goal-output",
    "step": "subgoals",
    "title": "Subgoal tensor",
    "bounds": [
      1050,
      411,
      159,
      139
    ]
  },
  {
    "id": "prompt-parts",
    "step": "text",
    "title": "Task, subtask & metadata",
    "bounds": [
      108,
      885,
      382,
      133
    ]
  },
  {
    "id": "tokenizer",
    "step": "text",
    "title": "Gemma tokenizer",
    "bounds": [
      348,
      967,
      197,
      83
    ]
  },
  {
    "id": "text-output",
    "step": "text",
    "title": "Text tensor",
    "bounds": [
      572,
      948,
      104,
      27
    ]
  },
  {
    "id": "state-history",
    "step": "proprioception",
    "title": "State history",
    "bounds": [
      816,
      844,
      265,
      101
    ]
  },
  {
    "id": "state-output",
    "step": "proprioception",
    "title": "Proprioception tensor",
    "bounds": [
      1027,
      887,
      111,
      29
    ]
  },
  {
    "id": "context-output",
    "step": "together",
    "title": "Context tensor",
    "bounds": [
      1073,
      1030,
      184,
      32
    ]
  },
  {
    "id": "vlm-context",
    "step": "vlm",
    "title": "Context tensor",
    "bounds": [
      1392,
      435,
      185,
      28
    ]
  },
  {
    "id": "action-chunk",
    "step": "flow",
    "title": "Action chunk",
    "bounds": [
      1779,
      356,
      215,
      34
    ]
  },
  {
    "id": "noise-time",
    "step": "flow",
    "title": "Noise & flow time",
    "bounds": [
      1788,
      391,
      220,
      52
    ]
  },
  {
    "id": "velocity",
    "step": "flow",
    "title": "Target velocity",
    "bounds": [
      1761,
      443,
      261,
      29
    ]
  },
  {
    "id": "context-qkv",
    "step": "transformer",
    "title": "Context queries, keys & values",
    "bounds": [
      1338,
      543,
      323,
      64
    ]
  },
  {
    "id": "action-qkv",
    "step": "transformer",
    "title": "Action queries, keys & values",
    "bounds": [
      1696,
      543,
      333,
      64
    ]
  },
  {
    "id": "concatenate",
    "step": "transformer",
    "title": "Sequence concatenation",
    "bounds": [
      1392,
      608,
      632,
      38
    ]
  },
  {
    "id": "attention-mask",
    "step": "transformer",
    "title": "Attention mask",
    "bounds": [
      1437.269,
      686.308,
      290.935,
      161.631
    ]
  },
  {
    "id": "attention-output",
    "step": "transformer",
    "title": "Attention output",
    "bounds": [
      1581,
      851,
      191,
      53
    ]
  },
  {
    "id": "context-residual",
    "step": "transformer",
    "title": "Context residual & feed-forward",
    "bounds": [
      1460,
      918,
      237,
      192
    ]
  },
  {
    "id": "action-residual",
    "step": "transformer",
    "title": "Action residual & feed-forward",
    "bounds": [
      1713,
      918,
      185,
      192
    ]
  },
  {
    "id": "eam-output",
    "step": "eam-projection",
    "title": "[B, 50, D]",
    "bounds": [
      1826.177,
      1192.018,
      60.783,
      20.052
    ]
  },
  {
    "id": "eam-linear",
    "step": "eam-projection",
    "title": "Linear(D, d)",
    "bounds": [
      1849.049,
      1211.443,
      80.835,
      20.365
    ]
  },
  {
    "id": "eam-prediction",
    "step": "eam-projection",
    "title": "[B, 50, d]",
    "bounds": [
      1825.863,
      1232.436,
      62.036,
      21.305
    ]
  },
  {
    "id": "eam-target",
    "step": "flow-loss",
    "title": "Target Velocity: V* = ε − A*",
    "bounds": [
      1779.18,
      1255.308,
      154.464,
      15.352
    ]
  },
  {
    "id": "eam-squared-error",
    "step": "flow-loss",
    "title": "Squared error",
    "bounds": [
      1866.908,
      1274.42,
      166.057,
      38.851
    ]
  },
  {
    "id": "eam-mean",
    "step": "flow-loss",
    "title": "Average across B × 50 × d",
    "bounds": [
      1716.203,
      1315.151,
      324.281,
      19.739
    ]
  },
  {
    "id": "loss-context",
    "step": "fast-input",
    "title": "X context",
    "bounds": [
      1416.978,
      1192.306,
      117.448,
      36.285
    ]
  },
  {
    "id": "fast-tokenizer",
    "step": "fast-input",
    "title": "FAST tokenizer",
    "bounds": [
      1488.593,
      1229.546,
      142.593,
      26.1
    ]
  },
  {
    "id": "fast-embedding",
    "step": "fast-input",
    "title": "Gemma embedding lookup",
    "bounds": [
      1493.367,
      1255.964,
      127.633,
      26.1
    ]
  },
  {
    "id": "fast-concatenate",
    "step": "fast-input",
    "title": "Concatenate with X context",
    "bounds": [
      1493.049,
      1284.292,
      116.812,
      23.553
    ]
  },
  {
    "id": "fast-mask",
    "step": "fast-transformer",
    "title": "Causal attention mask",
    "bounds": [
      1374.964,
      1306.89,
      89.757,
      59.52
    ]
  },
  {
    "id": "fast-positions",
    "step": "fast-transformer",
    "title": "F target FAST ID positions",
    "bounds": [
      1497.823,
      1337.127,
      171.875,
      15.596
    ]
  },
  {
    "id": "fast-meaning",
    "step": "fast-transformer",
    "title": "The size of the tensor has not changed. The meaning has.",
    "bounds": [
      1355.23,
      1403.65,
      115.539,
      71.615
    ]
  },
  {
    "id": "vocabulary-projection",
    "step": "token-probabilities",
    "title": "Linear(2560, 262208)",
    "bounds": [
      1508.008,
      1352.405,
      116.175,
      14.96
    ]
  },
  {
    "id": "fast-logits",
    "step": "token-probabilities",
    "title": "[B, F, 262208] · logits",
    "bounds": [
      1501.961,
      1367.683,
      78.299,
      10.504
    ]
  },
  {
    "id": "fast-log-softmax",
    "step": "token-probabilities",
    "title": "log-softmax",
    "bounds": [
      1510.555,
      1381.369,
      71.933,
      12.732
    ]
  },
  {
    "id": "fast-probabilities",
    "step": "token-probabilities",
    "title": "[B, F, 262208] · probabilities",
    "bounds": [
      1501.642,
      1395.056,
      96.441,
      11.458
    ]
  },
  {
    "id": "target-probability",
    "step": "cross-entropy",
    "title": "Correct target ID",
    "bounds": [
      1506.735,
      1416.381,
      122.541,
      50.608
    ]
  },
  {
    "id": "negative-log-likelihood",
    "step": "cross-entropy",
    "title": "−log p(zⱼ | Xctx, z<ⱼ)",
    "bounds": [
      1508.645,
      1469.535,
      98.669,
      14.323
    ]
  },
  {
    "id": "zero-loss",
    "step": "cross-entropy",
    "title": "p(zⱼ) = 1 → loss = 0",
    "bounds": [
      1508.963,
      1484.495,
      112.992,
      21.644
    ]
  },
  {
    "id": "token-confidence",
    "step": "cross-entropy",
    "title": "0.80 · 0.60 · 0.75 · 0.65",
    "bounds": [
      1345.363,
      1476.538,
      143.23,
      86.574
    ]
  },
  {
    "id": "mean-cross-entropy",
    "step": "cross-entropy",
    "title": "Mean loss",
    "bounds": [
      1494.004,
      1522.053,
      182.697,
      20.689
    ]
  },
  {
    "id": "transformer-fast-input",
    "step": "transformer",
    "title": "FAST tokens / X context",
    "bounds": [
      1337.928,
      491.072,
      202.411,
      56.695
    ]
  },
  {
    "id": "rope",
    "step": "transformer",
    "title": "RoPE",
    "bounds": [
      1548.793,
      540.804,
      112.395,
      19.893
    ]
  },
  {
    "id": "context-attention",
    "step": "transformer",
    "title": "Context attention",
    "bounds": [
      1526.414,
      700.446,
      79.572,
      56.198
    ]
  },
  {
    "id": "fast-attention",
    "step": "transformer",
    "title": "FAST causal attention",
    "bounds": [
      1526.414,
      756.643,
      143.727,
      48.241
    ]
  },
  {
    "id": "action-attention",
    "step": "transformer",
    "title": "Action attention",
    "bounds": [
      1526.414,
      804.884,
      203.406,
      38.294
    ]
  },
  {
    "id": "adaptive-rmsnorm",
    "step": "transformer",
    "title": "AdaptiveRMSNorm",
    "bounds": [
      1607.975,
      928.22,
      258.609,
      148.203
    ]
  },
  {
    "id": "ff-activation",
    "step": "transformer",
    "title": "FF Activation",
    "bounds": [
      1349.863,
      1002.819,
      89.518,
      95.984
    ]
  },
  {
    "id": "transformer-vlm-output",
    "step": "transformer",
    "title": "VLM output",
    "bounds": [
      1623.889,
      1083.883,
      107.422,
      21.385
    ]
  },
  {
    "id": "transformer-eam-output",
    "step": "transformer",
    "title": "EAM output",
    "bounds": [
      1732.804,
      1083.883,
      54.706,
      21.385
    ]
  }
];

// Only verbatim author copy. Empty entries are ready for later writing.
export const annotations = {
  "vlm": [
    "But it hasn't necessarily learned the detailed relationship between those observations and a robot's 50-step motor trajectory."
  ],
  "flow": [
    "You attach an initially untrained 860M action expert."
  ],
  "flow-loss": [
    "Now, suppose you only optimize:",
    "\\[\\mathcal L_{\\rm flow}.\\]",
    "You have two options:",
    "Option A: Allow flow gradients into Gemma.",
    "The randomly initialized action expert initially produces poor velocity predictions. Backpropagating those errors into Gemma can disrupt its useful pretrained representations.",
    "Option B: Freeze Gemma against flow gradients.",
    "Now its existing knowledge is protected, but there's no action-supervision signal directly teaching Gemma how observations and instructions relate to robot trajectories."
  ],
  "vlm-loss": [
    "The answer is Knowledge Insulation: π0.7 wants the VLM to understand robot actions without letting the continuous flow-matching loss damage its pretrained vision-language knowledge."
  ],
  "fast-input": [
    "FAST solves this problem.",
    "The hope is that at the beginning, before any transformer layers, you literally have the FAST tokens in the input tensor."
  ],
  "token-probabilities": [
    "After 34 transformer layers, at those same positions, you now have 262208-dimensional vectors carrying the probabilities for all next possible token IDs."
  ],
  "context": [],
  "observation": [],
  "subgoals": [],
  "text": [],
  "proprioception": [],
  "together": [],
  "decision": [],
  "l1": [],
  "l2": [],
  "l33": [],
  "l34": [],
  "transformer": [],
  "eam-loss": [],
  "eam-projection": [],
  "fast-transformer": [],
  "cross-entropy": [],
  "observation-input": [],
  "patches": [],
  "embeddings": [],
  "vision-backbone": [],
  "pooling": [],
  "observation-output": [],
  "goal-input": [],
  "goal-patches": [],
  "goal-backbone": [],
  "goal-embeddings": [],
  "goal-output": [],
  "prompt-parts": [],
  "tokenizer": [],
  "text-output": [],
  "state-history": [],
  "state-output": [],
  "context-output": [],
  "vlm-context": [],
  "action-chunk": [],
  "noise-time": [],
  "velocity": [],
  "context-qkv": [],
  "action-qkv": [],
  "concatenate": [],
  "attention-mask": [],
  "attention-output": [],
  "context-residual": [],
  "action-residual": [],
  "eam-output": [],
  "eam-linear": [],
  "eam-prediction": [],
  "eam-target": [],
  "eam-squared-error": [],
  "eam-mean": [],
  "loss-context": [],
  "fast-tokenizer": [],
  "fast-embedding": [],
  "fast-concatenate": [],
  "fast-mask": [],
  "fast-positions": [],
  "fast-meaning": [],
  "vocabulary-projection": [],
  "fast-logits": [],
  "fast-log-softmax": [],
  "fast-probabilities": [],
  "target-probability": [],
  "negative-log-likelihood": [],
  "zero-loss": [],
  "token-confidence": [],
  "mean-cross-entropy": [],
  "transformer-fast-input": [],
  "rope": [],
  "context-attention": [],
  "fast-attention": [],
  "action-attention": [],
  "adaptive-rmsnorm": [],
  "ff-activation": [],
  "transformer-vlm-output": [],
  "transformer-eam-output": []
};
