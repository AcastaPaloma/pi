// Navigation and geometry only. Author commentary is intentionally empty.
// Bounds use a 2048-wide logical canvas, mapped from the supplied SVG viewBox.
export const artwork = {
  "src": "./assets/full-backbone.svg",
  "width": 2048,
  "height": 1122.5187033283646,
  "nativeWidth": 4118.032888760241,
  "nativeHeight": 2257.1137395286646
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
      "context-qkv",
      "action-qkv",
      "concatenate",
      "attention-mask",
      "attention-output",
      "context-residual",
      "action-residual"
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
  }
];

// Populate later with the author's supplied text: { 'step-or-detail-id': ['Paragraph'] }.
export const annotations = {};
