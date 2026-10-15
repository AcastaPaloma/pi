// Article prose, keyed by canvas destination. Each entry is an ordered list of blocks:
//   'text'                 paragraph (**bold** is supported)
//   { list: [...] }        bulleted list of paragraphs
//   { note: 'text' }       a note where a drawn label differs from the paper
//   { placeholder: 'text'} a reserved slot the author will write
// The author's original passages (author-copy.json) appear verbatim and in order.
// Saved editor text in text.json overrides individual sections.
export const annotations = {
  "preface": [
    { placeholder: "Preface — reserved for the author. A short personal note on why this write-up exists and how it came together will go here." }
  ],
  "map": [
    "π0.7 (Physical Intelligence, April 2026, arXiv 2604.15483) is a robot control model with about 5 billion parameters. It reads camera images, the robot's joint positions, and a text prompt, and outputs the robot's next 50 actions. Models of this kind are called vision-language-action models, or VLAs.",
    "The paper's main idea is the prompt. Earlier models in the family (π0, π0.5, π0.6) were prompted with a short task description. π0.7 is also trained with a subtask instruction, episode metadata (how fast and how well the recorded robot performed, and whether it made a mistake), the control mode, and, in some examples, images of what the scene should look like a few seconds later. These labels let one model learn from mixed-quality data, including failed episodes, autonomous rollouts, and human video, without averaging good and bad behavior together. At runtime, you steer the model by choosing what to put in the prompt.",
    "Three models work together at runtime:",
    { list: [
      "**π0.7, the low-level policy.** Images, joint positions, and the prompt go in; a 50-step action chunk comes out. Covered in the Ingestion, Decision model, and Training losses chapters.",
      "**The high-level policy.** The same architecture, fine-tuned to write the next subtask as text, for example “turn on the faucet”. Covered in the High-level policy chapter.",
      "**The world model.** A 14-billion-parameter image generator, initialized from ByteDance's BAGEL, that draws the subgoal images for the current subtask. Covered in the World model chapter; its diagram is still being drawn."
    ] },
    "The boxed sketch at the top of the canvas shows only the low-level policy: ingest the context, encode it with the Gemma 3 backbone (the “Encoder (VLM)”), and pass it to the action expert. The dashed “Decoder (VLM)” box is the backbone's text output. π0.7 uses it only during training, to predict discrete action tokens; the high-level policy uses the same output to write subtasks.",
    "You do not have to read this in order. Each main section opens with what goes in, what comes out, and why. Its component notes are folded underneath; open them when you want the tensor-level detail. Clicking any box in the drawing opens its note. The routes below are shorter paths through the article.",
    "The π0.7 paper does not print every number in the drawings. Layer sizes, patch sizes, the memory encoder's attention schedule, and the action tokenizer come from the work π0.7 builds on: Gemma 3, MEM, knowledge insulation, FAST, and π0. The text says when a detail is not stated in the π0.7 paper."
  ],
  "notation": [
    "Tensor shapes are written in square brackets, one entry per axis. [B, L, 2560] means B examples, each with L tokens, each token a list of 2560 numbers. The same letters appear on every part of the canvas:",
    { list: [
      "**B**: batch size, the number of training examples processed together.",
      "**V**: number of cameras, at most 4 (front, left wrist, right wrist, and an optional rear camera on mobile robots).",
      "**6**: frames per camera: the current frame and 5 past frames, 1 second apart.",
      "**448**: image height and width in pixels after resizing.",
      "**1024**: patches per image: a 32 × 32 grid of 14 × 14-pixel patches.",
      "**1152**: width of a token inside the SigLIP vision encoder.",
      "**2560**: width of a token inside the Gemma 3 backbone. Every input must reach this width before it enters the backbone.",
      "**256**: tokens per image after pooling. Inside attention, 256 is also the width of one attention head; the two meanings are unrelated.",
      "**L**: number of text tokens in the prompt.",
      "**S**: length of the full context, S = L + 256V + 774, where 774 = 768 subgoal-image tokens + 6 state tokens.",
      "**F**: number of FAST action tokens for one action chunk. It varies from example to example.",
      "**T**: sequence length inside the backbone during training, T = S + F − 1. (In the text-prompt box, T instead names the raw text.)",
      "**50**: actions per chunk. At 50 control steps per second, this is one second of motion.",
      "**d**: numbers per action, set by the robot (joint targets and gripper commands).",
      "**D, D_E**: width of the action expert and of its feed-forward layer. The paper gives only the expert's total size, 860 million parameters.",
      "**262208**: number of entries in Gemma 3's vocabulary.",
      "**(1)**: an axis of size 1, kept in the shape to show that an axis was reduced to a single entry, for example one frame after the history is compressed.",
      "**A*, ε, τ**: the ground-truth action chunk, Gaussian noise, and the flow-matching time between 0 and 1."
    ] }
  ],

  "context": [
    "This is the entire ingestion engine (fancy fancy words). During training, it provides all the context needed to predict the next 50 actions the robot will take:",
    { list: [
      "Full images from 3 (or 4) cameras, each with 6 frames of history;",
      "Text, which includes the task, the current subtask, episode metadata, and the control mode;",
      "Proprioception, which tells the robot where IT is: its joint positions over the same 6 time steps;",
      "Visual subgoals: up to three images of what the scene should look like a few seconds from now. During training these are real future frames or world-model images; at runtime the world model generates them."
    ] },
    "Each input has a different form: pixels, words, and joint angles. Ingestion turns every one of them into a sequence of vectors 2560 numbers wide, the width of the Gemma 3 backbone. Concatenated, they form one tensor, X_ctx, shown in the bottom-right corner of this panel. Everything downstream reads only X_ctx.",
    "During training, parts of the prompt are randomly dropped (the control mode never is), so the model also works with any subset of them at runtime. Subgoal images are present in only 25% of training examples."
  ],
  "observation": [
    "This follows a standard VLM image encoder, extended to video. Each camera contributes 6 frames: the current frame and 5 past frames, 1 second apart. Every frame is cut into patches, and each patch is given a positional embedding that records where it was in the image.",
    "The patches then pass through 27 transformer layers. Every layer applies spatial attention, in which each patch reads the other patches of the same frame. Every fourth layer also applies temporal attention, in which each patch reads the same patch position in earlier frames. The shape is still [B, V, 6, 1024, 1152], but the current frame's patches now also carry information from the past frames. Cameras do not exchange information here; that happens later, inside Gemma.",
    "After the last temporal layer, the 5 past frames have nothing left to contribute, so they are dropped: [B, V, (1), 1024, 1152]. Pooling and a linear projection then reduce each camera to 256 tokens of width 2560. The final tensor is [B, 256V, 2560].",
    "This history encoder comes from MEM (Torne et al., 2026). It adds no new learned weights to the image encoder, so it can start from the pretrained encoder that ships with Gemma 3."
  ],
  "observation-input": [
    "Hopefully, straightforward enough: it's a little video. O = [B, V, 6, 3, 448, 448] holds, for each of B examples, V cameras × 6 frames, each frame 448 × 448 pixels in 3 color channels (red, green, blue).",
    "The frames are sampled 1 second apart, so the model sees the last 5 seconds. During training the whole history is dropped 30% of the time, and the rear camera, when present, is dropped 30% of the time."
  ],
  "patches": [
    "With a patch size of 14 × 14 pixels, each 448 × 448 image becomes a 32 × 32 grid: 1024 patches. Each patch spans all 3 color channels, so it is effectively a 14 × 14 × 3 prism.",
    "We flatten the 14 × 14 × 3 = 588 values, project them to 1152 numbers through Linear(588, 1152), and now call what used to be patches “tokens”. The same projection runs on every patch of every frame, camera, and example: 1024 × 6 × V × B patches in total, giving [B, V, 6, 1024, 1152]."
  ],
  "embeddings": [
    "This step looks circular. Patchifying flattens the image into a list of 1024 tokens, and attention treats that list as an unordered set: on its own, it cannot tell the top-left patch from the bottom-right one. For a video, it's trivial to recognize where each patch belongs in both **space and time**, but once flattened, we need to write that information back into the tensor.",
    "**Space.** SigLIP adds a learned 1152-number vector to each token, one vector per grid position (1024 in total). The same position receives the same vector in every frame and every camera.",
    "**Time.** MEM adds a fixed (not learned) sinusoidal vector e(t) that depends on how many seconds old the frame is, with e(0) = 0 for the current frame. Because e(0) = 0, a single current frame is encoded exactly as the original SigLIP would encode it.",
    { note: "the drawing places both embeddings at the input. In MEM, the time embedding is added to the tokens inside each temporal-attention layer, right before that layer's attention, rather than once at the input." }
  ],
  "vision-backbone": [
    "The encoder is SigLIP So400m, the 400-million-parameter vision encoder that ships with Gemma 3. Its 27 layers each apply spatial attention: every token attends to every other token of the same frame. Each layer has 16 attention heads, each with its own keys and values, followed by a 2-layer MLP; each sub-block adds its output back to its input (the two residuals drawn in the first box).",
    "To run spatial attention, the frames are moved into the batch axis: [BV·6, 1024, 1152]. Each of the BV·6 frames is processed independently.",
    "Every fourth layer (layers 4, 8, …, 24: six layers in total) also applies temporal attention. A token at patch position p in a frame attends to the tokens at position p in that frame and in earlier frames, never later ones. This is how information from the past reaches the current frame. Attending only within a frame or only along one patch position keeps the cost far below attention across all 6 × 1024 tokens at once.",
    "It is okay to view the inside of every transformer block in this paper as a black box. What matters is which tokens are allowed to read which."
  ],
  "pooling": [
    "This column reduces the encoder output to what Gemma needs.",
    { list: [
      "**Discard the past frames.** [B, V, 6, 1024, 1152] → [B, V, (1), 1024, 1152]. The drawing asks whether the first 5 frames are discarded: yes. After the last temporal layer, spatial attention never crosses frames, so the past frames can no longer change the current one. MEM drops their tokens and passes only the current frame onward, so the backbone receives the same number of tokens as a model without memory.",
      "**Average pooling with a 2 × 2 kernel.** Each 2 × 2 block of neighboring patches is averaged into one token: 1024 → 256 tokens. The tokens are still a flat list, so the grid is recovered with integer division and remainder (“mod”): token i sits in row i ÷ 32 and column i mod 32.",
      "**Linear(1152, 2560).** Gemma 3's image projector widens each token to the backbone width: [B, V, (1), 256, 2560]."
    ] }
  ],
  "observation-output": [
    "Reshaping lines up all cameras one after another: [B, 256V, 2560]. With 3 cameras, that is 768 tokens per example. These tokens summarize the last 5 seconds of video, but there are only as many of them as one frame per camera would produce."
  ],
  "subgoals": [
    "A visual subgoal is a picture of what each camera should see a few seconds from now: the outcome the robot should produce next. The drawing calls it “1 single wishful, fictitious, GOAL frame”. A subtask such as “open the fridge door” says what to do; a subgoal image also shows how, for example where the gripper should be on the handle.",
    "There are up to 3 subgoal images: one each for the front camera and the two wrist cameras. The rear camera has no subgoal. The images pass through the same SigLIP encoder, with the same weights, as the observation frames. Each subgoal is a single frame, so the temporal layers have nothing to attend to, and the result is [B, 768, 2560].",
    "At runtime, the subgoals are generated by the world model (see the World model chapter), which is initialized from BAGEL and trained with flow matching. Its flow matching is separate from the action expert's: one generates images, the other generates actions. During training, π0.7 sees a mix of real future frames and generated images. Real frames are taken from the end of the current subtask segment with probability 0.25, and otherwise from a random time 0 to 4 seconds ahead.",
    "Subgoal images make training much easier, because predicting the actions between the current image and a future image is close to an inverse-dynamics problem. To keep the model from relying on them, they appear in only 25% of training examples. When they are present, the text subtask is dropped 30% of the time."
  ],
  "goal-input": [
    "g_t = [G¹_t, G²_t, G³_t] holds the three goal images for time t, one per camera. Each is [B, 3, (1), 3, 448, 448]: B examples, 3 cameras, a single frame, 3 color channels, 448 × 448 pixels. The first 3 counts cameras and the second counts color channels."
  ],
  "goal-patches": [
    "Patchification is identical to the observation's: 1024 patches of 14 × 14 × 3 = 588 values per image, each projected to 1152 numbers. There is one frame per camera, so 1024 × 1 × 3 × B patches in total: [B, 3, (1), 1024, 1152]."
  ],
  "goal-backbone": [
    "The same SigLIP/MEM encoder: 27 layers with the observation's weights. With one frame there is no earlier frame to attend to, and the time embedding of the current frame is zero, so the encoder computes exactly what plain SigLIP computes."
  ],
  "goal-embeddings": [
    "Only positional embeddings are added: the same learned vector per grid position as for the observation. A single frame needs no time embedding."
  ],
  "goal-output": [
    "As for the observation: 2 × 2 average pooling gives 256 tokens per image, Linear(1152, 2560) widens them, and reshaping concatenates the 3 cameras: [B, 768, 2560]."
  ],
  "text": [
    "Everything the model is told in words is one string of raw text, with three parts: ℓ, the overall task (“clean the kitchen”); ℓ̂, the current subtask (“clean cup”); and m, the episode metadata. The control mode is appended at the end.",
    "The Gemma tokenizer splits the string into L integer IDs, each between 0 and 262207, and an embedding table turns each ID into 2560 numbers: [B, L, 2560].",
    "In the paper's example prompt, the image tokens come first, then this text, then the joint-position tokens: “<Multi-view observation><Multi-view subgoals> Task: peel vegetables. Subtask: pick up the peeler. Speed: 8000. Quality: 5. Mistake: false. Control Mode: joint.<Proprioception>”."
  ],
  "prompt-parts": [
    "Each line of the prompt has a specific job:",
    { list: [
      "**Task** is the overall instruction, given by a person.",
      "**Subtask** is the next step. Human annotators label it in the training data; at runtime it comes from the high-level policy or from a person coaching the robot step by step.",
      "**Speed** is the length of the episode in control steps, rounded to the nearest 500. Shorter episodes usually mean fewer mistakes.",
      "**Quality** is a human score from 1 to 5.",
      "**Mistake** marks whether the robot made an error, such as a failed grasp, within the current segment.",
      "**Control Mode** says whether actions are joint targets (joint) or end-effector poses (ee). It is never dropped during training."
    ] },
    "At runtime the metadata asks for the best behavior seen in the data: quality 5, no mistake, and the speed set to the 15th percentile of that task's episode lengths.",
    { note: "the example prompt writes Speed: \"normal\". In the paper, speed is a number of steps, such as Speed: 8000." }
  ],
  "tokenizer": [
    "The Gemma tokenizer maps the string to a list of integer IDs, z₁, z₂, z₃, … Each ID indexes one of Gemma 3's 262208 vocabulary entries, so z ∈ [0, 262207]. The result is [B, L, (1)]: one integer per token. Prompts of different lengths are padded to a common L within a batch."
  ],
  "text-output": [
    "Each ID selects one row of Gemma's embedding table, a 262208 × 2560 matrix. Selecting a row equals multiplying a one-hot vector by the matrix, which is why the drawing labels the step “Linear”. The result is [B, L, 2560]."
  ],
  "proprioception": [
    "Proprioception is the robot's sense of its own body: joint angles and gripper openings. The model receives the state at the same 6 time steps as the camera frames, Q = [Q_{t−5}, …, Q_{t−1}, Q_t], shaped [B, 6, d_q], where d_q is the number of state values for that robot.",
    "One linear layer maps each state to 2560 numbers, giving one token per time step: [B, 6, 2560]. π0.6 wrote the state out as text tokens instead; six states in text would take many more tokens, so π0.7 follows MEM and uses the projection. When the history frames are dropped during training, the matching state tokens are masked out too."
  ],
  "state-history": [
    "Q_t is the state now, and Q_{t−5} … Q_{t−1} are the states 1 to 5 seconds earlier, aligned with the past camera frames. Each Q holds d_q numbers describing the actuators: for example, two arms with 6 joints and one gripper each give 14."
  ],
  "state-output": [
    "Linear(d_q, 2560) is applied to each of the 6 states separately: [B, 6, d_q] → [B, 6, 2560]. The current state is the last state token."
  ],
  "together": [
    "The four inputs are concatenated along the sequence axis into one tensor:",
    { list: [
      "observation: [B, 256V, 2560]",
      "subgoal images: [B, 768, 2560]",
      "text: [B, L, 2560]",
      "proprioception: [B, 6, 2560]"
    ] },
    "X_ctx = [B, L + 256V + 774, 2560], where 774 = 768 + 6. This length is called S in the rest of the canvas. At this stage the tensor is mostly image patches, with the other modalities appended; after the backbone's attention layers, every image token also carries information from the text and the state.",
    "When a part of the prompt is dropped, its tokens are left out and S shrinks. In the 75% of training examples without subgoal images, the 768 subgoal tokens are absent."
  ],
  "context-output": [
    "X_ctx : [B, L + 256V + 774, 2560] is the only input the decision model reads. With 3 cameras and every prompt part present, that is L + 1542 tokens per example."
  ],

  "decision": [
    "The decision model turns X_ctx into 50 actions. It is two transformers placed side by side:",
    { list: [
      "**The VLM backbone** (left, yellow): Gemma 3 4B, a pretrained vision-language model with 34 layers of width 2560. It reads X_ctx.",
      "**The action expert** (right, blue): a smaller transformer with 860 million parameters. It reads 50 noisy action tokens and learns, by flow matching, to turn noise into a clean action chunk."
    ] },
    "The two have separate weights but share one attention operation in every layer: in layer n, the action tokens read the keys and values that the backbone computed in its layer n. This is how the actions see the images and the prompt. The lower panel, “Transformer Layer L(n)”, draws one such layer.",
    "The canvas title reads “One Forward Pass (two, really)”. In training, one pass computes both models at once. At runtime the backbone runs once on the context and stores its keys and values; the expert then runs 5 times, once per denoising step, reading the stored context each time. The full model answers in 38 to 127 milliseconds on one H100 GPU, depending on whether memory and subgoal images are used."
  ],
  "vlm": [
    "Gemma 3 was pretrained on images and text from the web. It already knows what a mug is, where it sits in the image, and what “the upper cabinet” refers to.",
    "But it hasn't necessarily learned the detailed relationship between those observations and a robot's 50-step motor trajectory.",
    "The panel shows the backbone's input: B sequences, each of length L + 256V + 774 and width 2560. This whole context is X_ctx, whose length is written S. The arrow sends it into the first of the 34 layers."
  ],
  "vlm-context": [
    "X_ctx : [B, S, 2560], with S = L + 256V + 774. The red bracket marks the sequence length S, which every later shape uses."
  ],
  "flow": [
    "You attach an initially untrained 860M action expert.",
    "The expert produces a whole action chunk at once: the next 50 actions. It is trained with flow matching. A training example takes the true chunk A*, mixes it with Gaussian noise ε by an amount τ, and asks the expert which direction leads from the mixture back toward the data. At runtime, the expert starts from pure noise and follows its own predicted directions in 5 steps to a clean chunk.",
    "Flow matching is used instead of predicting one fixed answer because many different motions can complete a task correctly. A model trained to predict the average of all those motions can produce a trajectory that matches none of them. Generating a sample, as image generators do, avoids that.",
    "π0.7 also trains for real-time chunking. The robot keeps moving while the next chunk is computed, so the first few actions of a new chunk are already committed. During training, the first 0 to 12 actions of a chunk (up to 240 ms at 50 Hz) are given to the expert without noise and excluded from the loss. At runtime the already-committed actions fill those slots, and the new chunk continues smoothly from them."
  ],
  "action-chunk": [
    "A* : [B, 50, d] is the ground-truth chunk: 50 consecutive actions from the recorded episode, each a list of d numbers. On the paper's 50 Hz robots, 50 actions cover 1 second; the UR5e arms run at 20 Hz, so on them a chunk covers 2.5 seconds.",
    "A linear layer widens each action to the expert's width: [B, 50, d] → [B, 50, D], one token per action.",
    { note: "the drawing suggests d = 12 for two 6-joint arms. The paper's bimanual robots also have a 1-DoF gripper on each arm, which gives 14, and the mobile robots add lift and base axes. π0 and π0.5 pad every robot's actions to one shared width (32 in their open-source release) so a single model serves all robots." }
  ],
  "noise-time": [
    "The noisy input is A_τ = (1 − τ)·A* + τ·ε, with shape [B, 50, d].",
    { list: [
      "τ is the flow time, one number per example: [B, (1)]. At τ = 0 the input is the clean chunk; at τ = 1 it is pure noise.",
      "ε is Gaussian noise with mean 0 and variance 1, the same shape as the chunk: [B, 50, d]."
    ] },
    { note: "ε is labeled [B, 1024] here and in the transformer panel. The noise must have the chunk's shape, [B, 50, d], because it is added to A* entry by entry." }
  ],
  "velocity": [
    "The target velocity is V* = ε − A*. It is the derivative of A_τ with respect to τ: the direction in which the noisy chunk moves as τ grows from 0 to 1.",
    "The expert learns to predict V* from A_τ, τ, and the context. At runtime, sampling runs this path backwards: start at τ = 1 with pure noise and take 5 steps of size 0.2, each A ← A − 0.2 · (predicted velocity), until τ = 0."
  ],
  "l1": [
    "The backbone is a stack of 34 identical layers, drawn as L1 through L34. L1 receives X_ctx directly; during training, the embedded FAST action tokens are appended to it (see the VLM loss).",
    "The action expert has its own stack. In π0-style models, each expert layer is paired with the backbone layer at the same depth, because its attention reads that layer's keys and values. The arrow from the flow-matching panel into L1 shows the 50 action tokens entering the first paired layer."
  ],
  "l2": [
    "L2 has exactly the same structure as L1; only its weights differ. The dots between L2 and L33 stand for layers 3 to 32, which repeat the same structure. One layer is drawn in full in the panel below, “Transformer Layer L(n)”."
  ],
  "l33": [
    "Every layer keeps the shapes unchanged: the backbone's tokens stay [B, T, 2560] and the expert's tokens stay [B, 50, D]. What changes from layer to layer is what each vector encodes."
  ],
  "l34": [
    "L34 is the last layer. Its outputs, the yellow “Output” box, split two ways. The backbone's vectors at the FAST-token positions go to the vocabulary projection and the VLM loss. The expert's 50 vectors go to Linear(D, d), which produces the predicted velocity for the EAM loss."
  ],
  "transformer": [
    "This panel opens one of the 34 layers. The left half (yellow) uses the backbone's weights; the right half (blue) uses the expert's. They meet only in the middle, in attention.",
    { list: [
      "**Project.** Each side normalizes its tokens and projects them to queries (Q), keys (K), and values (V).",
      "**Concatenate.** The backbone's T tokens and the expert's 50 tokens are joined into one sequence of length T + 50.",
      "**Attend, with a mask.** One attention operation runs over the joined sequence. A mask decides which tokens may read which.",
      "**Split.** The output is cut back into the backbone's T rows and the expert's 50 rows. Each side applies its own output projection, residual connections, and feed-forward network."
    ] },
    "Because the mask decides who reads whom, the same weights serve three roles at once: the context reads the context, the FAST tokens predict actions as text, and the action tokens read the context.",
    "During training the backbone sequence is T = S + F − 1 tokens long: the S context tokens plus all but the last of the F FAST tokens (the VLM loss explains the − 1).",
    { note: "several labels in this panel write S + F where S + F − 1 = T is meant, for example [B, S + F, 2048] after attention. The output of attention is [B, 8, T + 50, 256], not [B, 8, T, 256], and the concatenated values are [B, 4, T + 50, 256]: values have 4 heads, like keys." }
  ],
  "transformer-fast-input": [
    "The backbone's input to this layer. In layer 1 it is X_ctx : [B, S, 2560] with the embedded FAST tokens appended: X' : [B, T, 2560], with T = S + F − 1. In later layers it is the previous layer's output, with the same shape."
  ],
  "context-qkv": [
    "RMSNorm rescales each token to a standard size. Three linear layers then produce queries [B, T, 2048], keys [B, T, 1024], and values [B, T, 1024]. These projections have no bias term.",
    "Reshaping splits the widths into heads of 256 numbers each: 8 query heads (Q_B : [B, 8, T, 256]) and 4 key and value heads (K_B, V_B : [B, 4, T, 256]). This is grouped-query attention: each key-value head is shared by 2 query heads, which halves the memory for keys and values. Gemma 3 also normalizes each query and key vector before RoPE (QK-norm); the drawing omits that step."
  ],
  "action-qkv": [
    "The expert does the same with its own weights. In layer 1, its input is A_τ projected to width D (in later layers, the previous layer's output), normalized by AdaptiveRMSNorm, which injects the flow time τ (see AdaptiveRMSNorm below). Its projections produce Q_A : [B, 8, 50, 256] and K_A, V_A : [B, 4, 50, 256].",
    "The expert must use exactly the backbone's head count and head width. Otherwise its queries could not be compared with the backbone's keys, and the two sequences could not be joined."
  ],
  "rope": [
    "RoPE, rotary position embedding, rotates each query and key vector by an angle proportional to the token's position in the sequence. After rotation, the score between two tokens depends on how far apart they are. This is how the backbone knows token order without adding position vectors to its inputs."
  ],
  "concatenate": [
    "The backbone's and the expert's projections are joined along the sequence axis:",
    { list: [
      "Q : [B, 8, T + 50, 256]",
      "K : [B, 4, T + 50, 256]",
      "V : [B, 4, T + 50, 256]"
    ] },
    "From here until the split, attention treats the T context and FAST tokens and the 50 action tokens as one sequence. The first T positions belong to the backbone (“first T elements, for VLM”) and the last 50 to the action expert (“last T elements, for AE”; read: last 50)."
  ],
  "attention-mask": [
    "For one head, attention compares every query with every key: Q @ Kᵀ gives a (T + 50) × (T + 50) table of scores, scaled by 1/√256. The mask adds −∞ to every pair that is not allowed. After the softmax, those pairs get weight exactly 0. The table in the drawing is that mask; rows are the readers (queries) and columns are what they read (keys).",
    { list: [
      "**Context tokens read only the context.** They must not depend on FAST or action tokens, which encode the answer.",
      "**FAST tokens read the context and earlier FAST tokens.** Each one is predicted from the tokens before it, like ordinary text.",
      "**Action tokens read the context and all 50 action tokens.** The chunk is generated as a whole, so its tokens attend to each other in both directions.",
      "**FAST and action tokens never read each other.** FAST tokens exist only in training, so the expert must not depend on them; and the noisy actions contain the ground truth, which the FAST predictions must not see."
    ] },
    "The drawing simplifies the context block. In the paper, observation tokens attend to each other in both directions; subgoal tokens attend to each other and to the observation; the text and state tokens are causal, reading only earlier tokens."
  ],
  "context-attention": [
    "The top-left block: context tokens c₁ … c_S score against each other (s₁₁ … s_SS). Every column to their right is −∞, so no context token reads a FAST or action token. The context is computed exactly as it would be at runtime, when no FAST or action tokens exist."
  ],
  "fast-attention": [
    "The middle rows, f₁ … f_F: each FAST token reads all context tokens and the FAST tokens up to and including itself. This lower-triangular pattern is causal, autoregressive training: the vector at FAST position j can only use what comes before token j + 1, so it can be trained to predict token j + 1."
  ],
  "action-attention": [
    "The bottom rows, a₁ … a₅₀: each action token reads every context token and every action token, and nothing in the FAST columns.",
    "The example on the right follows one head for action a₁. Its weights are α(a₁) = softmax(s_α, …, s_γ, −∞, …, −∞, s_δ, …, s_ζ): a probability for each allowed key, 0 for each masked one. Its output is the weighted sum of the allowed value vectors, o(a₁) = Σ α·v.",
    "Knowledge insulation adds one rule that a mask cannot express: when the expert reads the backbone's keys and values, gradients are stopped there. The flow loss can change how the expert reads the context, but not the context itself (see the VLM loss)."
  ],
  "attention-output": [
    "O = softmax(Q Kᵀ / √256 + mask) V. Each head returns one 256-number vector per token, so the result is [B, 8, T + 50, 256].",
    "The output is split back. The first T rows go to the backbone: its 8 heads are concatenated to [B, T, 2048], and Linear(2048, 2560) returns them to the backbone width. The last 50 rows go to the expert: [B, 50, 2048], then Linear(2048, D)."
  ],
  "context-residual": [
    "The backbone side follows Gemma 3's layer exactly:",
    { list: [
      "RMSNorm the attention output, then add it to the layer's input (Residual 1), giving X''.",
      "RMSNorm X'' and apply the feed-forward network: Linear(2560, 10240) twice (a gate and an up projection), GELU(gate) ⊙ up, then Linear(10240, 2560).",
      "RMSNorm the result and add it to X'' (Residual 2)."
    ] },
    "Normalizing both before and after each sub-block is Gemma 3's design. The output, [B, T, 2560], is the input to the next layer. The note “this comes aaaaall the way before attention scoring” points back to X', the layer input that the first residual adds back."
  ],
  "action-residual": [
    "The expert side has the same pattern with three differences: its width is D, its feed-forward width is D_E, and every normalization is AdaptiveRMSNorm, so the flow time τ conditions every sub-block.",
    { list: [
      "Linear(2048, D) on the attention output, AdaptiveRMSNorm, then Gated Residual 1.",
      "AdaptiveRMSNorm, Linear(D, D_E) as gate and up, GELU(gate) ⊙ up, Linear(D_E, D).",
      "Gated Residual 2."
    ] },
    "The output, [B, 50, D], goes to the next layer's expert half."
  ],
  "adaptive-rmsnorm": [
    "The expert must behave differently at different noise levels: near τ = 1 it sees almost pure noise; near τ = 0 it sees nearly finished actions. AdaptiveRMSNorm feeds τ into every layer.",
    { list: [
      "τ, one number per example, is expanded into a sinusoidal vector [B, (1), D] and passed through Linear(D, 3D).",
      "The result is split into three vectors: a scale γ, a shift β, and a gate g, each [B, (1), D].",
      "Each action token h is RMS-normalized, r = h / √(mean(h²) + ε), then modulated: h̃ = r ⊙ (1 + γ) + β.",
      "After a sub-block computes its output O_A from h̃, the gated residual adds it back: H' = g ⊙ O_A + H."
    ] },
    "γ, β, and g depend only on τ, so all 50 action tokens of an example share them (“same g!”). The ε in the RMS formula is a tiny constant that prevents division by zero. It is not the flow-matching noise ε."
  ],
  "ff-activation": [
    "The feed-forward network is a gated GELU unit (GeGLU), the same on both sides. For an input H':",
    { list: [
      "u = W_up · H'",
      "v = GELU(W_gate · H')",
      "F = W_down · (u ⊙ v)"
    ] },
    "⊙ is element-wise multiplication. GELU makes the layer nonlinear, and the product lets one projection decide how much of the other passes through. F has the same shape as H': [B, T, 2560] on the backbone side, [B, 50, D] on the expert side."
  ],
  "transformer-vlm-output": [
    "[B, S + F − 1, 2560]: the backbone tokens after this layer, the input to the next. After L34, the vectors at the FAST-token positions feed the VLM loss."
  ],
  "transformer-eam-output": [
    "[B, 50, D]: the action tokens after this layer. After L34, they feed the EAM loss."
  ],

  "eam-loss": [
    "Training has two losses, one per half of the model. EAM stands for expert action model, the drawing's name for the action expert. Its loss is a squared error on the predicted velocity.",
    "Only the expert learns from this loss. The stop-gradient described under knowledge insulation keeps it out of the backbone."
  ],
  "eam-projection": [
    "After L34, each of the 50 action tokens is D numbers wide. One linear layer maps it back to the action's own size, d, giving the predicted velocity Û : [B, 50, d]."
  ],
  "eam-output": [
    "[B, 50, D]: the expert's output after the last layer, one token per action."
  ],
  "eam-linear": [
    "Linear(D, d) reads each token separately. It is the inverse of the input projection, which widened each action from d to D."
  ],
  "eam-prediction": [
    "[B, 50, d]: Û, the predicted velocity. It has exactly the shape of the target V*, so the two can be compared entry by entry."
  ],
  "flow-loss": [
    "The flow loss squares the difference between the predicted velocity Û and the target V* = ε − A*, and averages it over all B × 50 × d entries.",
    "Now, suppose you only optimize:",
    "\\[\\mathcal L_{\\rm flow}.\\]",
    "You have two options:",
    "Option A: Allow flow gradients into Gemma.",
    "The randomly initialized action expert initially produces poor velocity predictions. Backpropagating those errors into Gemma can disrupt its useful pretrained representations.",
    "Option B: Freeze Gemma against flow gradients.",
    "Now its existing knowledge is protected, but there's no action-supervision signal directly teaching Gemma how observations and instructions relate to robot trajectories.",
    "π0.7 needs both properties: Gemma protected from the flow loss, and Gemma still trained on actions. The VLM loss provides both."
  ],
  "eam-target": [
    "Target velocity: V* = ε − A*, [B, 50, d]. The same ε that produced the noisy input A_τ is reused here, so each example's target is known exactly."
  ],
  "eam-squared-error": [
    "For every example b, action t, and dimension j, the error is (Û_{b,t,j} − (ε_{b,t,j} − A*_{b,t,j}))². Squaring makes all errors positive and penalizes large errors more than small ones."
  ],
  "eam-mean": [
    "L_flow = 1 / (B · 50 · d) × the sum of all squared errors: “essentially an average across all B × 50 × d entries”. Actions excluded by real-time-chunking training, the first 0 to 12 of each chunk, are left out of the average."
  ],
  "vlm-loss": [
    "The answer is Knowledge Insulation: π0.7 wants the VLM to understand robot actions without letting the continuous flow-matching loss damage its pretrained vision-language knowledge.",
    "Knowledge insulation (Driess et al., 2025) combines the two options:",
    { list: [
      "**From Option B: stop the flow gradients.** The expert reads the backbone's keys and values, but the flow loss never changes the backbone.",
      "**Replace what Option B lost: give the backbone its own action loss.** The same 50-step chunk is written as a short sequence of discrete tokens, FAST tokens, and the backbone learns to predict them with next-token cross-entropy, the loss Gemma was pretrained with. A discrete, well-behaved loss of the familiar kind is far less likely to damage what Gemma already knows."
    ] },
    "The backbone also keeps training on non-robot data, such as visual question answering, object localization, and video captioning, which preserves its vision-language skills.",
    "“It's supervised!”: every target token is known from the recorded actions. This panel follows one training example from X_ctx to a single number. FAST tokens exist only during training. At runtime the model never generates them; the action expert produces the actions."
  ],
  "fast-input": [
    "Cross-entropy needs discrete targets, such as token IDs. Robot actions are continuous numbers.",
    "FAST solves this problem.",
    "The hope is that at the beginning, before any transformer layers, you literally have the FAST tokens in the input tensor.",
    "This is teacher forcing, the standard way to train a language model: the true tokens are part of the input, and the attention mask hides each token from the positions that must predict it. One forward pass then trains a prediction at every position at once."
  ],
  "loss-context": [
    "X_ctx : [B, S, 2560] is the context from the ingestion chapter. The blue arrow carries it to the concatenation step, where the embedded FAST tokens are appended to it."
  ],
  "fast-tokenizer": [
    "FAST (Pertsch et al., 2025) compresses an action chunk into a few integers, much as JPEG compresses an image:",
    { list: [
      "Normalize each action dimension to a common range.",
      "Apply a discrete cosine transform along time to each dimension separately. Smooth motions concentrate in a few low-frequency coefficients.",
      "Scale and round the coefficients to integers. Most high-frequency coefficients round to zero.",
      "Flatten the coefficients, lowest frequencies first, and merge frequent patterns with byte-pair encoding into a small vocabulary (1,024 tokens in the FAST paper)."
    ] },
    "The result is Z_FAST : [B, F], a sequence of F token IDs. F depends on how complex the motion is. The model's input uses only the first F − 1: [B, F − 1]. The last token never needs to be an input, because nothing after it is predicted (“we're trynna predict the last one”).",
    "In the non-realistic example (B = 1, F = 4, S = 3), Z_FAST = [17, 42, 9, 105]; the input keeps [17, 42, 9] and drops 105."
  ],
  "fast-embedding": [
    "Each FAST ID is assigned a reserved ID inside Gemma's own vocabulary, so the ordinary embedding table can embed it: each ID becomes 2560 numbers (the drawing's “Embedding(1, 2560)”: one ID in, one 2560-wide vector out). [B, F − 1] → [B, F − 1, 2560]. In the example, [17, 42, 9] becomes the vectors [f₁₇, f₄₂, f₉]."
  ],
  "fast-concatenate": [
    "The FAST embeddings are appended after the context: [B, S, 2560] + [B, F − 1, 2560] → [B, S + F − 1, 2560]. In the example, the Gemma input is [c₁, c₂, c₃, f₁₇, f₄₂, f₉]: six tokens, because S + F − 1 = 3 + 4 − 1 = 6."
  ],
  "fast-transformer": [
    "The sequence passes through the same 34 layers described in the transformer panel; the line labeled “old stuff” marks the end of what was already covered. Everything below it is new: turning the output vectors into a loss.",
    "During the 34 layers, the mask lets each FAST position read only the context and the FAST tokens before it. So after the last layer, the vector at each position can hold a prediction of the token that comes next."
  ],
  "fast-mask": [
    "The 6 × 6 mask from the example. Rows read, columns are read; ✓ is allowed and ✗ is masked.",
    { list: [
      "c₁, c₂, c₃ read each other and nothing else.",
      "f₁₇ reads the context and itself.",
      "f₄₂ reads the context, f₁₇, and itself.",
      "f₉ reads everything before it and itself."
    ] },
    "The intuition, as the drawing puts it: the f tokens come from ground-truth action data, and the c tokens are the context needed to decide on an action."
  ],
  "fast-positions": [
    "Only the positions that predict a FAST token are kept: the last context token and the first F − 1 FAST tokens, F positions in total. [B, S + F − 1, 2560] → [B, F, 2560].",
    "In the example these are the positions of c₃, f₁₇, f₄₂, and f₉. They should predict 17, 42, 9, and 105, the full FAST sequence including the token that was never an input."
  ],
  "fast-meaning": [
    "“The size of the tensor has not changed. The meaning has.” At the input, the position of f₁₇ held the embedding of token 17. After 34 layers of masked attention, it holds the model's guess for the token that follows 17. The F selected entries, where c₃, f₁₇, f₄₂, and f₉ used to be, now hold the most probable next entries (hopefully, respectively, 17, 42, 9, and 105)."
  ],
  "token-probabilities": [
    "After 34 transformer layers, at those same positions, you now have 262208-dimensional vectors carrying the probabilities for all next possible token IDs.",
    "Two steps produce them: a linear projection to one score per vocabulary entry, then a log-softmax, which turns the scores into log-probabilities."
  ],
  "vocabulary-projection": [
    "Linear(2560, 262208) gives every vocabulary entry one score. Gemma 3 reuses its input embedding table, transposed, for this projection, so a vector scores highly for a token when it points in the same direction as that token's embedding."
  ],
  "fast-logits": [
    "[B, F, 262208]: the logits, one unnormalized score for every possible next token at each of the F positions. Logits can be any real number; they are not probabilities yet."
  ],
  "fast-log-softmax": [
    "log-softmax(x)ᵢ = xᵢ − log Σⱼ exp(xⱼ). It subtracts the same constant from every score so that the exponentials sum to 1. The result is the logarithm of a probability distribution over the 262208 tokens."
  ],
  "fast-probabilities": [
    "[B, F, 262208]: log-probabilities, all 0 or negative. The drawing labels them “probabilities!”; exponentiating them gives the probabilities, which sum to 1 at each position. The loss uses the logarithms directly, so the exponentials are never computed."
  ],
  "cross-entropy": [
    "Cross-entropy turns the log-probabilities into one number: the average of −log p over the correct target tokens. It is small when the model gives the correct tokens high probability."
  ],
  "target-probability": [
    "“We only care about the probability assigned to the correct target ID.” Of the 262208 entries at a position, only the one at the true FAST token (red in the drawing) enters the loss. Raising it lowers all others, because the probabilities must sum to 1."
  ],
  "negative-log-likelihood": [
    "l_j = −log p(z_j | X_ctx, z_{<j}): the negative log-probability of the correct token z_j, given the context and the FAST tokens before it."
  ],
  "zero-loss": [
    "If p(z_j) were 1, the loss would be −log 1 = 0. As p(z_j) approaches 0, the loss grows without bound, so confidently wrong predictions are punished hardest."
  ],
  "token-confidence": [
    "The drawing's example: a fairly trained Gemma gives the correct tokens probabilities of 0.80, 0.60, 0.75, and 0.65. Their negative logarithms are 0.223, 0.511, 0.288, and 0.431."
  ],
  "mean-cross-entropy": [
    "Loss = (−ln 0.80 − ln 0.60 − ln 0.75 − ln 0.65) / 4 ≈ 1.452 / 4 ≈ 0.363. In a real batch the average runs over every target token of every example, B × F terms, excluding padding. Gradients of this loss update the backbone; how they flow back is left to the author's upcoming write-up on PyTorch autograd."
  ],

  "subgoal-policy": [
    "This chapter is the high-level policy, the model that decides what to do next. Despite the drawing's title, it does not produce subgoal images. It writes the next subtask as text, such as “Turn on the faucet”, and that text becomes the Subtask line of π0.7's prompt. The world model then draws images for that subtask.",
    "The paper says only that it is based on the same architecture as π0.7. The drawing uses the text half of that architecture: images and text go in, and next-token cross-entropy trains it to write the subtask. No action expert or FAST target appears.",
    "The paper's high-level policy is π0.7 fine-tuned on coaching data. A person first talks the robot through a new task step by step (“grasp the handle of the air fryer with the left hand”, …), and the policy is then trained to issue those same instructions on its own. For the five unseen tasks tested, the resulting autonomous runs roughly match the coached runs, without any new teleoperated action data.",
    "Subtask text is generated in its own thread at runtime; π0.7 always uses the latest subtask available.",
    "The drawing reuses the low-level pieces, so you can skim: the observation is encoded as in Ingestion, and the loss works as in the VLM loss."
  ],
  "sg-observation": [
    "The input is only the current frame: O = [B, 3, 3, 448, 448]. The first 3 counts cameras, the second counts color channels. No history frames are drawn. What the robot has already done is given in text instead, as the list of previous subtasks."
  ],
  "sg-current-frame": [
    "B examples × 3 cameras × 1 frame, each 448 × 448 pixels in RGB."
  ],
  "sg-patches": [
    "Each image becomes 32 × 32 = 1024 patches of 14 × 14 × 3 = 588 values, projected by Linear(588, 1152): [B, 3, 1024, 1152]. That is 1024 × 3 × B patches in total."
  ],
  "sg-position": [
    "SigLIP's learned positional embedding is added: one 1152-number vector per grid position. A single frame needs no time embedding. Reshaping moves the cameras into the batch axis for the encoder: [B·3, 1024, 1152]."
  ],
  "sg-vision": [
    "The same SigLIP encoder as in the low-level model: 27 layers of spatial attention, each token reading the other tokens of its own image. With one frame there is no temporal attention.",
    "The output is pooled to 256 tokens per image, widened to 2560, and the 3 cameras are concatenated: [B, 768, 2560].",
    { note: "the box label “[BV*6, 1024, 1152]” is carried over from the low-level drawing. Here the encoder's input is [B·3, 1024, 1152]: one frame for each of 3 cameras." }
  ],
  "sg-backbone": [
    "SigLIP's 27 transformer layers. Each layer applies attention within one image, then a 2-layer MLP. The red dashed lines are carried over from the low-level drawing, where they follow the temporal layers; this encoder has no temporal attention."
  ],
  "sg-spatial-attention": [
    "Each block: 16 attention heads over the 1024 tokens of one image, a residual connection, a 2-layer MLP, and a second residual connection."
  ],
  "sg-pooling": [
    "Reshaping restores the camera axis, [B·3, 1024, 1152] → [B, 3, 1024, 1152]. Average pooling with a 2 × 2 kernel merges neighboring patches: [B, 3, 256, 1152]. As in the low-level model, the flat list is mapped back onto the 32 × 32 grid with division and “mod” before pooling."
  ],
  "sg-projection": [
    "Linear(1152, 2560), Gemma 3's image projector: [B, 3, 256, 1152] → [B, 3, 256, 2560]."
  ],
  "sg-camera-tokens": [
    "Flattening the camera axis gives one sequence of image tokens: [B, 3, 256, 2560] → [B, 768, 2560]."
  ],
  "sg-text": [
    "The text input has three parts; a fourth, the next subtask, is the target:",
    { list: [
      "the task: “Clean the kitchen.”",
      "the previous subtasks, in order: open the upper cabinet, pick up the dirty plate, …, place the mug in the sink;",
      "the episode metadata: Quality 5, Speed Normal, Mistake None."
    ] },
    "The previous subtasks give the policy its memory of progress. Without them, a scene with the cabinet closed would look the same before and after the plate was put away.",
    "The paper lists the high-level policy's inputs as the observations, the task, and the history of past subtasks. The metadata line is included in the drawing."
  ],
  "sg-task": [
    "The overall instruction, the same task text the low-level model receives."
  ],
  "sg-history": [
    "The subtasks already issued in this episode, numbered in order. Each new subtask is appended to this list for the next prediction."
  ],
  "sg-metadata": [
    "Episode metadata written as text: Quality, Speed, and Mistake, as in the low-level prompt."
  ],
  "sg-text-tokenizer": [
    "The Gemma tokenizer turns the input text into L integer IDs: [B, L, (1)]."
  ],
  "sg-text-embedding": [
    "Gemma's embedding table turns each ID into 2560 numbers: [B, L, 2560]."
  ],
  "sg-context": [
    "The image tokens and the text tokens are concatenated along the sequence axis into the policy's context: [B, L + 256 × 3, 2560]. This plays the role X_ctx plays in the low-level model."
  ],
  "sg-context-concat": [
    "[B, 768, 2560] image tokens and [B, L, 2560] text tokens are joined into one sequence."
  ],
  "sg-context-tensor": [
    "[B, L + 256 × 3, 2560] = [B, L + 768, 2560]: everything the policy knows before it writes."
  ],
  "sg-next-subtask": [
    "The next subtask, “Turn on the faucet”, is the training target. It is prepared exactly like the FAST tokens in the VLM loss: tokenized, shifted for teacher forcing, embedded, and appended after the context."
  ],
  "sg-target-text": [
    "The annotated next step of the episode, shown in green in the drawing."
  ],
  "sg-target-tokenizer": [
    "Tokenizing the target gives L′ IDs: [B, L′, (1)]. The input uses the first L′ − 1: [B, L′ − 1, (1)]. As with FAST, the last token is only ever predicted, never read."
  ],
  "sg-target-embedding": [
    "The L′ − 1 input IDs are embedded with the same table: [B, L′ − 1, 2560]."
  ],
  "sg-combined-tokens": [
    "The target embeddings follow the context: [B, L + 768 + L′ − 1, 2560]. A causal mask lets each target position read the context and only the earlier target tokens."
  ],
  "sg-prediction": [
    "The sequence passes through Gemma's 34 layers. The L′ positions that predict target tokens (the last context position and the L′ − 1 target inputs) are projected to the vocabulary, and cross-entropy compares them with the true subtask tokens. The arithmetic is identical to the VLM loss.",
    "At runtime the policy writes the subtask one token at a time: it predicts a token, appends it to the input, and repeats until it predicts the end of the text."
  ],
  "sg-transformer": [
    "The same 34 Gemma 3 layers as the low-level backbone (“34 familiar transformer blocks”), with a causal mask over the target text."
  ],
  "sg-logits": [
    "[B, L′, 262208]: one score for every vocabulary entry at each of the L′ predicting positions."
  ],
  "sg-cross-entropy": [
    "Log-softmax, then the average of −log p over the L′ correct tokens: the same loss, computed the same way, as the VLM loss on FAST tokens."
  ],

  "world-model": [
    "The world model draws the subgoal images: for the current subtask, three pictures (front, left wrist, right wrist) of what the cameras should see when that subtask is done. π0.7 reads them as part of its prompt.",
    "It is a separate, larger network: 14 billion parameters, initialized from BAGEL, ByteDance's open model for image understanding, editing, and generation. Its inputs are the current images, the subtask text, and the episode metadata. Its output is three images, generated by flow matching.",
    "The reason for a separate model is transfer. BAGEL was pretrained on web-scale images and video, and the world model is further trained on robot data, human egocentric video, open-source image-editing data, and video data. It can therefore picture situations the robot data never shows. Subgoal images carry that knowledge into π0.7: with them, π0.7 follows unusual instructions more reliably, and they were critical in the paper's “Reverse Fridge to Microwave” task, and they improved zero-shot towel and shirt folding on a robot that had no folding data. Results with generated subgoals are labeled “π0.7 (GC)” in the paper's plots.",
    { placeholder: "Diagram in progress. The dashed slots on the canvas mark the five parts this chapter describes." }
  ],
  "wm-inputs": [
    "Each training example has a subtask instruction, the 3 current camera images o_t, and 3 target images o_{t_end}: the frames at the last timestep of the subtask segment that contains t. The episode metadata is also an input.",
    "The examples come from robot data and human egocentric video with especially accurate subtask labels. The paper reports that label quality, especially how precisely segments start and end, strongly affects subgoal quality."
  ],
  "wm-encoders": [
    "Following BAGEL, every image is encoded twice:",
    { list: [
      "**ViT, for meaning.** A SigLIP-style encoder with 14-pixel patches. Images are resized to 448 × 336, which gives 32 × 24 = 768 tokens per image.",
      "**VAE, for detail.** An image autoencoder that keeps fine pixel detail; with its patching, each token covers 16 pixels. Images are resized to 512 × 384, which also gives 32 × 24 = 768 tokens per image. The images to be generated live in this space."
    ] },
    "During training the model receives three copies of the scene: the current images through the ViT, the current images through the VAE, and the noisy target images in VAE space. Each group of three camera views attends to itself in both directions."
  ],
  "wm-backbone": [
    "BAGEL is a mixture of transformers. BAGEL starts from the Qwen2.5 7B language model. Two 7-billion-parameter transformers share one self-attention operation per layer, the same arrangement as π0.7's backbone and action expert:",
    { list: [
      "the understanding transformer processes the text and the ViT tokens;",
      "the generation transformer processes the VAE tokens."
    ] },
    "Each token goes through one of the two, so about 7 billion parameters are active per token out of 14 billion in total."
  ],
  "wm-loss": [
    "The world model is trained with flow matching on the target images. The VAE tokens of the three target images are mixed with noise, the generation transformer predicts the velocity from the noisy tokens back toward the clean ones, and a squared error compares it with the true velocity. It is the same idea as the EAM loss, applied to image tokens instead of actions.",
    "Generation is conditioned on the current images, the subtask, and the metadata. Because every target image is a real later frame of the same episode, the model learns what progress on a subtask looks like from the robot's own cameras."
  ],
  "wm-inference": [
    "At runtime the world model runs alongside π0.7, not in front of it:",
    { list: [
      "**When.** A new set of subgoals is generated whenever the subtask changes, or 4 seconds after the previous set, whichever comes first. Generation runs in its own thread; π0.7 keeps acting on the previous subgoals until the new ones arrive.",
      "**How.** 25 denoising steps, each using classifier-free guidance on both the text and the image. The guided and unguided variants are packed into one sequence with a mask that keeps them from reading each other, so one forward pass computes all three.",
      "**How fast.** With a sequence of nearly 10,000 tokens, one set of subgoals takes 1.25 seconds on 4 H100 GPUs, using 4-way tensor parallelism, 8-bit matrix multiplication, and a modified SageAttention kernel."
    ] },
    "Generated images are not perfect, so π0.7 is trained on them as well. Besides real future frames, many training examples contain world-model images in place of the real ones, which narrows the gap between what π0.7 sees in training and at runtime."
  ]
};
