// The inference walkthrough: one entry per block of the inference drawing, in time order.
// bounds are [x, y, width, height] in the drawing's native SVG units (viewBox 0 0 4510 986).
// stage tells the 3D view what to show: a beat (camera and overlay choreography) and the
// robot's task time, tau. tau counts subtasks: 0–1 is the first subtask of a cycle (A), 1–2 the
// second (B). caption: which subtask the high-level policy has issued; type animates its writing.
// goal: [subtask, from, to] progress of BAGEL's images. chunk/pending: tau range of a new action
// chunk (pending ones are computed while the previous chunk still executes); splice switches to it.
// Paragraph blocks follow article.js: strings (with **bold**), { list }, and { note }.

export const drawing = { src: './assets/inference.svg', width: 4510.057716404866, height: 985.9905687118917 };

export const task = 'Wash the plate';
// The demo loops through these; each pass through the timeline covers two of them.
export const subtasks = ['Open the faucet', 'Scrub the plate', 'Put the sponge back', 'Close the faucet'];

export const lanes = {
  motors: { title: 'Motors', kind: 'hardware' },
  sensors: { title: 'Sensors', kind: 'hardware' },
  low: { title: 'Low-level VLM (π0.7)', kind: 'model' },
  high: { title: 'High-level VLM', kind: 'model' },
  bagel: { title: 'BAGEL world model', kind: 'model' }
};

export const steps = [
  {
    id: 'overview', lane: null, time: null, title: 'What happens when the robot runs',
    bounds: [0, 0, 4510, 986],
    stage: { beat: 'establish', tau: 0, caption: null, dwell: 11 },
    text: [
      "π0.7 is a robot control model from Physical Intelligence (arXiv 2604.15483). This page follows one robot as it runs, block by block, through the inference timeline on the left. The Training tab explains how the model inside each block is built and trained.",
      "Read the drawing as five clocks running side by side. The two blue lanes are hardware: **Motors** execute actions and **Sensors** read the cameras and joints. The three white lanes are models: the **low-level VLM**, π0.7 itself, turns observations into actions; the **high-level VLM** writes the next subtask as text; and the **BAGEL world model** draws pictures of what the scene should look like when that subtask is done.",
      { list: [
        "Yellow tags are clock times. t = 0 ms is the moment the first action executes.",
        "Green blocks read sensors, pink blocks run a model, and grey blocks execute actions.",
        "Green arrows carry data from one lane to another."
      ] },
      "After startup, the three models run in separate threads and do not wait for each other: each one uses the latest output of the others that is available. That is why the slow world model (BAGEL) does not slow down the robot.",
      "The 3D view replays each block on two SO-101 arms with an overhead camera and a camera on each wrist, the rig this project targets. Real inference happens in milliseconds; the view slows it down and exaggerates it so each step can be seen. The motions are scripted: no model runs in your browser. Use ← and → (or the buttons) to move between blocks. Left alone, the walkthrough plays and loops, and each loop moves the robot on to its next two subtasks."
    ]
  },
  {
    id: 'startup', lane: 'sensors', time: 'before t = 0', title: 'Startup: read the sensors',
    bounds: [362, 380, 240, 130],
    stage: { beat: 'capture-full', tau: 0, caption: null, dwell: 12 },
    text: [
      "Before anything moves, the robot collects its first observation.",
      { list: [
        "**Camera frames.** One image from each camera, resized to 448 × 448 pixels. This rig has three: an overhead camera on the gantry, which plays the role of the paper's front camera, and one camera on each wrist.",
        "**Proprioception.** The current joint positions of both arms: five joints and a gripper per SO-101.",
        "**Metadata.** Not measured; set by the operator. At runtime the paper always asks for the best behavior seen in training: quality 5, no mistake, and a speed equal to the 15th percentile of that task's episode lengths. The control mode (joint or end-effector) is set here too."
      ] },
      "During normal running, π0.7 also sees up to 5 earlier frames per camera, 1 second apart, with the matching joint states. At startup there is no history yet. The model handles that: during training the whole history is removed from 30% of examples."
    ]
  },
  {
    id: 'hl0', lane: 'high', time: 'before t = 0', title: 'The high-level policy writes the first subtask',
    bounds: [375, 695, 300, 130],
    stage: { beat: 'subtask', tau: 0, caption: 'A', type: true, dwell: 9 },
    text: [
      "The high-level policy reads the current camera images, the overall task, and the list of subtasks issued so far (empty at startup). It writes the next subtask as text, one token at a time: ℓ̂₀ = “Open the faucet”.",
      "It has the same Gemma-based architecture as π0.7, fine-tuned on coaching data: a person first talked the robot through tasks step by step, and the policy learned to issue those instructions on its own.",
      { note: "this lane is labelled “subgoal generator”. Its subgoal is the subtask text. The subgoal images come from the BAGEL lane below." }
    ]
  },
  {
    id: 'bagel0', lane: 'bagel', time: 'before t = 0', title: 'BAGEL draws the first subgoal images',
    bounds: [412, 855, 245, 128],
    stage: { beat: 'subgoal', tau: 0, caption: 'A', goal: ['A', 0, 1], dwell: 10 },
    text: [
      "The world model receives the subtask ℓ̂₀, the current camera images, and the metadata. It generates g = [G¹, G², G³]: one image per camera, showing what that camera should see when the subtask is done. Here, the faucet lever raised and water running.",
      "It is a 14-billion-parameter model initialized from BAGEL, an open image generation and editing model. It generates each set of images in 25 denoising steps, starting from noise. The paper measures 1.25 seconds per set on four H100 GPUs.",
      "At startup, π0.7 waits for this first set (Algorithm 1 in the paper samples the subgoals before the first action chunk). After that, new images are generated in the background whenever the subtask changes or 4 seconds have passed."
    ]
  },
  {
    id: 'll1', lane: 'low', time: 'before t = 0', title: 'π0.7 computes the first action chunk',
    bounds: [688, 525, 372, 150],
    stage: { beat: 'chunk', tau: 0, caption: 'A', goal: ['A', 1, 1], chunk: [0, 0.45], dwell: 11 },
    text: [
      "With the full observation and the prompt in hand (task, subtask, subgoal images, metadata, control mode), π0.7 computes an action chunk: the next 50 actions, [B, 50, d], with B = 1 at runtime. At 50 actions per second, 50 actions cover one second of motion.",
      "The Gemma backbone reads the context once. The action expert then starts from random noise, the spiral in the drawing, and refines it in 5 denoising steps into a clean chunk, the smile. In the 3D view, each dot is the gripper's position after one of the 50 actions: the dots start scattered and settle onto the path in 5 jumps.",
      "On one H100 GPU this takes 38 ms in the minimal configuration (three cameras, no history, no subgoal images) and up to 127 ms with history and subgoal images.",
      "Optionally, π0.7 can run each denoising step twice, with and without the metadata, and push the result further in the direction the metadata points. This is classifier-free guidance; the paper uses strengths of 1.3 to 2.2 for dexterous tasks."
    ]
  },
  {
    id: 'm0', lane: 'motors', time: 't = 0 ms', title: 'The first action executes',
    bounds: [695, 243, 372, 100],
    stage: { beat: 'execute-close', tau: 0.04, caption: 'A', goal: ['A', 1, 1], dwell: 8 },
    text: [
      "The first action, a₀, reaches the motors. In joint control mode, an action is a target position for every joint; a simple PD controller drives each joint toward its target (the SO-101's STS3215 servos run this position loop on board).",
      "The robot runs at 50 Hz, so a new action is applied every 20 ms. The real arm moves only millimeters in that time; the 3D view exaggerates it."
    ]
  },
  {
    id: 'm1', lane: 'motors', time: 't = 20–300 ms', title: 'Actions a₁ to a₁₄ execute',
    bounds: [1052, 240, 345, 100],
    stage: { beat: 'execute', tau: 0.27, caption: 'A', goal: ['A', 1, 1], dwell: 9 },
    text: [
      "The robot executes the chunk one action every 20 ms without asking the model again. π0.7 executes only the first 15 or 25 of the 50 actions; the drawing uses 15, so a₀ … a₁₄ take 15 × 20 ms = 300 ms.",
      "Executing the whole chunk would act on a one-second-old plan. Asking the model for every single action would be impossible, because one inference takes longer than 20 ms. Executing 15 actions and then planning again keeps the robot both smooth and responsive."
    ]
  },
  {
    id: 's300', lane: 'sensors', time: 't = 300 ms', title: 'A new observation',
    bounds: [1158, 365, 252, 135],
    stage: { beat: 'capture-quick', tau: 0.32, caption: 'A', goal: ['A', 1, 1], dwell: 8 },
    text: [
      "After 15 actions the sensors are read again: new camera frames and joint positions. The metadata does not change. This observation starts the next inference."
    ]
  },
  {
    id: 'll2', lane: 'low', time: 't = 300–440 ms', title: 'The second chunk is computed while the arm keeps moving',
    bounds: [1328, 525, 370, 150],
    stage: { beat: 'chunk-overlap', tau: 0.37, caption: 'A', goal: ['A', 1, 1], pending: [0.3, 1], dwell: 10 },
    text: [
      "π0.7 computes a new 50-action chunk from the new observation, with the same subtask and subgoal images. The drawing gives it 140 ms (the paper measures 38 to 127 ms, depending on the configuration).",
      "The robot does not stop to wait. The actions that will execute during those 140 ms are already fixed: they are the rest of the first chunk. π0.7 is told which actions those are and generates a chunk that begins with exactly them. This is real-time chunking. During training, π0.7 practiced it: the first 0 to 12 actions of a chunk were given to it without noise, and it learned to continue them."
    ]
  },
  {
    id: 'm15', lane: 'motors', time: 't = 300–440 ms', title: 'Meanwhile, actions a₁₅ to a₂₁ execute',
    bounds: [1308, 245, 388, 100],
    stage: { beat: 'execute', tau: 0.43, caption: 'A', goal: ['A', 1, 1], dwell: 8 },
    text: [
      "While the GPU computes, the motors keep executing the first chunk: a₁₅ … a₂₁, seven actions in 140 ms. Nothing in the robot waits for a model."
    ]
  },
  {
    id: 'm440', lane: 'motors', time: 't = 440 ms', title: 'Switch to the second chunk',
    bounds: [1640, 228, 410, 165],
    stage: { beat: 'splice', tau: 0.82, caption: 'A', goal: ['A', 1, 1], splice: true, dwell: 11 },
    text: [
      "The second chunk arrives. Its actions were planned from the observation taken at 300 ms, so its first actions describe moments that have already passed: the robot executed them from the first chunk while the second was being computed.",
      "Execution therefore enters the new chunk partway in, at the action that matches the current time. The drawing marks it as a₆ of the second chunk: “it is already 6 actions behind! thus the sixth and not the first!”. Because real-time chunking made the new chunk begin with the actions the robot actually took, the handover is smooth.",
      "From here the cycle repeats: execute, observe, compute the next chunk in the background, switch."
    ]
  },
  {
    id: 'hl600', lane: 'high', time: 't = 600 ms', title: 'The subtask changes',
    bounds: [1932, 692, 320, 145],
    stage: { beat: 'subtask', tau: 0.9, caption: 'B', type: true, goal: ['A', 1, 1], dwell: 9 },
    text: [
      "The high-level policy runs again on the latest observation. With “Open the faucet” in its list of finished subtasks, it writes the next one: ℓ̂ = “Scrub the plate”.",
      "The paper runs subtask generation in its own thread but does not say how often it runs. The drawing triggers it at 600 ms."
    ]
  },
  {
    id: 'bagel600', lane: 'bagel', time: 't = 600–2100 ms', title: 'BAGEL redraws the subgoals, slowly',
    bounds: [2138, 855, 1402, 128],
    stage: { beat: 'subgoal-slow', tau: 0.95, caption: 'B', goal: ['B', 0, 0.4], dwell: 11 },
    text: [
      "A new subtask makes the old subgoal images obsolete, so the world model starts on a new set, from the current images and “Scrub the plate”.",
      "“BAGEL is hella slow”: the drawing gives it 1.5 seconds, 75 robot actions. The paper measures 1.25 seconds, even with four H100 GPUs, 8-bit matrix multiplication, and a sequence of nearly 10,000 tokens processed 25 times, with guidance on both the text and the image at every step.",
      "Generation does not block anything. π0.7 keeps acting while BAGEL works."
    ]
  },
  {
    id: 'm600', lane: 'motors', time: 't = 600–900 ms', title: 'The arm finishes the old chunk',
    bounds: [2020, 240, 335, 105],
    stage: { beat: 'execute', tau: 1, caption: 'B', goal: ['B', 0.4, 0.55], dwell: 8 },
    text: [
      "The motors execute the rest of the second chunk, a₁₆ … a₃₀ in the drawing. That chunk was planned under “Open the faucet”; the new subtask takes effect at the next inference."
    ]
  },
  {
    id: 's900', lane: 'sensors', time: 't = 900 ms', title: 'Observation',
    bounds: [2125, 368, 255, 138],
    stage: { beat: 'capture-quick', tau: 1, caption: 'B', goal: ['B', 0.55, 0.65], dwell: 7 },
    text: [
      "The same read as at 300 ms: new frames and joint positions, 15 actions after the last switch."
    ]
  },
  {
    id: 'll3', lane: 'low', time: 't = 900–1040 ms', title: 'The third chunk, under the new subtask',
    bounds: [2305, 525, 372, 150],
    stage: { beat: 'chunk', tau: 1.04, caption: 'B', goal: ['B', 0.65, 0.8], chunk: [1, 1.35], dwell: 10 },
    text: [
      "This chunk is the first planned with “Scrub the plate”. Its subgoal images do not exist yet: BAGEL is still working.",
      "The paper says π0.7 always uses the latest subgoals available, but not whether it keeps showing the images for the old subtask or leaves them out during this gap. Either way the model is prepared: during training, subgoal images were missing from 75% of examples, so π0.7 can act on the subtask text alone.",
      { note: "the end tag of this block reads t = 440 ms; it should be t = 1040 ms." }
    ]
  },
  {
    id: 'm900', lane: 'motors', time: 't = 900–1040 ms', title: 'Execute, observe, compute, switch',
    bounds: [2295, 245, 385, 100],
    stage: { beat: 'execute', tau: 1.16, caption: 'B', goal: ['B', 0.8, 0.9], dwell: 8 },
    text: [
      "The motors keep executing until the third chunk is ready at 1040 ms, then switch to it as before. From here the drawing elides the repetition (“…”): every 300 ms a new observation, about 140 ms of inference, and a switch."
    ]
  },
  {
    id: 'll2100', lane: 'low', time: 't = 2100 ms', title: 'The new subgoal images arrive',
    bounds: [3435, 675, 480, 135],
    stage: { beat: 'subgoal-land', tau: 2, caption: 'B', goal: ['B', 0.9, 1], chunk: [1.15, 2], dwell: 16 },
    text: [
      "BAGEL finishes. From the next inference on, every chunk is planned with images of the plate scrubbed clean. The images stay in use until the subtask changes or 4 seconds pass, when BAGEL redraws them from the latest observation.",
      "The robot continues this way until the task is done: the high-level policy changes the subtask, BAGEL redraws the goal, and π0.7 turns both into 50-action chunks every few hundred milliseconds.",
      "The walkthrough now loops. Each pass moves the robot on to its next two subtasks: put the sponge back and close the faucet, then open it again.",
      { note: "this block is drawn on the High-level lane, but planning action chunks is the low-level policy's job." }
    ]
  }
];
