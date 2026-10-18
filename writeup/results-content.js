// Results tab: the experiment tree. Each node is one thing we tried, in the order we tried it.
// kind: 'main' (kept and built on), 'dead' (tried and dropped), 'running' (being measured now).
// runs: training-run ids; their scores, progress and filmed clips come from results-runs.js, which
// scripts/sync_results_site.py regenerates from the training boxes. Text holds only settled numbers.
// depth 0 sits on the main line; depth 1 branches off the node above it.
// score: strict successes (the object really lifted, not pushed) out of `of` trials; omitted when not measured.
// clips: rollout videos filmed by scripts/record_showcase.py on one of the test layouts; rate is that model's
// strict successes on that task over all 30 layouts, when it was measured.

export const tasks = {
  pick_place: { name: 'Pick and place, one arm', short: 'One arm', goal: 'Put the cube on the round target.' },
  pick_place_any: { name: 'Pick and place', short: 'Pick and place', goal: 'Put the cube on the round target.' },
  bin: { name: 'Into the bin', short: 'Bin', goal: 'Pick up the cube and drop it in the bin.' },
  handover: { name: 'Handover', short: 'Handover', goal: 'One arm passes the cube to the other, which puts it in the bin.' },
  two_bins: { name: 'Two cubes, two arms', short: 'Two cubes', goal: 'Each arm puts its own cube in the shared bin.' },
};

const clip = (stem, task, layout, success, seconds, rate, label) =>
  ({ task, layout, success, seconds, rate, label, video: `assets/results/${stem}.mp4`, poster: `assets/results/${stem}.jpg` });

// Starting points of the lineage tree: models we started from but did not train ourselves.
export const treeRoots = [
  { id: 'root-gemma', name: 'Gemma 3', note: 'A model trained on images and text, never on robots.' },
  { id: 'root-pi05', name: 'π0.5', note: 'A model pretrained on more than 10,000 hours of real robot data.' },
];

// Lineage (Tree view): each node hangs from the checkpoint its runs resumed from (results-runs.js `lineage`,
// read from runs/<run>/run.json); `from` sets it by hand where no run records it; `short` is the tree label.
export const nodes = [
  {
    id: 'paper', short: 'π0.7 from the paper', from: 'root-gemma', kind: 'dead', depth: 0, when: 'Oct 14–15',
    name: 'π0.7 built from the paper',
    score: { value: 2, of: 20, label: 'best checkpoint, one arm' },
    changed: 'We built the full π0.7 model from the paper at its real size (about 5 billion parameters) and trained it on our scripted demonstrations, starting from Gemma 3, a model trained on images and text.',
    result: 'It never learned to find the cube in its camera images. The best checkpoint picked and placed the cube 2 times in 20, most none at all. A smaller version did 1 in 20.',
    next: 'Gemma 3 had never seen a robot. We needed a starting point that already connects pictures to arm movements.',
    clips: [],
  },
  {
    id: 'public', short: 'Public model', from: 'root-pi05', kind: 'dead', depth: 1, when: 'Oct 15',
    name: 'A public robot model, as is',
    score: { value: 0, of: 10, label: 'first test' },
    changed: 'We tried a publicly released π0.5 model fine-tuned for SO-101 arms, without any training of our own.',
    result: 'It failed every attempt at the simplest test in our simulator.',
    next: 'Useful pretrained weights, wrong world: it had to learn our scene.',
    clips: [],
  },
  {
    id: 'pi05-single', short: 'π0.5, one arm', from: 'root-pi05', kind: 'main', depth: 0, when: 'Oct 15',
    name: 'Switch to π0.5’s pretrained brain, one arm',
    score: { value: 26, of: 30, label: 'one arm' },
    changed: 'We kept our data and the π0.7 way of prompting, but started from π0.5, a model pretrained on more than 10,000 hours of real robot data.',
    result: 'The same simple task went from almost never to 26 of 30.',
    next: 'Vision worked. Time for both arms and harder tasks.',
    clips: [],
  },
  {
    id: 'two-arms', short: 'Two arms, 4 tasks', from: 'root-pi05', kind: 'main', depth: 0, when: 'Oct 16',
    name: 'Two arms, four tasks',
    score: { value: 83, of: 120 },
    changed: 'One model for both arms and four tasks of rising difficulty, trained for 20,000 steps on 4,800 demonstrations.',
    result: 'It improved as training went on: 75, then 76, then 83 of 120 at steps 13,000, 17,000 and 20,000. Handing the cube between arms and coordinating two cubes stayed the hard part.',
    next: 'The arms often got stuck in situations the demonstrations never showed.',
    clips: [],
  },
  {
    id: 'soup', short: 'Checkpoint average', from: 'two-arms', kind: 'dead', depth: 1, when: 'Oct 16',
    name: 'Average several checkpoints',
    score: { value: 74, of: 120 },
    changed: 'Checkpoints were good at different tasks, so we averaged six of them into one model.',
    result: 'No better than any single checkpoint.',
    next: 'Dropped.',
    clips: [],
  },
  {
    id: 'recover', short: 'Staged mistakes', from: 'two-arms', kind: 'dead', depth: 1, when: 'Oct 16',
    name: 'Practice recovering from staged mistakes',
    score: { value: 72, of: 120 },
    changed: 'We added demonstrations with deliberate fumbles (knocked cubes, missed grasps) followed by recoveries.',
    result: 'Worse than without them.',
    next: 'Staged mistakes are not the mistakes the model actually makes.',
    clips: [],
  },
  {
    id: 'restart', short: 'Different start', from: 'pi05-single', kind: 'dead', depth: 1, when: 'Oct 16',
    name: 'A second run from a different start',
    score: { value: 52, of: 120 },
    changed: 'A separate two-arm run warm-started from the one-arm model, with the recovery data mixed in.',
    result: 'Well behind the main run.',
    next: 'Dropped.',
    clips: [],
  },
  {
    id: 'dagger', short: 'From its mistakes', from: 'two-arms', kind: 'main', depth: 0, when: 'Oct 16–17',
    name: 'Learning from its own mistakes',
    score: { value: 86, of: 120 },
    changed: 'We let the model run, and when it got stuck the scripted teacher took over from exactly that moment. Those rescues became new training data (a method called DAgger), in two rounds.',
    result: '85, then 86 of 120. The second arm in the two-cube task improved most, from 12 to 18 of 30.',
    next: 'Every rescue round gave a few points. Next: train longer and more gently.',
    clips: [],
  },
  {
    id: 'gentle', short: 'Gentle training', kind: 'main', depth: 0, when: 'Oct 17',
    name: 'Longer, gentler training',
    runs: ['x_control_s0', 'x_control_s1'],
    changed: 'The same model and data, trained 6,000 more steps with smaller updates. Run twice with different random seeds, because two identical runs landed 16 points apart at the same checkpoint.',
    result: 'Across both runs it averages 97 of 120 over eight checkpoints, against 90 for the model it started from. The best single checkpoint reached 104.',
    next: 'Every idea below is measured against these two runs.',
    clips: [],
  },
  {
    id: 'robust', short: 'Lighting and colours', from: 'gentle', kind: 'running', depth: 1, when: 'Oct 17–18', robust: true,
    name: 'Does it still work with different lighting and colours?',
    changed: 'Every training example had random table and cube colours, lighting and small camera shifts, but the main scores use the simulator’s plain look. Here the same 30 layouts per task are scored again with that randomness switched on.',
    result: 'So far the gain from gentle training carries over, though less cleanly than the plain look suggests: the table compares each model on both.',
    next: 'More checkpoints are being scored on the small GPU as they finish.',
    clips: [],
  },
  {
    id: 'rtc', short: 'Acting while thinking', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'Acting while thinking',
    runs: ['x_rtc_s0', 'x_rtc_s1'],
    changed: 'A real robot keeps moving while the model computes its next move. This trains the model to continue smoothly from the actions it already committed to.',
    result: 'With a delay it did worse, not better. At its last checkpoint it scored 89 and 76 of 120 with delays of 0.12 and 0.24 seconds; the plain run from the line above scored 99 and 82 with the same delays.',
    next: 'In this simulator acting on slightly old information costs little, so there was nothing for it to fix.',
    clips: [],
  },
  {
    id: 'memory', short: 'Memory', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'Memory of the last few seconds',
    runs: ['x_mem_s0', 'x_mem_s1'],
    changed: 'Instead of only the current camera frame, the model also sees frames from the last five seconds, so it could remember where things were.',
    result: 'Both runs averaged about 94 (94 and 95 over their checkpoints), below the plain runs’ 97, and the two-cube task, where memory should help most, did not improve.',
    next: 'Dropped: in these short tasks the current frame already holds what the model needs.',
    clips: [],
  },
  {
    id: 'goals', short: 'Goal pictures', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'Pictures of the goal',
    runs: ['x_goal_s0', 'x_goal_s1'],
    changed: 'The model also learns from a picture of what the scene should look like two seconds ahead. At test time the simulator supplies a perfect picture.',
    result: 'Without a goal picture it does as well as the plain runs (105 at its last checkpoint). With perfect goal pictures it does worse on both seeds: 90 instead of 105, and 75 instead of 89.',
    next: 'If perfect pictures hurt, a model that draws them would not help this policy.',
    clips: [],
  },
  {
    id: 'goals-most', short: 'Goal pictures, more', kind: 'running', depth: 1, when: 'Oct 17',
    name: 'Goal pictures in most training examples',
    runs: ['x_goal75_s0'],
    changed: 'The same idea, but with a goal picture in three of every four training examples instead of one in four, in case the model simply never learned to use them.',
    result: 'Scored both with and without goal pictures.',
    next: 'The last test of this idea.',
    clips: [],
  },
  {
    id: 'insulation', short: 'Protected vision', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'Protecting the vision part',
    runs: ['x_ki_s0'],
    changed: 'π0.7 trains its vision-and-language part separately from the part that moves the arms. We tried that here.',
    result: 'It collapsed at every checkpoint: the arm reaches the cube but never lifts it.',
    next: 'A gentler version followed.',
    clips: [],
  },
  {
    id: 'insulation-slow', short: 'Protected, slowly', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'Protecting it, more slowly',
    runs: ['x_ki_vlm01_s0'],
    changed: 'The same, with the vision-and-language part learning ten times more slowly, so the part that moves the arms could keep up.',
    result: 'It collapsed too.',
    next: 'Added on top of an already trained model, this kind of training broke it at both speeds we tried.',
    clips: [],
  },
  {
    id: 'rescues-1', short: 'Rescues, previous model', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'More rescues, from the previous model',
    runs: ['x_data_v11', 'x_data_v11_s1'],
    changed: 'About 980 new rescue episodes on the two hard tasks: the previous best model ran, and the scripted teacher took over whenever it got stuck. Added to the training data.',
    result: 'Both runs averaged about 95 (95 and 96 over their checkpoints), below the plain runs’ 97, and handover, the task these rescues targeted, did not improve.',
    next: 'Dropped: rescues collected from an older model teach situations the current model rarely reaches. The next batch comes from the best model instead.',
    clips: [],
  },
  {
    id: 'rescues-2', short: 'Rescues, best model', kind: 'running', depth: 1, when: 'Oct 17–18',
    name: 'More rescues, from the best model',
    runs: ['x_data_b2'],
    changed: 'About 510 rescue episodes collected from the current best model instead, added to the data, and trained further from that model.',
    result: 'Compared against the second round of plain training below, which starts from the same model without the new data.',
    next: 'Kept only if it beats that comparison.',
    clips: [],
  },
  {
    id: 'longer', short: 'Twice as long', kind: 'running', depth: 1, when: 'Oct 17–18',
    name: 'Twice as long',
    runs: ['x_control_long_s0'],
    changed: 'The plain recipe run for 12,000 steps instead of 6,000, since the plain runs were still improving when they stopped.',
    result: 'Scored every 2,000 steps.',
    next: 'If it ends higher, the next combined run trains longer.',
    clips: [],
  },
  {
    id: 'round-two', short: 'Round two', kind: 'running', depth: 1, when: 'Oct 17–18',
    name: 'A second round from the best checkpoint',
    runs: ['x_control_r2_s0', 'x_control_r2_s1'],
    changed: 'The same gentle training again, starting from the best checkpoint (104), to see whether the gain keeps adding up. Also the comparison for the two experiments that start from that checkpoint.',
    result: 'Two seeds.',
    next: 'Shows whether repeating the recipe still pays.',
    clips: [],
  },
  {
    id: 'two-cubes-more', short: 'Two cubes, more often', kind: 'running', depth: 1, when: 'Oct 18',
    name: 'Practise the two-cube task more',
    runs: ['x_twobins2x_r2'],
    changed: 'The same second round, but showing the two-cube task twice as often, since it is now the weakest of the four.',
    result: 'Compared against the second round above.',
    next: 'Kept if the two-cube task improves without the others slipping.',
    clips: [],
  },
];

// Clips attach to nodes here, so a new recording is one line.
const add = (id, ...clips) => nodes.find(n => n.id === id).clips.push(...clips);
add('gentle',
  clip('pick_place_any_c1_5000_s4000', 'pick_place_any', 1, true, 3.5, 29, 'Seed 1 · step 5,000'),
  clip('bin_c1_5000_s4001', 'bin', 2, true, 9.0, 30, 'Seed 1 · step 5,000'),
  clip('handover_c1_5000_s4002', 'handover', 3, true, 12.0, 24, 'Seed 1 · step 5,000'),
  clip('two_bins_c1_5000_s4003', 'two_bins', 4, false, 30.0, 21, 'Seed 1 · step 5,000 · layout 4'),
  clip('two_bins_c1_5000_s4004', 'two_bins', 5, true, 10.8, 21, 'Seed 1 · step 5,000 · layout 5'),
  clip('two_bins_c1_5000_s4005', 'two_bins', 6, true, 18.0, 21, 'Seed 1 · step 5,000 · layout 6'));
add('pi05-single',
  clip('pick_place_single_6000_s4000', 'pick_place', 1, true, 4.0, 26));
add('dagger',
  clip('handover_dagger2_7000_s4002', 'handover', 3, true, 25.0, 16),
  clip('two_bins_dagger2_7000_s4003', 'two_bins', 4, false, 30.0, 18));
add('two-arms',
  clip('handover_v1_8000_s4002', 'handover', 3, false, 30.0, undefined, 'Step 8,000'),
  clip('two_bins_v1_8000_s4003', 'two_bins', 4, false, 30.0, undefined, 'Step 8,000'),
  clip('handover_v1_20000_s4002', 'handover', 3, false, 30.0, 19, 'Step 20,000'),
  clip('two_bins_v1_20000_s4003', 'two_bins', 4, true, 7.9, 12, 'Step 20,000'));
add('paper',
  { ...clip('pick_place_full_v1_5000_s4000', 'pick_place', 1, false, 30.0, 0, 'Step 5,000'), of: 20 });

// Loss view: the training runs drawn for each node. A node with `runs` draws them by default (one line per seed);
// `lines` overrides that with [run, label] pairs, `ref` adds the matched control as a dashed reference, and `note`
// replaces the chart when there was no training of our own. Curves come from results-loss.js (generated).
const CONTROL = [['x_control_s0', 'Control'], ['x_control_s1', 'Control']];
const ROUND_TWO = [['x_control_r2_s0', 'Control'], ['x_control_r2_s1', 'Control']];
export const lossPlan = {
  paper: { lines: [['full_v1', 'Paper size (5B)'], ['small_v1', 'Small (0.3B)']] },
  public: { note: 'This step had no training of our own, so there is no loss curve.' },
  robust: { note: 'This is a test of finished models, not a training run, so there is no loss curve.' },
  'pi05-single': { lines: [['pi05_single', 'One arm']] },
  'two-arms': { lines: [['pi05_v1', 'Two arms, four tasks']] },
  soup: { note: 'Averaging checkpoints involves no training, so there is no loss curve.' },
  recover: { lines: [['pi05_v1_rec2', 'With staged mistakes']] },
  restart: { lines: [['pi05_2arm_v2', 'Warm-started run']] },
  dagger: { lines: [['pi05_dagger1', 'Round 1'], ['pi05_dagger2', 'Round 2']] },
  rtc: { ref: CONTROL }, memory: { ref: CONTROL }, goals: { ref: CONTROL }, 'goals-most': { ref: CONTROL },
  insulation: { ref: CONTROL }, 'insulation-slow': { ref: CONTROL }, 'rescues-1': { ref: CONTROL }, longer: { ref: CONTROL },
  'rescues-2': { ref: ROUND_TWO }, 'two-cubes-more': { ref: ROUND_TWO },
};
