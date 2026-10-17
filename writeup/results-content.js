// Results tab: the experiment tree. Each node is one thing we tried, in the order we tried it.
// kind: 'main' (kept and built on), 'dead' (tried and dropped), 'running' (being measured now).
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

export const nodes = [
  {
    id: 'paper', kind: 'dead', depth: 0, when: 'Oct 14–15',
    name: 'π0.7 built from the paper',
    score: { value: 2, of: 20, label: 'best checkpoint, one arm' },
    changed: 'We built the full π0.7 model from the paper at its real size (about 5 billion parameters) and trained it on our scripted demonstrations, starting from Gemma 3, a model trained on images and text.',
    result: 'It never learned to find the cube in its camera images. The best checkpoint picked and placed the cube 2 times in 20, most none at all. A smaller version did 1 in 20.',
    next: 'Gemma 3 had never seen a robot. We needed a starting point that already connects pictures to arm movements.',
    clips: [],
  },
  {
    id: 'public', kind: 'dead', depth: 1, when: 'Oct 15',
    name: 'A public robot model, as is',
    score: { value: 0, of: 10, label: 'first test' },
    changed: 'We tried a publicly released π0.5 model fine-tuned for SO-101 arms, without any training of our own.',
    result: 'It failed every attempt at the simplest test in our simulator.',
    next: 'Useful pretrained weights, wrong world: it had to learn our scene.',
    clips: [],
  },
  {
    id: 'pi05-single', kind: 'main', depth: 0, when: 'Oct 15',
    name: 'Switch to π0.5’s pretrained brain, one arm',
    score: { value: 26, of: 30, label: 'one arm' },
    changed: 'We kept our data and the π0.7 way of prompting, but started from π0.5, a model pretrained on more than 10,000 hours of real robot data.',
    result: 'The same simple task went from almost never to 26 of 30.',
    next: 'Vision worked. Time for both arms and harder tasks.',
    clips: [],
  },
  {
    id: 'two-arms', kind: 'main', depth: 0, when: 'Oct 16',
    name: 'Two arms, four tasks',
    score: { value: 83, of: 120 },
    changed: 'One model for both arms and four tasks of rising difficulty, trained for 20,000 steps on 4,800 demonstrations.',
    result: 'It improved as training went on: 75, then 76, then 83 of 120 at steps 13,000, 17,000 and 20,000. Handing the cube between arms and coordinating two cubes stayed the hard part.',
    next: 'The arms often got stuck in situations the demonstrations never showed.',
    clips: [],
  },
  {
    id: 'soup', kind: 'dead', depth: 1, when: 'Oct 16',
    name: 'Average several checkpoints',
    score: { value: 74, of: 120 },
    changed: 'Checkpoints were good at different tasks, so we averaged six of them into one model.',
    result: 'No better than any single checkpoint.',
    next: 'Dropped.',
    clips: [],
  },
  {
    id: 'recover', kind: 'dead', depth: 1, when: 'Oct 16',
    name: 'Practice recovering from staged mistakes',
    score: { value: 72, of: 120 },
    changed: 'We added demonstrations with deliberate fumbles (knocked cubes, missed grasps) followed by recoveries.',
    result: 'Worse than without them.',
    next: 'Staged mistakes are not the mistakes the model actually makes.',
    clips: [],
  },
  {
    id: 'restart', kind: 'dead', depth: 1, when: 'Oct 16',
    name: 'A second run from a different start',
    score: { value: 52, of: 120 },
    changed: 'A separate two-arm run warm-started from the one-arm model, with the recovery data mixed in.',
    result: 'Well behind the main run.',
    next: 'Dropped.',
    clips: [],
  },
  {
    id: 'dagger', kind: 'main', depth: 0, when: 'Oct 16–17',
    name: 'Learning from its own mistakes',
    score: { value: 86, of: 120 },
    changed: 'We let the model run, and when it got stuck the scripted teacher took over from exactly that moment. Those rescues became new training data (a method called DAgger), in two rounds.',
    result: '85, then 86 of 120. The second arm in the two-cube task improved most, from 12 to 18 of 30.',
    next: 'Every rescue round gave a few points. Next: train longer and more gently.',
    clips: [],
  },
  {
    id: 'gentle', kind: 'main', depth: 0, when: 'Oct 17',
    name: 'Longer, gentler training',
    score: { value: 104, of: 120, label: 'best checkpoint' },
    changed: 'The same model and data, trained 6,000 more steps with smaller updates. Run twice with different random seeds.',
    result: 'The best checkpoint succeeds 104 times in 120; across eight checkpoints of both runs the average is 97, against 90 before.',
    next: 'This is the baseline every new idea is measured against.',
    clips: [],
  },
  {
    id: 'rtc', kind: 'running', depth: 1, when: 'Oct 17',
    name: 'Acting while thinking',
    score: { value: 88, of: 120, label: 'average of three checkpoints' },
    changed: 'A real robot keeps moving while the model computes its next move. This run trains the model to continue smoothly from actions it already committed to.',
    result: 'With no delay it scores 88 on average, below the baseline. With a 0.12-second delay it scores 82, the same as a model trained without it. Still running.',
    next: 'Measuring the last checkpoints and a second seed.',
    clips: [],
  },
  {
    id: 'memory', kind: 'running', depth: 1, when: 'Oct 17',
    name: 'Memory of the last few seconds',
    score: { value: 94, of: 120, label: 'first checkpoint' },
    changed: 'Instead of only the current camera frame, the model also sees frames from the last five seconds.',
    result: '94 at its first measured checkpoint. Still training.',
    next: 'Waiting for more checkpoints and a second seed.',
    clips: [],
  },
  {
    id: 'goals', kind: 'running', depth: 1, when: 'Oct 17',
    name: 'Pictures of the goal',
    score: { value: 96, of: 120, label: 'without a goal picture' },
    changed: 'The model also learns from a picture of what the scene should look like two seconds ahead.',
    result: '87, then 96 when run without a goal picture. Its score with goal pictures is being measured.',
    next: 'If perfect goal pictures help, a model that draws them is worth building.',
    clips: [],
  },
  {
    id: 'insulation', kind: 'dead', depth: 1, when: 'Oct 17',
    name: 'Protecting the vision part',
    score: { value: 2, of: 120 },
    changed: 'π0.7 trains its vision-and-language part separately from the part that moves the arms. We tried that here.',
    result: 'It collapsed: the arm reaches the cube but never lifts it.',
    next: 'A gentler version, with the vision part changing ten times more slowly, is running.',
    clips: [],
  },
  {
    id: 'data-loop', kind: 'running', depth: 1, when: 'Oct 17',
    name: 'More rescues on the hard tasks',
    changed: 'A third round of teacher rescues from the current models, aimed at the moment both hard tasks fail: an arm hovering while holding the cube.',
    result: 'About 1,000 new rescue episodes so far. Their effect is being tested against the baseline.',
    next: 'Kept only if it beats the baseline in two separate runs.',
    clips: [],
  },
];

// Clips attach to nodes here, so a new recording is one line.
const add = (id, ...clips) => nodes.find(n => n.id === id).clips.push(...clips);
add('gentle',
  clip('pick_place_any_c1_5000_s4000', 'pick_place_any', 1, true, 3.5, 29),
  clip('bin_c1_5000_s4001', 'bin', 2, true, 9.0, 30),
  clip('handover_c1_5000_s4002', 'handover', 3, true, 12.0, 24),
  clip('two_bins_c1_5000_s4003', 'two_bins', 4, false, 30.0, 21));
