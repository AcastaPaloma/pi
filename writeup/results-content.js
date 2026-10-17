// Results tab: rollout clips of trained checkpoints. Append a clip to `clips` to publish it; nothing else changes.
// Videos are filmed by scripts/record_showcase.py (one episode, fixed 3D viewpoint, same settings as the 30-seed test).

export const models = {
  best: {
    name: 'Best model so far',
    run: 'x_control_s1, step 5,000',
    about: 'The previous best, trained for 5,000 more steps on the same demonstrations at a low learning rate.',
    score: 104,
  },
  previous: {
    name: 'Previous best',
    run: 'pi05_dagger2, step 7,000',
    about: 'The model at the end of the first training round, before this round of experiments.',
    score: 90,
  },
};

export const tasks = {
  pick_place_any: { name: 'Pick and place', setup: 'Cube placed at random', goal: 'Put the cube on the round target.' },
  bin: { name: 'Into the bin', setup: 'Cube and bin placed at random', goal: 'Pick up the cube and drop it in the bin.' },
  handover: { name: 'Handover', setup: 'Cube placed at random', goal: 'One arm passes the cube to the other, which puts it in the bin.' },
  two_bins: { name: 'Two cubes, two arms', setup: 'Both cubes placed at random', goal: 'Each arm puts its own cube in the shared bin.' },
};

// rate: successes out of 30 random layouts for that model and task (the 30-seed test).
export const clips = [
  { id: 'pick-best', task: 'pick_place_any', model: 'best', layout: 1, success: true, seconds: 3.5, rate: 29,
    video: 'assets/results/pick_place_any_c1_5000_s4000.mp4', poster: 'assets/results/pick_place_any_c1_5000_s4000.jpg' },
  { id: 'handover-best', task: 'handover', model: 'best', layout: 3, success: true, seconds: 12.0, rate: 24,
    video: 'assets/results/handover_c1_5000_s4002.mp4', poster: 'assets/results/handover_c1_5000_s4002.jpg' },
];
