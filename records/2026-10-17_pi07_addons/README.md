# Records: π0.7 add-ons on the π0.5 backbone (2026-10-17 to 18)

Raw records of the second training round, copied from the three machines. The narrative, hypotheses and verdicts
are in `docs/RESEARCH_LOG.md`; the queue with every hypothesis is `experiments/queue.json`; the live summary with
videos, loss curves and the experiment tree is https://acastapaloma.github.io/pi/results.html.

| folder | machine | contents |
| --- | --- | --- |
| `main_2xH100/` | 192.222.52.191 (2x H100) | `training/x_*/`: each run's `config.json`, `run.json` (arguments incl. the resume checkpoint) and `log.jsonl` (loss every 20 steps); `evals/c30_<run>_<step>[_goal]_d<delay>_<task>/results.json`: 30-seed scores per task; `evals/c30_summary.txt`: one line per scored checkpoint |
| `node_4xH100/` | 192.222.55.167 (4x H100) | the same for the runs trained there |
| `a10/` | 193.122.189.119 (A10) | `evals/robust_*`: scores with visual randomization; `evals/failmine_*`: failure mining on non-test seeds (5000+); `logs/`: DAgger data collection and evaluation logs |
| `scheduler.log` | the Mac | every scheduler launch |

Scoring: strict success (task done and the object actually lifted), seeds 4000-4029 per task, oracle subtask
prompts, 25 actions per chunk, delay 0 unless the folder name says `d6` or `d12`, OSMesa rendering. Checkpoints and
the new DAgger datasets are on Hugging Face (`KuanKuanKuan/so101-pi07-sim-models`, `KuanKuanKuan/so101-pi07-sim-data`,
under `round2/`).
