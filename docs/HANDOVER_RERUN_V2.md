# Handover: rerun of the chained-credit shaper arms with the estimator fix (21 September 2026)

For the pipeline agent on a fresh pod. Read `review_round5.md` section 2 for why. In one sentence: the executed chained
estimator bootstrapped the value at an episode's last live step from the next episode's first round (a fresh pool),
which inverts the restraint signal wherever episodes end by collapse; the fix zeroes that bootstrap and keeps the
lambda-chain of later advantages, so cross-episode credit is preserved. Every arm whose agent 2 used chained credit is
rerun; the controls and the split-credit arm are untouched.

## What is rerun

Training (19 runs, 100 epochs): `m2_shaper_matched` (5 seeds), `m2_shaper_slow` (3), `m2_shapellm` (5),
`m2_shapellm_history_off` (3), `m3_shapellm` (3).
Then their evaluation: E1–E4 and probes for `m2_shapellm`, `m2_shaper_matched`, `m3_shapellm`; probes of
`m2_shaper_slow` and `m2_shapellm_history_off`; probes of every E1/E2 learner. The E1 controls of 21 September
(`m2_e1_transfer_tbn_matched`, `m2_e1_transfer_naive`) run in the same phase since their records do not exist yet.

Not rerun: `m2_naive`, `m2_slow`, `m2_tbn_matched`, `m2_tbn_slow`, `m2_shaper_matched_split`, `m3_naive`, `m3_slow`,
the gates, and every evaluation run that reads only those (the split arm's E1–E4 and probes, the control probes).

## Before anything runs (on the pod)

```
git pull && git log --oneline -1                       # must include "terminal values at episode ends"
python -m pytest tests -q                              # all pass on trl 0.11.4
python scripts/make_dial_configs.py && git status --short configs/dial     # prints nothing
grep -l '"episode_terminal_values": true' configs/dial/m2_shapellm.json configs/dial/m2_shaper_matched.json \
  configs/dial/m2_shaper_slow.json configs/dial/m2_shapellm_history_off.json configs/dial/m3_shapellm.json | wc -l   # 5
```

## Move the estimator-v1 runs aside (do not delete)

The scheduler skips a run whose records exist, so the affected folders must move. They are the record of the bug's
effect and the thesis reports them.

```
mkdir -p checkpoints/dial_v1 results/dial_v1
cp -r results/dial/* results/dial_v1/
for f in m2_shaper_matched m2_shaper_slow m2_shapellm m2_shapellm_history_off m3_shapellm \
  m2_e1_transfer_shapellm m2_e1_transfer_shaper_matched m2_e2_replay_shapellm m2_e2_replay_shaper_matched \
  m2_e3_frozen_partner_shapellm m2_e3_frozen_partner_shaper_matched m2_e4_untrained_partner_shapellm \
  m2_e4_untrained_partner_shaper_matched m2_probe_of_m2_e1_transfer_shapellm m2_probe_of_m2_e1_transfer_shaper_matched \
  m2_probe_of_m2_e2_replay_shapellm m2_probe_of_m2_e2_replay_shaper_matched m2_probe_of_m2_shapellm \
  m2_probe_of_m2_shapellm_history_off m2_probe_of_m2_shaper_matched m2_probe_of_m2_shaper_slow \
  m3_e1_transfer_shapellm m3_e2_replay_shapellm m3_e3_frozen_partner_shapellm m3_e4_untrained_partner_shapellm \
  m3_probe_of_m3_e1_transfer_shapellm m3_probe_of_m3_e2_replay_shapellm m3_probe_of_m3_shapellm; do
  [ -d checkpoints/dial/$f ] && mv checkpoints/dial/$f checkpoints/dial_v1/$f
done
python scripts/schedule_dial.py train --dry-run       # must list exactly 19 jobs
```

## Smoke, then train

```
SMOKE=1 WAIT=1 ./scripts/launch_dial.sh m2_shapellm 0 0
grep -c "chained credit with terminal values at" checkpoints/dial/logs/m2_shapellm_smoke_s0.log   # >= 1 (one line per trial)
grep -L "completed" checkpoints/dial/logs/m2_shapellm_smoke_s0.log                                # prints nothing
rm -r checkpoints/dial/m2_shapellm_smoke
python scripts/schedule_dial.py train                 # 19 runs, about 3.2 h each; five GPUs: about 13 h
```

**Early check, logged, at epoch 30 of the first wave (about 1 h in).** Under the v1 estimator every chained seed sat at
survival 0.00 with agent 2's restraint falling from epoch 1. Read the per-epoch survival from the records of any
finished-30-epoch seed, or from the console's epoch summaries. If every running chained seed still shows survival
below 0.05 at epoch 30 with restraint below its epoch-1 level, stop the phase and report; the fix did not change the
dynamics and the thesis reports that. Otherwise let the phase finish. This check is a stop rule, not a gate: it cannot
make a result look better, only save a day.

## Evaluate, decide, copy off

```
python scripts/schedule_dial.py evaluate              # only the missing runs: about 9 h on five GPUs
python scripts/evaluate_dial.py --decide --length 100 --out results/dial     # regenerates decision.json
python scripts/dial_results_tables.py && python scripts/dial_figures.py      # thesis tables and figures
python scripts/dial_credit_replay.py --folder checkpoints/dial/m2_shapellm > results/dial/credit_replay_v2.txt
```

Copy `checkpoints/dial`, `checkpoints/dial_v1` (records and logs at least), and `results/` off the pod before
terminating it. Commit `results/dial` and `results/dial_v1` with the decision files.

## Budget and clock

Training 19 x 3.2 h = 61 GPU-h; evaluation about 45 GPU-h. On five L40S: about 13 h + 9 h. On seven: about 9 h + 7 h.
Launch by 22 September 02:00 UTC for results by the evening of the 22nd on seven GPUs, the morning of the 23rd on five.
