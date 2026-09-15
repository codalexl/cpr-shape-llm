# Runbook: two-player stochastic CPR, amended design (from 16 September 2026)

This is the operational companion to `docs/PREREGISTRATION_STOCHASTIC_CPR.md`. Where the two disagree, the pre-registration wins, as amended on 15 September.

**The design point.**
- Takes 1–2, labelled `1` and `2`.
- A fixed 50-round horizon, with no round counter and no length stated in the prompt.
- Five episodes of five parallel games per trial.
- A full pool of 20.
- The dial at m = 2 or m = 3.

**The scheduler.** Every phase runs through `scripts/schedule_dial.py`, which:
- starts **one process per GPU** as GPUs free up;
- runs the jobs that read another group's adapters only after that group finishes;
- skips runs whose records already exist, so a phase can be restarted after a pod restart;
- leaves alone any GPU running a process it did not start;
- prints the log path of every failure.

Add `--dry-run` to list the jobs. To re-run one job, use `--only`, for example `--only "m2_naive seed 3"`. While a phase runs, do not launch jobs by hand.

**Timings.** These are estimates scaled from the pond. An epoch costs about 2.3 times the pond's (50 rounds × 5 games against 36 × 3): about 3.2 h per 100 epochs for a two-learner run and about 1.3 h for a single learner.

## Setup, G0 and smoke

```
git pull && git log --oneline -1
pip install -r requirements.txt                         # trl 0.11.4; needs a Hugging Face token for google/gemma-2-2b-it
python scripts/make_dial_configs.py && git status --short configs/dial    # prints nothing if the configs match the commit
python -m pytest tests -q
for a in cpr_learner_r2 cpr_shaper_r2; do [ -d adapter/$a ] || python init_lora_adapters.py --model_path google/gemma-2-2b-it --output_dir adapter/$a; done
mkdir -p results/dial && python scripts/cpr_preflight.py --dial | tee results/dial/g0_preflight_v2.txt   # G0: takes 1-2, amended rules, every history
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 10 > results/dial/smoke_memory_v2.csv &
python scripts/schedule_dial.py smoke
kill %1
python scripts/evaluate_dial.py checkpoints/dial/*_smoke --window 2 --out results/dial/smoke_v2
```

**G0 against the local screen.** Compare G0 with the local screen in `results/dial/prompt_screens.txt`, section (3), amended rules: 0.32 restraint in the first round and 0.05–0.38 after a history line (mean 0.16). If G0 on the pod falls outside 0.10–0.25 on average after a history line, stop and log it before the gate; the shaping window depends on it.

**Smoke checks.**
- The scheduler reports no failures.
- `grep -L "completed" checkpoints/dial/logs/*_smoke_s0.log` prints nothing.
- The peak in `smoke_memory_v2.csv` is well below the card's memory. A shaper trial holds 5 × 50 × 5 = 1,250 positions, below the 1,620 the forced-108 run of the first design survived.

Then delete the smoke folders: `rm -r checkpoints/dial/*_smoke`.

## Gate: the third G3 (G3a and G3b)

**Deadline.**
- The two pilots start on 15 September (UTC) or not at all.
- Training follows only if G3a has passed by 12:00 UTC on 16 September. A pass over epochs 81–100 can be read as soon as the pilot reaches epoch 100.
- The 181–200 window counts only if the pilot has completed it by 12:00 UTC.
- There is no extension and no fourth attempt.

**What runs.**
- **G1 and G2** passed on 15 September and are not re-run. The scheduler skips them because their records exist.
- **The G3a pilot** is the additional split-credit arm, in `checkpoints/dial/m2_shaper_matched_split_g3`.
- **The tbn-matched pilot** runs beside it in `m2_tbn_matched_g3tbn`. G3a's criterion is applied to it, but it does not gate.
- **The second G3** (chained GAE) stays in `m2_shaper_matched_g3`. G3b reads it too, so keep its records.

```
git pull && python -m pytest tests -q
python scripts/make_dial_configs.py && git status --short configs/dial    # prints nothing if the configs match the commit
SMOKE=1 WAIT=1 ./scripts/launch_dial.sh m2_shaper_matched_split 0 0       # 2 trials: split credit runs end to end
grep "split credit: baseline from 1 previous trial" checkpoints/dial/logs/m2_shaper_matched_split_smoke_s0.log && rm -r checkpoints/dial/m2_shaper_matched_split_smoke
python scripts/schedule_dial.py gate            # both pilots, 200 epochs each, in parallel (about 6.4 h)
python scripts/evaluate_dial.py --gate --out results/dial/gate_v3    # G3a with the training length, the tbn pilot, G3b, the reading
```

**Smoke check.**
- The log ends with "Experiment 1 completed." and shows no traceback.
- The split-credit line appears once per trial: from 0 previous trials in trial 1, then from 1 in trial 2, with a nonzero SD.

**G3a.** Agent 2's restraint share must be at least 0.3 and pool survival at least 0.5:
- over epochs 81–100, which sets training to 100 epochs;
- otherwise over epochs 181–200, which sets training to 200 epochs.

The tbn pilot's line applies the same test and does not gate.

**G3b** is reported and does not gate. For each pilot and for the second G3, it gives the correlation between agent 2's restraint in an episode and agent 1's change in restraint into the next episode, with its interval and the partial correlation. It reads channel, none or reversed.

**After the verdict.**
- Log it in the pre-registration's amendment log with:
  - the per-seed numbers and the training length;
  - the tbn pilot's G3a;
  - G3b for all three runs;
  - the printed reading, and the time it was read.
- Commit it with `results/dial/gate_v3/gate.json`.
- On NO GO, or if no pass has been read by 12:00 UTC on 16 September, no training starts. The pond is the thesis, and the dial is an exploratory chapter.

## Training

With `LENGTH` set to the gate's training length (100 or 200):

```
python scripts/schedule_dial.py train --length $LENGTH
python scripts/check_seed_divergence.py checkpoints/dial/m2_naive    # once seed 1 has written its epoch-1 openings
```

**Size.**
- At 100 epochs: 43 runs, the 38 pre-registered plus five seeds of the split arm, for about 138 GPU-hours, just over a day on five GPUs.
- At 200 epochs: 39 runs, for about 250 GPU-hours, about two days. Shaper-matched, tbn-matched and the split arm keep five seeds, and every other arm runs three.

**Order.** Jobs run in seed order. Shaper-matched and tbn-matched, the contrast that decides S, lead each seed, and the additional split arm closes it.

**Extra seeds (200 epochs only).** If time remains after the planned runs, run `python scripts/schedule_dial.py extra --length 200` before anything else. It trains ShapeLLM-style seeds 3 and 4, then probes them. They count toward S only if both finish.

**Optional arm.** If time still allows, run `python scripts/schedule_dial.py optional --length $LENGTH`.

## Evaluation arms and probes

```
python scripts/schedule_dial.py evaluate --length $LENGTH
```

The evaluated shapers are ShapeLLM-style and shaper-matched at m = 2, the additional split arm, and ShapeLLM-style at m = 3. The phase runs in two groups:
1. **E1–E4 and the training-arm probes.**
   - E1 and E2: 100 epochs, checkpointed at 100.
   - E3 and E4: 20 epochs, reading the training adapters at epoch `LENGTH`.
   - A 20-epoch probe of every training arm's agent 1. The probe restrains or harvests, each with probability 1/2.
2. **Probes of the E1 and E2 learners.**

## Read-out

```
python scripts/evaluate_dial.py --decide --length $LENGTH     # every condition of S, T and C, the verdict, and the split arm's own line; writes results/dial/decision.json
python scripts/evaluate_dial.py checkpoints/dial/m2_* checkpoints/dial/m3_* --out results/dial    # per-run tables with Wilson intervals
```

The verdict line reads only the pre-registered arms. The additional split arm's S or T is printed on its own line. Like every dial result, it is exploratory (amendment of 15 September, evening).

Copy `checkpoints/dial` off the pod before terminating it: records, logs, tables and adapters.
