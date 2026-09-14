# Runbook: two-player stochastic CPR (16–22 September 2026)

This runbook is the operational companion to `docs/PREREGISTRATION_STOCHASTIC_CPR.md`. Where the two disagree, the pre-registration wins.

Every phase runs through `scripts/schedule_dial.py`. The scheduler:
- starts **one process per GPU** as GPUs free up;
- runs the jobs that read another group's adapters only after that group has finished;
- skips a run whose records already exist, so a phase can be restarted after a pod restart;
- leaves alone any GPU that is running a process it did not start;
- prints the log path of every failed run.

Add `--dry-run` to list the jobs without starting them. To re-run one job, add `--only`, for example `--only "m2_naive seed 3"` or `--only "m2_probe_of_*"`. While a phase is running, do not launch runs by hand. Single runs can still be launched with `scripts/launch_dial.sh`; its header has examples.

## 16 September: setup, G0, smoke

```
git pull && git log --oneline -1
pip install -r requirements.txt                         # trl 0.11.4; needs a Hugging Face token for google/gemma-2-2b-it
python scripts/make_dial_configs.py && git status --short configs/dial    # prints nothing if the configs match the commit
python -m pytest tests -q
for a in cpr_learner_r2 cpr_shaper_r2; do [ -d adapter/$a ] || python init_lora_adapters.py --model_path google/gemma-2-2b-it --output_dir adapter/$a; done
mkdir -p results/dial && python scripts/cpr_preflight.py --dial | tee results/dial/g0_preflight.txt    # G0: one run covers both regimes
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -l 10 > results/dial/smoke_memory.csv &
python scripts/schedule_dial.py smoke
kill %1
python scripts/evaluate_dial.py checkpoints/dial/*_smoke --window 2 --out results/dial/smoke
```

The smoke phase runs every config for 2 epochs: the 10 training arms, the optional arm, G1, G2 and the forced-108 memory run. It then runs E1–E4 and a probe for each evaluated shaper, from the smoke adapters.

Checks before moving on:
- The scheduler reports no failures.
- `grep -L "completed" checkpoints/dial/logs/*_smoke_s0.log` prints nothing.
- `checkpoints/dial/logs/smoke_forced108_m2_shapellm_smoke_s0.log` contains no "out of memory". The peak in `results/dial/smoke_memory.csv` is well below the card's memory: a forced-108 trial holds 1,620 positions, three times the pond's.
- The evaluator prints a table for every smoke folder.

Then delete the smoke folders: `rm -r checkpoints/dial/*_smoke`.

## 17 September: G1, G2, G3

```
python scripts/schedule_dial.py gate            # 7 runs; the G3 pilot (two learners, about 1.4 h) starts first
python scripts/check_seed_divergence.py checkpoints/dial/g1_m3_harvest checkpoints/dial/g2_m2_tft    # once epoch 1 is written
python scripts/evaluate_dial.py --gate          # prints GO or NO GO; writes results/dial/gate.json
```

Log the verdict and the per-seed numbers in the pre-registration's amendment log, then commit it with `results/dial/gate.json`. On NO GO, follow Section 9: no training starts.

## 18–20 September: training

```
python scripts/schedule_dial.py train           # 38 runs of about 1.4 h each, so about 11 h on five GPUs
python scripts/check_seed_divergence.py checkpoints/dial/m2_naive    # once seed 1 has written its epoch-1 openings
```

Jobs go in seed order: seed 0 of every arm first, and the four arms that decide S lead each seed. A late finish therefore costs seeds evenly rather than whole arms. If training finishes by 20 September 18:00, run the optional arm with `python scripts/schedule_dial.py optional`.

## 21 September: evaluation arms and probes

```
python scripts/schedule_dial.py evaluate
```

The phase takes about 26 GPU-hours, roughly 5–6 h on five GPUs. It runs in two groups:
1. **E1–E4 and the training-arm probes.**
   - E1 and E2: 100 epochs, checkpointed at epoch 100 so their learners can be probed.
   - E3 and E4: 20 epochs.
   - A 20-epoch probe of every training arm's agent 1.
2. **Probes of the E1 and E2 learners.**

## 22 September: read-out

```
python scripts/evaluate_dial.py --decide        # Section 8: every condition of S, T and C, and the verdict; writes results/dial/decision.json
python scripts/evaluate_dial.py checkpoints/dial/m2_* checkpoints/dial/m3_* --out results/dial    # per-run tables with Wilson intervals
```

Copy `checkpoints/dial` off the pod before terminating it: records, logs, tables and adapters.
