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

## Gate: G1, G2 and the 200-epoch G3 pilot

```
python scripts/schedule_dial.py gate            # G1 and G2: 3 seeds x 100 epochs each; G3: 1 seed x 200 epochs, started first (about 6.4 h)
python scripts/check_seed_divergence.py checkpoints/dial/g1_m3_harvest checkpoints/dial/g2_m2_tft    # once epoch 1 is written
python scripts/evaluate_dial.py --gate          # GO or NO GO, and the training length; writes results/dial/gate.json
```

**G3.** The pilot passes when the shaper's restraint share is at least 0.3 and pool survival is at least 0.5:
- over epochs 81–100, which sets training to 100 epochs;
- otherwise over epochs 181–200, which sets training to 200 epochs.

If neither window passes, the verdict is NO GO.

**After the verdict.** Log it, with the per-seed numbers and the training length, in the pre-registration's amendment log, and commit it with `results/dial/gate.json`. On NO GO, follow the amendment: the null result is reported, and no further levers are tried.

## Training

With `LENGTH` set to the gate's training length (100 or 200):

```
python scripts/schedule_dial.py train --length $LENGTH
python scripts/check_seed_divergence.py checkpoints/dial/m2_naive    # once seed 1 has written its epoch-1 openings
```

**Size.** At 100 epochs: 38 runs, about 122 GPU-hours, about a day on five GPUs. At 200 epochs: 34 runs, since ShapeLLM-style and shaper-matched keep five seeds and every other arm runs three, for about 218 GPU-hours, about two days.

**Order.** Jobs run in seed order, with the arms that decide S leading each seed.

**Optional arm.** If time allows, run `python scripts/schedule_dial.py optional --length $LENGTH`.

## Evaluation arms and probes

```
python scripts/schedule_dial.py evaluate --length $LENGTH
```

The phase runs in two groups:
1. **E1–E4 and the training-arm probes.**
   - E1 and E2: 100 epochs, checkpointed at 100.
   - E3 and E4: 20 epochs, reading the training adapters at epoch `LENGTH`.
   - A 20-epoch probe of every training arm's agent 1. The probe restrains or harvests, each with probability 1/2.
2. **Probes of the E1 and E2 learners.**

## Read-out

```
python scripts/evaluate_dial.py --decide --length $LENGTH     # every condition of S, T and C, and the verdict; writes results/dial/decision.json
python scripts/evaluate_dial.py checkpoints/dial/m2_* checkpoints/dial/m3_* --out results/dial    # per-run tables with Wilson intervals
```

Copy `checkpoints/dial` off the pod before terminating it: records, logs, tables and adapters.
