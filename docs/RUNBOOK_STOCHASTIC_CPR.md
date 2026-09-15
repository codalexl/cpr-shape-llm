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

## Gate: the third G3 attempt, with repaired shaper credit

**What runs.**
- **G1 and G2** passed on 15 September and are not re-run. The scheduler skips them because their records exist.
- **The second G3** (whole-trial credit) stays in `checkpoints/dial/m2_shaper_matched_g3`.
- **The third G3** runs in `m2_shaper_matched_g3r`.
- **The tbn-matched reference pilot** runs beside it in `m2_tbn_matched_g3tbn`.

```
git pull && python -m pytest tests -q
python scripts/make_dial_configs.py && git status --short configs/dial    # prints nothing: every shaper config uses decomposed credit
SMOKE=1 WAIT=1 ./scripts/launch_dial.sh m2_shaper_matched 0 0             # 2 epochs: the repaired estimator runs end to end
grep "agent2 live A_raw" checkpoints/dial/logs/m2_shaper_matched_smoke_s0.log && rm -r checkpoints/dial/m2_shaper_matched_smoke
python scripts/schedule_dial.py gate            # G3 and the tbn reference, 200 epochs each, in parallel (about 6.4 h)
python scripts/evaluate_dial.py --gate --out results/dial/gate_v3    # verdict, training length, and the tbn reference line
```

**Smoke check.** The log ends with "Experiment 1 completed." and shows no traceback. The shaper's `live A_raw` line appears for both epochs.

**G3.** The criteria and length rule are unchanged. The shaper's restraint share must be at least 0.3 and pool survival at least 0.5:
- over epochs 81–100, which sets training to 100 epochs;
- otherwise over epochs 181–200, which sets training to 200 epochs.

The tbn reference line does not gate.

**After the verdict.**
- Log it in the pre-registration's amendment log with the per-seed numbers, the training length and the tbn reference.
- Commit it with `results/dial/gate_v3/gate.json`.
- On NO GO, follow item 5 of the deviation of 15 September: no training starts, and the null result reports all three attempts.

## Training

With `LENGTH` set to the gate's training length (100 or 200):

```
python scripts/schedule_dial.py train --length $LENGTH
python scripts/check_seed_divergence.py checkpoints/dial/m2_naive    # once seed 1 has written its epoch-1 openings
```

**Size.** At 100 epochs: 38 runs, about 122 GPU-hours, about a day on five GPUs. At 200 epochs: 34 runs, since shaper-matched and tbn-matched keep five seeds and every other arm runs three, for about 218 GPU-hours, about two days.

**Order.** Jobs run in seed order, with shaper-matched and tbn-matched, the contrast that decides S, leading each seed.

**Extra seeds (200 epochs only).** If time remains after the planned runs, run `python scripts/schedule_dial.py extra --length 200` before anything else. It trains ShapeLLM-style seeds 3 and 4, then probes them. They count toward S only if both finish.

**Optional arm.** If time still allows, run `python scripts/schedule_dial.py optional --length $LENGTH`.

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
