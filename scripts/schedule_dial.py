#!/usr/bin/env python3
"""Run one phase of the two-player stochastic CPR over the pod's GPUs, one process per GPU (docs/RUNBOOK_STOCHASTIC_CPR.md).

    python scripts/schedule_dial.py {smoke,gate,train,optional,evaluate} [--length 100|200] [--gpus 0,1,2,3,4] [--dry-run] [--only GLOB] [--redo]

`--length` is the training length the G3 pilot set (scripts/evaluate_dial.py --gate); it picks the epochs, the seed
plan and the adapters the evaluation arms read.

A job is one seed of one config, started through scripts/launch_dial.sh with WAIT=1 as soon as a GPU is free. A phase
is a list of groups run in order (the second group reads the first group's adapters); within a group, jobs wait only
for GPUs. A job is skipped if its records exist (so a phase can be rerun after a pod restart; --redo runs it anyway)
or if its pid file names a live process. A GPU with a compute process the scheduler did not start is left alone.
--only keeps jobs whose run folder or "<folder> seed <k>" label matches the glob.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_dial_configs import ADDITIONAL, DECISIVE, EVAL_SHAPERS, EXTRA_SEEDS_LONG, GATE_EPOCHS, GATE_RUNS, OPTIONAL, PILOT_RUNS, SEEDS, SEEDS_LONG  # noqa: E402

DIAL = "checkpoints/dial"
LAUNCH_VARS = ("NAME", "SEED", "SMOKE", "SUFFIX", "OUT_SUFFIX", "EPOCHS", "CKPT_FREQ", "WAIT", "REPLAY_WINDOW",
               "PARTNER_ADAPTER", "LEARNER_ADAPTER", "REPLAY_RECORDS", "PARTNER_ADAPTER_TEMPLATE",
               "LEARNER_ADAPTER_TEMPLATE", "REPLAY_RECORDS_TEMPLATE")


@dataclass(frozen=True)
class Job:
    name: str  # config stem in configs/dial
    seed: int
    env: tuple = ()  # sorted (VAR, value) pairs for launch_dial.sh

    @property
    def vars(self) -> dict:
        return dict(self.env)

    @property
    def folder(self) -> str:
        return f"{DIAL}/{self.name}{self.vars.get('SUFFIX', '')}{'_smoke' if self.vars.get('SMOKE') == '1' else ''}"

    @property
    def label(self) -> str:
        return f"{Path(self.folder).name} seed {self.seed}"

    def done(self, root: Path = None) -> bool:
        return ((root or ROOT) / self.folder / f"exp{self.seed + 1}_cpr_records").exists()

    def alive(self, root: Path = None) -> bool:
        pid_file = (root or ROOT) / DIAL / "logs" / f"{Path(self.folder).name}_s{self.seed}.pid"
        try:
            os.kill(int(pid_file.read_text()), 0)
            return True
        except (OSError, ValueError):
            return False


def job(name: str, seed: int, **env) -> Job:
    return Job(name, seed, tuple(sorted((k, str(v)) for k, v in env.items())))


def evaluation(stage: str, arm: str, seed: int, smoke: bool = False, length: int = 100) -> list:
    """E1-E4 for one seed of one evaluated shaper, reading that seed's final adapters (smoke: epoch 2). E1 and E2 run
    100 epochs whatever the training length, E3 and E4 run 20."""
    source, ckpt = f"{DIAL}/{stage}_{arm}{'_smoke' if smoke else ''}/exp%d_", 2 if smoke else length
    long, short = (dict(SMOKE=1), dict(SMOKE=1)) if smoke else (dict(EPOCHS=100, CKPT_FREQ=100), dict(EPOCHS=20, CKPT_FREQ=0))
    partner = dict(PARTNER_ADAPTER_TEMPLATE=f"{source}model2_model_checkpoint_{ckpt}")
    return [
        job(f"{stage}_e1_transfer_{arm}", seed, **partner, **long),
        job(f"{stage}_e2_replay_{arm}", seed, REPLAY_RECORDS_TEMPLATE=f"{source}cpr_records", **long,
            **(dict(REPLAY_WINDOW=2) if smoke else {})),
        job(f"{stage}_e3_frozen_partner_{arm}", seed, **partner, **short,
            LEARNER_ADAPTER_TEMPLATE=f"{source}model1_model_checkpoint_{ckpt}"),
        job(f"{stage}_e4_untrained_partner_{arm}", seed, **partner, **short),
    ]


def probe(folder: str, seed: int, smoke: bool = False, ckpt: int = 100) -> Job:
    """The scripted probe against agent 1 of `folder`, frozen at epoch `ckpt` (smoke: epoch 2)."""
    tag, ckpt = ("_smoke", 2) if smoke else ("", ckpt)
    return job(f"{folder[:2]}_probe", seed, SUFFIX=f"_of_{folder}", **(dict(SMOKE=1) if smoke else dict(EPOCHS=20, CKPT_FREQ=0)),
               LEARNER_ADAPTER_TEMPLATE=f"{DIAL}/{folder}{tag}/exp%d_model1_model_checkpoint_{ckpt}")


def phases(length: int = 100) -> dict:
    """Jobs per phase for a training length of 100 or 200 epochs."""
    seeds = SEEDS if length == 100 else SEEDS_LONG
    shapers = [(stage, arm) for stage, arms in EVAL_SHAPERS.items() for arm in arms]
    by_seed = lambda counts: [(name, seed) for seed in range(max(counts.values()))  # seed order: decisive arms first, additional last
                              for name in sorted(counts, key=lambda n: (DECISIVE.index(n) if n in DECISIVE else len(DECISIVE), n in ADDITIONAL, -counts[n]))
                              if seed < counts[name]]
    gate = [job(name, seed, EPOCHS=GATE_EPOCHS[name], CKPT_FREQ=0, **(dict(SUFFIX=suffix) if suffix else {}))
            for name, (suffix, n) in {**GATE_RUNS, **PILOT_RUNS}.items() for seed in range(n)]
    gate.sort(key=lambda j: "SUFFIX" not in j.vars)  # the two-learner G3 pilot is the longest run: start it first
    train = [job(name, seed, EPOCHS=length, CKPT_FREQ=length) for name, seed in by_seed(seeds)]
    optional = [job(name, seed, EPOCHS=length, CKPT_FREQ=length) for name, seed in by_seed(OPTIONAL)]
    evals = [j for stage, arm in shapers for seed in range(seeds[f"{stage}_{arm}"]) for j in evaluation(stage, arm, seed, length=length)]
    transferred = [f"{stage}_{kind}_{arm}" for stage, arm in shapers for kind in ("e1_transfer", "e2_replay")]
    smoke_first = [job(name, 0, SMOKE=1) for name in [*SEEDS, *OPTIONAL, *(n for n in GATE_RUNS if n not in SEEDS)]]
    smoke_second = [j for stage, arm in shapers for j in evaluation(stage, arm, 0, smoke=True) + [probe(f"{stage}_{arm}", 0, smoke=True)]]
    return {
        "smoke": [smoke_first, smoke_second],
        "gate": [gate],
        "train": [train],
        "optional": [optional, [probe(name, seed, ckpt=length) for name, seed in by_seed(OPTIONAL)]],
        "evaluate": [sorted(evals, key=lambda j: -int(j.vars["EPOCHS"])) + [probe(name, seed, ckpt=length) for name, seed in by_seed(seeds)],
                     [probe(folder, seed) for folder in transferred for seed in range(planned_seeds(folder, seeds))]],
        # 200-epoch training only, if time remains after the planned runs: ShapeLLM-style seeds 3 and 4, then their probes
        "extra": [[job(name, seed, EPOCHS=length, CKPT_FREQ=length) for name, extra in EXTRA_SEEDS_LONG.items() for seed in extra],
                  [probe(name, seed, ckpt=length) for name, extra in EXTRA_SEEDS_LONG.items() for seed in extra]] if length == 200 else [[], []],
    }


def planned_seeds(folder: str, seeds: dict = SEEDS) -> int:
    """Seeds of an E1 or E2 folder: those of the shaper it evaluates."""
    stage = folder[:2]
    return next(seeds[f"{stage}_{arm}"] for arm in EVAL_SHAPERS[stage] if folder.endswith(f"_{arm}"))


def busy_gpus() -> set:
    """Indices of GPUs running any compute process; empty when nvidia-smi is unavailable."""
    def query(what):
        return subprocess.run(["nvidia-smi", f"--query-{what}", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.splitlines()
    try:
        index = {uuid.strip(): int(i) for i, uuid in (line.split(",") for line in query("gpu=index,uuid") if line.strip())}
        return {index[u.strip()] for u in query("compute-apps=gpu_uuid") if u.strip() in index}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return set()


def run(groups: list, gpus: list, dry_run: bool = False, redo: bool = False, only: str = None, poll: float = 20) -> list:
    failed = []
    base_env = {k: v for k, v in os.environ.items() if k not in LAUNCH_VARS}
    for g, group in enumerate(groups, 1):
        jobs = [j for j in group if only is None or fnmatch.fnmatch(Path(j.folder).name, only) or fnmatch.fnmatch(j.label, only)]
        todo = [j for j in jobs if redo or not j.done()]
        print(f"group {g}: {len(todo)} job(s) to run, {len(jobs) - len(todo)} already done")
        if dry_run:
            for j in todo:
                print(f"  {j.label}: " + " ".join(f"{k}={v}" for k, v in j.env))
            continue
        running = {}
        while todo or running:
            for gpu, (j, proc, start) in list(running.items()):
                if proc.poll() is not None:
                    del running[gpu]
                    ok = proc.returncode == 0 and j.done()
                    failed += [] if ok else [j]
                    print(f"{time.strftime('%H:%M')} GPU {gpu}: {j.label} {'done' if ok else 'FAILED'} "
                          f"after {(time.time() - start) / 60:.0f} min", flush=True)
            busy = busy_gpus()
            for gpu in gpus:
                while todo and gpu not in running and gpu not in busy:
                    j = todo.pop(0)
                    if (j.done() and not redo) or j.alive():
                        print(f"  skipping {j.label}: {'done' if j.done() else 'already running'}")
                        continue
                    proc = subprocess.Popen(["./scripts/launch_dial.sh", j.name, str(j.seed), str(gpu)], cwd=ROOT,
                                            env={**base_env, **j.vars, "WAIT": "1"}, start_new_session=True)
                    running[gpu] = (j, proc, time.time())
                    print(f"{time.strftime('%H:%M')} GPU {gpu}: {j.label} started", flush=True)
            if todo or running:
                time.sleep(poll)
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=sorted(phases()))
    ap.add_argument("--length", type=int, choices=(100, 200), default=100)
    ap.add_argument("--gpus", default="0,1,2,3,4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    if not a.dry_run:
        missing = [name for name in ("cpr_learner_r2", "cpr_shaper_r2") if not (ROOT / "adapter" / name).is_dir()]
        if missing:  # concurrent first launches would race to create them
            print("create the initial adapters first:\n" + "\n".join(
                f"  python init_lora_adapters.py --model_path google/gemma-2-2b-it --output_dir adapter/{m}" for m in missing))
            return 1
    failed = run(phases(a.length)[a.phase], [int(x) for x in a.gpus.split(",")], a.dry_run, a.redo, a.only)
    for j in failed:
        print(f"FAILED {j.label}: {DIAL}/logs/{Path(j.folder).name}_s{j.seed}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
