"""Evaluation of the executed grid against docs/EXPERIMENT_PLAN.md.

Pure functions over `cpr_records` (one dict of equal-length lists per run) so
that every estimator is unit-tested against simulated policies of known value
before it is applied to a training run. The CLI is scripts/evaluate_grid.py.

Definitions follow the thesis: return G_i (eq. return), survival S (eq. collapse),
leave-2 L_i and stock-conditioned frequency pi_i(a|R) (eq. estimators), Wilson
interval (eq. wilson), paired per-seed difference Delta_s (eq. paired).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- records


def load_records(path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class Episode:
    epoch: int
    episode: int
    game: int
    open: Tuple[int, int]
    ret: Tuple[int, int]
    collapse: Optional[int]  # 1-indexed round at which the pool emptied; None = survived

    @property
    def survived(self) -> bool:
        return self.collapse is None


def episodes(rec: dict) -> List[Episode]:
    """One Episode per (epoch, episode, game), from the step rows."""
    n = len(rec["epoch"])
    acc: Dict[Tuple[int, int, int], dict] = {}
    for i in range(n):
        key = (int(rec["epoch"][i]), int(rec["episode"][i]), int(rec["game"][i]))
        g = acc.setdefault(key, {"r1": 0, "r2": 0, "collapse": None, "open": None})
        if int(rec["step"][i]) == 1:
            g["open"] = (int(rec["request_1"][i]), int(rec["request_2"][i]))
        g["r1"] += int(rec["reward_1"][i])
        g["r2"] += int(rec["reward_2"][i])
        if rec["depleted"][i] and int(rec["R_start"][i]) > 0 and g["collapse"] is None:
            g["collapse"] = int(rec["step"][i])
    return [Episode(k[0], k[1], k[2], g["open"], (g["r1"], g["r2"]), g["collapse"])
            for k, g in sorted(acc.items())]


def n_epochs(rec: dict) -> int:
    return int(max(rec["epoch"])) + 1


# ----------------------------------------------------------------------------- intervals


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for k successes out of n (thesis eq. wilson)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (float(xs[0]), float("nan"))
    return (float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))))


# ----------------------------------------------------------------------------- per-window statistics


@dataclass
class WindowStats:
    epochs: List[int]
    n_games: int
    ret: Tuple[Tuple[float, float], Tuple[float, float]]     # (mean, se) per agent
    survival: Tuple[int, int]                                # k, n
    leave2: Tuple[Tuple[int, int], Tuple[int, int]]          # (k, n) per agent, openings
    leave2_low: Tuple[Tuple[int, int], Tuple[int, int]]      # (k, n) per agent, unmasked steps with 0 < R < low_R
    surv_given_leave2: Tuple[Tuple[int, int], Tuple[int, int]]  # (k, n) per agent: survived | opened != 2
    joint: Tuple[float, float]                               # mean joint return, mean |G1 - G2|

    def wilson(self, name: str, agent: int = 0):
        k, n = getattr(self, name) if name == "survival" else getattr(self, name)[agent]
        return wilson(k, n)


def window_stats(rec: dict, epochs: Iterable[int], low_R: int = 12) -> WindowStats:
    epochs = sorted(set(int(e) for e in epochs))
    eps = [e for e in episodes(rec) if e.epoch in epochs]
    keep = set(epochs)
    ret = tuple(mean_se([e.ret[i] for e in eps]) for i in (0, 1))
    surv = (sum(e.survived for e in eps), len(eps))
    leave2 = tuple((sum(e.open[i] != 2 for e in eps), len(eps)) for i in (0, 1))
    sgl = tuple((sum(e.survived for e in eps if e.open[i] != 2), sum(e.open[i] != 2 for e in eps)) for i in (0, 1))
    low = [[0, 0], [0, 0]]
    for i in range(len(rec["epoch"])):
        if int(rec["epoch"][i]) not in keep or rec["masked"][i]:
            continue
        R = int(rec["R_start"][i])
        if not (0 < R < low_R):
            continue
        for a, key in ((0, "request_1"), (1, "request_2")):
            low[a][1] += 1
            if int(rec[key][i]) != 2:
                low[a][0] += 1
    joint = (float(np.mean([e.ret[0] + e.ret[1] for e in eps])) if eps else float("nan"),
             float(np.mean([abs(e.ret[0] - e.ret[1]) for e in eps])) if eps else float("nan"))
    return WindowStats(epochs, len(eps), ret, surv, leave2, (tuple(low[0]), tuple(low[1])), sgl, joint)


def per_epoch(rec: dict, low_R: int = 12) -> List[WindowStats]:
    return [window_stats(rec, [e], low_R) for e in range(n_epochs(rec))]


def pi_given_R(rec: dict, epochs: Iterable[int], agent: int, K: int = 40) -> np.ndarray:
    """Counts[R, a] of unmasked actions by stock at the start of the round (thesis eq. estimators)."""
    keep = set(int(e) for e in epochs)
    key = "request_1" if agent == 0 else "request_2"
    counts = np.zeros((K + 1, 4), dtype=int)
    for i in range(len(rec["epoch"])):
        if int(rec["epoch"][i]) in keep and not rec["masked"][i]:
            counts[int(rec["R_start"][i]), int(rec[key][i])] += 1
    return counts


def social_metrics(ws: WindowStats, W_star: float) -> Dict[str, float]:
    """Pérolat et al. metrics normalised against the exact optimum (thesis eq. metrics)."""
    k, n = ws.survival
    joint, gap = ws.joint
    return {
        "sustainability": k / n if n else float("nan"),
        "efficiency": joint / W_star if W_star else float("nan"),
        "equality": 1 - gap / joint if joint else float("nan"),
    }


# ----------------------------------------------------------------------------- dynamics of a run


def first_majority_epoch(series: Sequence[Tuple[int, int]], threshold: float = 0.5) -> Optional[int]:
    """1-indexed first epoch whose share k/n exceeds the threshold; None if never."""
    for e, (k, n) in enumerate(series):
        if n and k / n > threshold:
            return e + 1
    return None


def stationary(series: Sequence[float], window: int = 20) -> Tuple[bool, float, float]:
    """Last-window mean vs preceding-window mean, in units of the pooled SE (plan §5).

    Returns (is_stationary, difference, se). Stationary iff |diff| < se.
    """
    xs = np.asarray(series, dtype=float)
    if len(xs) < 2 * window:
        return (False, float("nan"), float("nan"))
    a, b = xs[-window:], xs[-2 * window:-window]
    diff = float(a.mean() - b.mean())
    se = float(math.sqrt(a.var(ddof=1) / window + b.var(ddof=1) / window))
    return (abs(diff) < se, diff, se)


# ----------------------------------------------------------------------------- paired contrasts


@dataclass
class Paired:
    name: str
    per_seed: Dict[int, float]
    mean: float
    n_positive: int
    n_seeds: int
    all_same_sign: bool

    @property
    def verdict(self) -> str:
        if self.n_seeds == 0:
            return "no data"
        if self.all_same_sign and self.n_positive == self.n_seeds:
            return f"positive on every seed ({self.n_seeds})"
        if self.all_same_sign and self.n_positive == 0:
            return f"negative on every seed ({self.n_seeds})"
        return f"mixed sign ({self.n_positive}/{self.n_seeds} positive)"


def paired(name: str, arm1: Dict[int, float], arm2: Dict[int, float]) -> Paired:
    """Delta_s = arm1(s) - arm2(s) over the seeds present in both (thesis eq. paired)."""
    seeds = sorted(set(arm1) & set(arm2))
    d = {s: arm1[s] - arm2[s] for s in seeds if not (math.isnan(arm1[s]) or math.isnan(arm2[s]))}
    vals = list(d.values())
    npos = sum(v > 0 for v in vals)
    same = len(vals) > 0 and (npos == len(vals) or npos == 0) and all(v != 0 for v in vals)
    return Paired(name, d, float(np.mean(vals)) if vals else float("nan"), npos, len(vals), same)


# ----------------------------------------------------------------------------- run / arm containers


@dataclass
class Run:
    arm: str
    seed: int
    rec: dict
    epochs: int = field(init=False)

    def __post_init__(self):
        self.epochs = n_epochs(self.rec)

    def last(self, window: int) -> WindowStats:
        return window_stats(self.rec, range(max(0, self.epochs - window), self.epochs))

    def curve(self) -> List[WindowStats]:
        return per_epoch(self.rec)


def discover_runs(folder, arm: str) -> List[Run]:
    """exp<k>_cpr_records -> seed k-1 (the post-fix convention; legacy folders are remapped by the CLI)."""
    folder = Path(folder)
    runs = []
    for p in sorted(folder.glob("exp*_cpr_records")):
        k = int(p.name[3:].split("_")[0])
        runs.append(Run(arm, k - 1, load_records(p)))
    return runs


def by_seed(runs: List[Run], fn) -> Dict[int, float]:
    return {r.seed: fn(r) for r in runs}


# ----------------------------------------------------------------------------- DP anchors


def dp_anchors(stage: str) -> Dict[str, float]:
    """Exact values the readouts are compared against (cpr_xi.py)."""
    from cpr_xi import XI_TENTHS, best_response_vs, const, expected_policy, feedback_low, joint_optimum, open_then
    xi = (10,) if stage == "A" else XI_TENTHS
    e = lambda p1, p2: expected_policy(p1, p2, xi=xi)
    hawk_fb = e(const(2), feedback_low(12))
    hawk_once = e(const(2), open_then(1, 2))
    return {
        "W_star": joint_optimum(xi=xi),
        "hawk_vs_feedback_return_hawk": hawk_fb[0],
        "hawk_vs_feedback_return_dove": hawk_fb[1],
        "hawk_vs_restrain_once_return_hawk": hawk_once[0],
        "hawk_vs_restrain_once_return_dove": hawk_once[1],
        "hawk_vs_restrain_once_survival": hawk_once[2],
        "hawk_vs_feedback_survival": hawk_fb[2],
        "br_vs_feedback": best_response_vs(feedback_low(12), xi=xi)[0],
        "br_vs_hawk": best_response_vs(2, xi=xi)[0],
        "mutual_1": e(const(1), const(1))[0],
        "mutual_2": e(const(2), const(2))[0],
        "hawk_vs_dove": e(const(2), const(1))[0],
    }


# ----------------------------------------------------------------------------- hypothesis readouts (plan §1)


def readout_C2(runs: List[Run], window: int, anchors: Dict[str, float]) -> dict:
    """Learnability ceiling: leave-2 at R<12, survival | leave-2, return vs DP cells, per seed."""
    out = {}
    for r in runs:
        ws = r.last(window)
        k, n = ws.leave2_low[0]
        ks, ns = ws.surv_given_leave2[0]
        out[r.seed] = {
            "leave2_low": k / n if n else float("nan"), "leave2_low_ci": wilson(k, n),
            "surv_given_leave2": ks / ns if ns else float("nan"), "surv_given_leave2_ci": wilson(ks, ns),
            "return": ws.ret[0][0], "return_se": ws.ret[0][1],
            # plan §1 C2: low-stock restraint >= 0.5 and conditional survival at or above the
            # constant-continuation cell (0.40 in Stage B; vacuous at 1.00 in Stage A).
            "passes": bool(n and k / n >= 0.5 and ns and ks / ns >= anchors["hawk_vs_restrain_once_survival"]),
        }
    return out


def readout_rung(name: str, upper: List[Run], lower: List[Run], window: int) -> Dict[str, Paired]:
    """One rung of the ladder: paired differences upper - lower on the H-B readouts."""
    f_ret2 = lambda r: r.last(window).ret[1][0]
    f_l1 = lambda r: (lambda kn: kn[0] / kn[1] if kn[1] else float("nan"))(r.last(window).leave2[0])
    f_low1 = lambda r: (lambda kn: kn[0] / kn[1] if kn[1] else float("nan"))(r.last(window).leave2_low[0])
    f_surv = lambda r: (lambda kn: kn[0] / kn[1] if kn[1] else float("nan"))(r.last(window).survival)
    f_speed = lambda r: (lambda e: float(e) if e else float(r.epochs + 1))(
        first_majority_epoch([ws.leave2_low[0] for ws in r.curve()]))
    return {
        "agent2_return": paired(f"{name}: agent-2 return", by_seed(upper, f_ret2), by_seed(lower, f_ret2)),
        "agent1_leave2": paired(f"{name}: agent-1 leave-2", by_seed(upper, f_l1), by_seed(lower, f_l1)),
        "agent1_leave2_low": paired(f"{name}: agent-1 leave-2 at R<12", by_seed(upper, f_low1), by_seed(lower, f_low1)),
        "survival": paired(f"{name}: survival", by_seed(upper, f_surv), by_seed(lower, f_surv)),
        "speed_low_restraint": paired(f"{name}: epochs to majority low-stock restraint (lower is faster)",
                                      by_seed(lower, f_speed), by_seed(upper, f_speed)),
    }


def readout_HB4(runs: List[Run], window: int, anchors: Dict[str, float]) -> dict:
    """Exploitation: shaper (agent 2) return against the plain-hawk cell and the feedback-dove best response."""
    out = {}
    for r in runs:
        m, se = r.last(window).ret[1]
        out[r.seed] = {"return": m, "se": se,
                       "above_hawk_cell": bool(m - 1.96 * (se if not math.isnan(se) else 0) > anchors["hawk_vs_feedback_return_hawk"]),
                       "hawk_cell": anchors["hawk_vs_feedback_return_hawk"], "ceiling": anchors["br_vs_feedback"]}
    return out


# ----------------------------------------------------------------------------- LaTeX emitters


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"


def latex_arm_table(arms: Dict[str, List[Run]], window: int, W_star: float) -> str:
    """Per-arm, per-seed table of the last-window readouts (plan §5)."""
    rows = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
            r"Arm & Seed & $\hat G_1$ & $\hat G_2$ & Survival & $\hat L_1$ & $\hat L_1(R<12)$ & Efficiency \\",
            r"\midrule"]
    for arm, runs in arms.items():
        for r in runs:
            ws = r.last(window)
            eff = social_metrics(ws, W_star)["efficiency"]
            g1 = f"{ws.ret[0][0]:.1f} ({ws.ret[0][1]:.1f})"
            g2 = f"{ws.ret[1][0]:.1f} ({ws.ret[1][1]:.1f})" if not math.isnan(ws.ret[1][0]) else "--"
            rows.append(f"{arm} & {r.seed} & {g1} & {g2} & {fmt_ci(*ws.survival)} & {fmt_ci(*ws.leave2[0])} & {fmt_ci(*ws.leave2_low[0])} & {eff:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def latex_paired_table(rungs: Dict[str, Dict[str, Paired]]) -> str:
    rows = [r"\begin{tabular}{llrrl}", r"\toprule",
            r"Rung & Readout & Mean $\Delta_s$ & Seeds & Sign agreement \\", r"\midrule"]
    for rung, d in rungs.items():
        for key, p in d.items():
            per = ", ".join(f"{v:+.2f}" for _, v in sorted(p.per_seed.items()))
            rows.append(f"{rung} & {key.replace('_', ' ')} & {p.mean:+.2f} & {per} & {p.verdict} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def latex_pi_table(counts: np.ndarray, R_range: Iterable[int]) -> str:
    rows = [r"\begin{tabular}{rrrrrr}", r"\toprule", r"$R$ & $n$ & $\hat\pi(0\mid R)$ & $\hat\pi(1\mid R)$ & $\hat\pi(2\mid R)$ & $\hat\pi(3\mid R)$ \\", r"\midrule"]
    for R in R_range:
        n = int(counts[R].sum())
        if n == 0:
            continue
        rows.append(f"{R} & {n} & " + " & ".join(f"{counts[R, a] / n:.2f}" for a in range(4)) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"
