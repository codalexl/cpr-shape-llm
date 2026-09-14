"""Exact analysis of the two-player stochastic CPR family used to design the next environment.

The family keeps the live mechanics of `cpr_env` (integer logistic regrowth with half-even rounding, the
scarcity rule, an absorbing empty pool, and the multiplicative shock xi in {0.7, 1.0, 1.3} on the growth
increment) and fixes four design choices:

  * actions are restrain / harvest / grab = 1 / 2 / 3, with no abstain action: take-0 lets one player keep
    the pool alive against a greedy partner, which turns the scarce regime back into Chicken;
  * episodes start with a full pool, so collapse comes from over-harvest rather than from the starting stock;
  * each round continues with probability `continuation` (35/36, an expected 36 rounds) and the prompt shows
    no round counter, so optimal play is stationary in the stock and last round's actions;
  * the growth rate is the dial. m, the peak integer growth at xi = 1, sets the class. At m = 2 no stock
    lets one restraining player carry a harvester; at m = 3 it can.

Everything is exact: finite states, a discrete shock, linear solves. Nothing is simulated.

  `invariants(dial)`     I2 (no unilateral salvation) as drift and survival; I3 (committed greed fails against
                         a best responder); what a restrained partner is worth to a best responder.
  `gate(dial, learner)`  I4: against a specified toy learner, the shaper's best myopic response, best fixed
                         rule and best learning-aware schedule, and the policy each leaves the learner with.
  `python cpr_dial.py`   prints the invariants for the design points; `--gate` adds the I4 table.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from cpr_env import XI_TENTHS, logistic_growth
from cpr_xi import harvest_and_grow

RESTRAIN, HARVEST, GRAB = 1, 2, 3
Policy = Callable[[int, int, int], int]  # (stock, own last action, partner's last action) -> action


@dataclass(frozen=True)
class Dial:
    """One point of the family. The defaults are the m = 2 design point."""
    K: int = 20
    rate_tenths: int = 5
    S0: Optional[int] = None  # None: start with a full pool
    actions: Tuple[int, ...] = (RESTRAIN, HARVEST, GRAB)
    xi_tenths: Tuple[int, ...] = XI_TENTHS
    continuation: float = 35 / 36

    @property
    def start(self) -> int:
        return self.K if self.S0 is None else self.S0

    def peak_growth(self, xi: int = 10) -> int:
        return max(logistic_growth(S, self.K, self.rate_tenths, xi) for S in range(1, self.K))

    @property
    def m(self) -> int:
        """Peak integer growth per round at xi = 1, in harvest units."""
        return self.peak_growth(10)


def always(action: int) -> Policy:
    return lambda stock, own, other: action


def tit_for_tat(stock: int, own: int, other: int) -> int:
    """Restrain if the partner restrained last round, otherwise harvest. Round 0 counts as mutual restraint."""
    return RESTRAIN if other == RESTRAIN else HARVEST


@lru_cache(maxsize=None)
def outcome(dial: Dial, R: int, a1: int, a2: int) -> Tuple[int, int, Tuple[Tuple[int, Fraction], ...]]:
    """Receipts and the exact next-stock distribution for one round, xi uniform on the dial's shock."""
    dist: Dict[int, Fraction] = {}
    for xi in dial.xi_tenths:
        r1, r2, R_next = harvest_and_grow(R, a1, a2, xi, K=dial.K, rate_tenths=dial.rate_tenths)
        dist[R_next] = dist.get(R_next, Fraction(0)) + Fraction(1, len(dial.xi_tenths))
    return r1, r2, tuple(sorted(dist.items()))


# ----------------------------------------------------------------------------- I2: drift and survival


def drift(dial: Dial, a1: int, a2: int) -> Tuple[Fraction, int, Fraction]:
    """Largest expected one-round stock change under a constant pair, over stocks where scarcity does not bind:
    (expected change, the stock where it is attained, probability the stock does not fall there)."""
    best = None
    for R in range(a1 + a2 + 1, dial.K + 1):
        dist = outcome(dial, R, a1, a2)[2]
        change = sum(p * (Rn - R) for Rn, p in dist)
        if best is None or change > best[0]:
            best = (change, R, sum(p for Rn, p in dist if Rn >= R))
    return best


def never_rises(dial: Dial, a1: int, a2: int) -> bool:
    return all(Rn <= R for R in range(1, dial.K + 1) for Rn, _ in outcome(dial, R, a1, a2)[2])


def alive_after(dial: Dial, a1: int, a2: int, rounds: int) -> float:
    """Probability the pool is still alive after `rounds` rounds of a constant pair, with no random end."""
    dist = {dial.start: 1.0}
    for _ in range(rounds):
        nxt: Dict[int, float] = {}
        for R, p in dist.items():
            for Rn, q in outcome(dial, R, a1, a2)[2]:
                nxt[Rn] = nxt.get(Rn, 0.0) + p * float(q)
        dist = nxt
    return 1.0 - dist.get(0, 0.0)


# ----------------------------------------------------------------------------- stationary play


class _Chain:
    """States (stock, player 1's last action, player 2's last action). Stock 0 is absorbing and pays nothing."""

    def __init__(self, dial: Dial):
        self.dial, self.A = dial, dial.actions
        self.k = len(self.A)
        self.pos = {a: i for i, a in enumerate(self.A)}
        self.n = (dial.K + 1) * self.k ** 2
        self.start = self.state(dial.start, RESTRAIN, RESTRAIN)  # round 0 counts as mutual restraint

    def state(self, R: int, last1: int, last2: int) -> int:
        return (R * self.k + self.pos[last1]) * self.k + self.pos[last2]

    def live_states(self):
        for R in range(1, self.dial.K + 1):
            for last1 in self.A:
                for last2 in self.A:
                    yield R, last1, last2

    def row(self, R: int, a1: int, a2: int):
        """Receipts, [(next state, probability)], and the probability the pool is alive next round."""
        r1, r2, dist = outcome(self.dial, R, a1, a2)
        nxt = [(self.state(Rn, a1, a2), float(p)) for Rn, p in dist]
        return r1, r2, nxt, float(sum(p for Rn, p in dist if Rn > 0))


def _play(dial: Dial, policy1: Policy, policy2: Policy):
    """Transitions, both players' receipts, and the probability the pool is alive after the round, per state."""
    ch = _Chain(dial)
    P, r, alive = np.zeros((ch.n, ch.n)), np.zeros((ch.n, 2)), np.zeros(ch.n)
    for R, last1, last2 in ch.live_states():
        s = ch.state(R, last1, last2)
        c1, c2, nxt, alive[s] = ch.row(R, policy1(R, last1, last2), policy2(R, last2, last1))
        r[s] = (c1, c2)
        for t, p in nxt:
            P[s, t] += p
    return ch, P, r, alive


def evaluate(dial: Dial, policy1: Policy, policy2: Policy) -> Tuple[float, float, float]:
    """Expected returns of both players and the probability that the episode ends before the pool empties, under
    the uncapped geometric end (mean 36 rounds at the design points)."""
    ch, P, r, alive = _play(dial, policy1, policy2)
    delta = dial.continuation
    v = np.linalg.solve(np.eye(ch.n) - delta * P, np.column_stack([r, (1 - delta) * alive]))[ch.start]
    return float(v[0]), float(v[1]), float(v[2])


def evaluate_capped(dial: Dial, policy1: Policy, policy2: Policy, cap: int = 108) -> Tuple[float, float, float]:
    """As `evaluate`, under the environment's end: the episode closes at min(Geometric, cap) rounds (mean 34.3 at
    the design points), so play that keeps the pool alive earns up to 1 - continuation**cap less."""
    ch, P, r, alive = _play(dial, policy1, policy2)
    delta, d = dial.continuation, np.zeros(ch.n)
    d[ch.start] = 1.0
    returns, survival = np.zeros(2), 0.0
    for t in range(1, cap + 1):
        reach = delta ** (t - 1)
        returns += reach * (d @ r)
        survival += (reach * (1 - delta) if t < cap else reach) * float(d @ alive)
        d = d @ P
    return float(returns[0]), float(returns[1]), survival


def _optimise(dial: Dial, partner: Policy, survival: bool = False):
    """Policy iteration for player 1 against a fixed partner; ties go to the smaller take."""
    ch, delta, rows = _Chain(dial), dial.continuation, None
    P, r = np.zeros((ch.k, ch.n, ch.n)), np.zeros((ch.k, ch.n))
    for R, last1, last2 in ch.live_states():
        s, b = ch.state(R, last1, last2), partner(R, last2, last1)
        for i, a in enumerate(ch.A):
            c1, _, nxt, alive = ch.row(R, a, b)
            r[i, s] = (1 - delta) * alive if survival else c1
            for t, p in nxt:
                P[i, s, t] += p
    rows, choice = np.arange(ch.n), np.zeros(ch.n, dtype=int)
    while True:
        V = np.linalg.solve(np.eye(ch.n) - delta * P[choice, rows], r[choice, rows])
        Q = r + delta * (P @ V)
        top = Q.max(axis=0)
        better = Q[choice, rows] < top - 1e-9
        if not better.any():
            return V, choice, ch
        choice = np.where(better, np.argmax(Q >= top - 1e-9, axis=0), choice)


def best_response(dial: Dial, partner: Policy) -> Tuple[float, Policy]:
    """Exact best-response value of player 1 against a fixed partner, and the policy that attains it."""
    V, choice, ch = _optimise(dial, partner)
    policy = lambda stock, own, other: ch.A[choice[ch.state(stock, own, other)]]
    return float(V[ch.start]), policy


def best_survival(dial: Dial, partner_action: int) -> float:
    """Highest probability, over all policies, that the episode ends before the pool empties against a partner
    that always takes `partner_action`."""
    V, _, ch = _optimise(dial, always(partner_action), survival=True)
    return float(V[ch.start])


def committed_leader(dial: Dial, leader: Policy) -> Tuple[float, float, float]:
    """(leader's return, follower's return, survival) when player 2 commits and player 1 best-responds."""
    _, follower = best_response(dial, leader)
    follower_return, leader_return, survival = evaluate(dial, follower, leader)
    return leader_return, follower_return, survival


@dataclass(frozen=True)
class Invariants:
    m: int
    peak_growth_by_xi: Tuple[int, ...]
    drift_restrain_vs_harvest: Fraction
    stock_never_rises: bool
    alive_after_36: float
    alive_after_100: float
    survival_restrain_vs_harvest: float
    best_survival_vs_harvest: float
    mutual_restraint: float
    restraint_worth: float
    committed_harvest: Tuple[float, float, float]
    tit_for_tat: Tuple[float, float, float]

    @property
    def no_unilateral_salvation(self) -> bool:
        """I2: restrain against harvest loses stock in expectation at every stock."""
        return self.drift_restrain_vs_harvest < 0

    @property
    def commitment_fails(self) -> bool:
        """I3: against a best-responding partner, committing to harvest earns less than tit-for-tat."""
        return self.committed_harvest[0] < self.tit_for_tat[0]


def invariants(dial: Dial) -> Invariants:
    change, _, _ = drift(dial, RESTRAIN, HARVEST)
    return Invariants(
        m=dial.m,
        peak_growth_by_xi=tuple(dial.peak_growth(xi) for xi in dial.xi_tenths),
        drift_restrain_vs_harvest=change,
        stock_never_rises=never_rises(dial, RESTRAIN, HARVEST),
        alive_after_36=alive_after(dial, RESTRAIN, HARVEST, 36),
        alive_after_100=alive_after(dial, RESTRAIN, HARVEST, 100),
        survival_restrain_vs_harvest=evaluate(dial, always(RESTRAIN), always(HARVEST))[2],
        best_survival_vs_harvest=best_survival(dial, HARVEST),
        mutual_restraint=evaluate(dial, always(RESTRAIN), always(RESTRAIN))[0],
        restraint_worth=best_response(dial, always(RESTRAIN))[0] - best_response(dial, always(HARVEST))[0],
        committed_harvest=committed_leader(dial, always(HARVEST)),
        tit_for_tat=committed_leader(dial, tit_for_tat),
    )


# ----------------------------------------------------------------------------- I4: the oracle gate

Rule = Tuple[int, int, int, int, int]  # (cut, below & partner restrained, below & not, above & restrained, above & not)
ALWAYS_HARVEST: Rule = (0, HARVEST, HARVEST, HARVEST, HARVEST)
ALWAYS_RESTRAIN: Rule = (0, RESTRAIN, RESTRAIN, RESTRAIN, RESTRAIN)
TIT_FOR_TAT: Rule = (0, RESTRAIN, HARVEST, RESTRAIN, HARVEST)


def rule_menu(cuts: Sequence[int] = (8, 12)) -> Tuple[Rule, ...]:
    """Every deterministic rule on (stock below the cut?, partner restrained last round?) -> restrain/harvest/grab.
    Rules that ignore the stock are listed once, with cut 0."""
    menu = set()
    for cut in cuts:
        for acts in itertools.product((RESTRAIN, HARVEST, GRAB), repeat=4):
            stockless = acts[0] == acts[2] and acts[1] == acts[3]
            menu.add((0,) + acts if stockless else (cut,) + acts)
    return tuple(sorted(menu))


def describe(rule: Rule) -> str:
    cut, br, bn, ar, an = rule
    if cut == 0:
        return f"partner restrained last -> {br}, else -> {bn}"
    return f"R<{cut}: restrained -> {br}, else -> {bn}; R>={cut}: restrained -> {ar}, else -> {an}"


@dataclass(frozen=True)
class ToyLearner:
    """A two-parameter softmax partner that restrains or harvests, updated after every episode by exact gradient
    ascent on its expected episode return. `kind` sets what its two restraint probabilities condition on:
    "reciprocity" (the shaper restrained last round, or not) or "stock_band" (stock below `band_cut`, or not)."""
    kind: str = "reciprocity"
    step: float = 0.3
    initial_restraint: float = 0.1
    band_cut: int = 12


class _GateKernel:
    """Dense transitions over (stock, shaper's last, learner's last) for shaper actions 1-3 and learner actions
    restrain/harvest, so one episode's values and the learner's gradient cost one matrix inverse."""

    def __init__(self, dial: Dial):
        assert dial.actions == (RESTRAIN, HARVEST, GRAB), "the gate is defined for restrain/harvest/grab"
        self.dial, self.n = dial, (dial.K + 1) * 9
        self.T = np.zeros((self.n, 3, 2, self.n))
        self.c_shaper, self.c_learner = np.zeros((self.n, 3, 2)), np.zeros((self.n, 3, 2))
        for R in range(1, dial.K + 1):
            for last_s, last_l in itertools.product(range(3), repeat=2):
                s = (R * 3 + last_s) * 3 + last_l
                for a_s, a_l in itertools.product(range(3), range(2)):
                    r_s, r_l, dist = outcome(dial, R, a_s + 1, a_l + 1)
                    self.c_shaper[s, a_s, a_l], self.c_learner[s, a_s, a_l] = r_s, r_l
                    for Rn, p in dist:
                        self.T[s, a_s, a_l, (Rn * 3 + a_s) * 3 + a_l] += float(p)
        index = np.arange(self.n)
        self.R, self.shaper_last, self.learner_last = index // 9, index // 3 % 3 + 1, index % 3 + 1
        self.start = dial.start * 9  # round 0 counts as mutual restraint

    def episode(self, rule: Rule, learner: ToyLearner, theta: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Shaper's and learner's expected episode returns, and the gradient of the learner's return in theta."""
        cut, br, bn, ar, an = rule
        restrained = self.learner_last == RESTRAIN
        a_s = np.where(self.R < cut, np.where(restrained, br, bn), np.where(restrained, ar, an)) - 1
        feature = (self.shaper_last == RESTRAIN) if learner.kind == "reciprocity" else (self.R < learner.band_cut)
        q = 1 / (1 + np.exp(-theta))
        p = np.where(feature, q[0], q[1])
        rows, delta = np.arange(self.n), self.dial.continuation
        T0, T1 = self.T[rows, a_s, 0], self.T[rows, a_s, 1]
        M = np.linalg.inv(np.eye(self.n) - delta * (p[:, None] * T0 + (1 - p)[:, None] * T1))
        cs0, cs1 = self.c_shaper[rows, a_s, 0], self.c_shaper[rows, a_s, 1]
        cl0, cl1 = self.c_learner[rows, a_s, 0], self.c_learner[rows, a_s, 1]
        v_learner = M @ (p * cl0 + (1 - p) * cl1)
        slope = (cl0 - cl1) + delta * ((T0 - T1) @ v_learner)
        start = M[self.start]
        grad = np.array([start @ (np.where(feature, q[0] * (1 - q[0]), 0.0) * slope),
                         start @ (np.where(~feature, q[1] * (1 - q[1]), 0.0) * slope)])
        return float(start @ (p * cs0 + (1 - p) * cs1)), float(v_learner[self.start]), grad


def _bilinear(V: np.ndarray, points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    step, n = grid[1] - grid[0], len(grid)
    f = (points - grid[0]) / step
    i = np.clip(np.floor(f).astype(int), 0, n - 2)
    t = f - i
    a, b, ta, tb = i[:, 0], i[:, 1], t[:, 0], t[:, 1]
    return (1 - ta) * (1 - tb) * V[a, b] + ta * (1 - tb) * V[a + 1, b] + (1 - ta) * tb * V[a, b + 1] + ta * tb * V[a + 1, b + 1]


@dataclass(frozen=True)
class GateResult:
    best_response: float
    best_fixed: float
    best_fixed_rule: Rule
    learning_aware: float
    always_harvest: float
    always_restrain: float
    tit_for_tat: float
    end_restraint: Dict[str, Tuple[float, float]]  # the learner's two restraint probabilities after the run

    @property
    def learning_aware_gain(self) -> float:
        return self.learning_aware - self.best_fixed


def gate(dial: Dial, learner: ToyLearner, episodes: int = 100, grid_points: int = 9,
         rules: Optional[Sequence[Rule]] = None, limit: float = 5.0) -> GateResult:
    """The I4 check against one toy learner that starts prior-bound. Values are per-episode shaper returns.

    best response   each episode, the rule that maximises that episode's return (the learner treated as fixed)
    best fixed      the single rule, held for every episode, with the highest total (the learner still learns)
    learning aware  a rule per episode chosen by backward induction on the learner's parameters, solved on a
                    grid with bilinear interpolation; its realised value is a lower bound on the optimum
    """
    rules = tuple(dict.fromkeys(tuple(rules or rule_menu()) + (ALWAYS_HARVEST, ALWAYS_RESTRAIN, TIT_FOR_TAT)))
    kernel = _GateKernel(dial)
    theta0 = np.full(2, np.log(learner.initial_restraint / (1 - learner.initial_restraint)))
    grid = np.linspace(-limit, limit, grid_points)

    def play(choose):
        theta, total = theta0.copy(), 0.0
        for h in range(episodes):
            value, _, g = kernel.episode(choose(h, theta), learner, theta)
            total, theta = total + value, np.clip(theta + learner.step * g, -limit, limit)
        return total / episodes, tuple(float(x) for x in 1 / (1 + np.exp(-theta)))

    table = [kernel.episode(r, learner, np.array([a, b])) for a in grid for b in grid for r in rules]
    J = np.array([t[0] for t in table]).reshape(grid_points, grid_points, len(rules))
    G = np.array([t[2] for t in table]).reshape(grid_points, grid_points, len(rules), 2)
    to_go = [np.zeros((grid_points, grid_points))]
    for _ in range(episodes):
        later, now = to_go[0], np.empty((grid_points, grid_points))
        for i, j in itertools.product(range(grid_points), repeat=2):
            moved = np.clip(np.array([grid[i], grid[j]]) + learner.step * G[i, j], -limit, limit)
            now[i, j] = np.max(J[i, j] + _bilinear(later, moved, grid))
        to_go.insert(0, now)

    def myopic(h, theta):
        return max(rules, key=lambda r: kernel.episode(r, learner, theta)[0])

    def learning_aware(h, theta):
        def score(r):
            value, _, g = kernel.episode(r, learner, theta)
            return value + _bilinear(to_go[h + 1], np.clip(theta + learner.step * g, -limit, limit)[None], grid)[0]
        return max(rules, key=score)

    fixed = {r: play(lambda h, theta, r=r: r) for r in rules}
    best_rule = max(fixed, key=lambda r: fixed[r][0])
    br, la = play(myopic), play(learning_aware)
    return GateResult(
        best_response=br[0], best_fixed=fixed[best_rule][0], best_fixed_rule=best_rule, learning_aware=la[0],
        always_harvest=fixed[ALWAYS_HARVEST][0], always_restrain=fixed[ALWAYS_RESTRAIN][0],
        tit_for_tat=fixed[TIT_FOR_TAT][0],
        end_restraint={"best fixed": fixed[best_rule][1], "learning aware": la[1],
                       "always harvest": fixed[ALWAYS_HARVEST][1]},
    )


# ----------------------------------------------------------------------------- report

DESIGN = {"m=2": Dial(K=20, rate_tenths=5), "m=3": Dial(K=20, rate_tenths=6)}
CHECKS = {"m=2, K=30": Dial(K=30, rate_tenths=3), "m=3, K=30": Dial(K=30, rate_tenths=4)}
POND = Dial(K=40, rate_tenths=9, S0=8, actions=(0, 1, 2, 3), xi_tenths=(10,))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true", help="also run the I4 oracle gate (a few minutes)")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args(argv)

    worth = lambda d: best_response(d, always(RESTRAIN))[0] - best_response(d, always(HARVEST))[0]
    print(f"live pond (K=40, rate 0.9, takes 0-3, no noise): a restrained partner is worth {worth(POND):.1f} to a "
          f"best responder from R0=8 and {worth(replace(POND, S0=20)):.1f} from R0=20")
    abstain = committed_leader(replace(DESIGN['m=2'], actions=(0, 1, 2, 3)), always(HARVEST))
    print(f"m=2 with an abstain action: committed harvest earns {abstain[0]:.1f} (survival {abstain[2]:.2f})\n")
    for name, d in {**DESIGN, **CHECKS}.items():
        inv = invariants(d)
        print(f"{name} (K={d.K}, rate {d.rate_tenths / 10}): m={inv.m}; peak integer growth at xi "
              f"{'/'.join(str(x / 10) for x in d.xi_tenths)} = {'/'.join(map(str, inv.peak_growth_by_xi))}")
        print(f"  I2 restrain vs harvest: best expected drift {float(inv.drift_restrain_vs_harvest):+.2f}/round, "
              f"never rises: {inv.stock_never_rises}; alive after 36/100 rounds with no random end "
              f"{inv.alive_after_36:.2f}/{inv.alive_after_100:.2f}; ends before emptying {inv.survival_restrain_vs_harvest:.2f} "
              f"(best policy {inv.best_survival_vs_harvest:.2f}) -> no unilateral salvation: {inv.no_unilateral_salvation}")
        print(f"  I3 against a best responder: committed harvest earns {inv.committed_harvest[0]:.1f} (partner "
              f"{inv.committed_harvest[1]:.1f}, survival {inv.committed_harvest[2]:.2f}); tit-for-tat earns "
              f"{inv.tit_for_tat[0]:.1f} (partner {inv.tit_for_tat[1]:.1f}); mutual restraint {inv.mutual_restraint:.1f} "
              f"-> commitment fails: {inv.commitment_fails}")
        print(f"  a restrained partner is worth {inv.restraint_worth:.1f} to a best responder")
    if not a.gate:
        return
    print(f"\nI4 gate: {a.episodes} episodes from a learner restraining 10% of the time; per-episode shaper return")
    for name, d in DESIGN.items():
        for kind in ("reciprocity", "stock_band"):
            for step in (0.1, 0.3):
                g = gate(d, ToyLearner(kind=kind, step=step), episodes=a.episodes)
                ends = "  ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in g.end_restraint.items())
                print(f"{name} {kind:11s} step {step}: best response {g.best_response:5.1f} | best fixed {g.best_fixed:5.1f} "
                      f"[{describe(g.best_fixed_rule)}] | learning aware {g.learning_aware:5.1f} "
                      f"(gain {g.learning_aware_gain:+.1f}) | always harvest {g.always_harvest:5.1f} | "
                      f"tit-for-tat {g.tit_for_tat:5.1f}\n    learner restraint at the end: {ends}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- run configs

DIAL_TOKENS = [235274, 235284, 235304]  # gemma-2-2b-it tokens for "1", "2", "3"
DESIGN_RATES = {5: "m2", 6: "m3"}


def check_dial_config(config: dict, allow_fixed_horizon: bool = False) -> str:
    """Refuse any config that is not one of the two pre-registered design points, and return "m2" or "m3".

    The launcher applies this as strictly as the pond lock. `allow_fixed_horizon` admits the memory smoke run, in
    which every episode runs the 108-round cap instead of closing at random.
    """
    from cpr_observation_managers import DIAL_RULES
    gp = config["game_parameters"]
    fixed = dict(t_max=108, e_max=5, n_games=3, R0=20, g=0, ceiling=20, n_actions=3, min_take=1)
    for key, value in fixed.items():
        assert gp.get(key) == value, f"game_parameters.{key} = {gp.get(key)!r}; the design point needs {value!r}"
    assert gp.get("rate_tenths") in DESIGN_RATES, f"rate_tenths {gp.get('rate_tenths')!r} is not 5 (m=2) or 6 (m=3)"
    assert list(gp.get("xi_tenths") or []) == list(XI_TENTHS), f"xi_tenths {gp.get('xi_tenths')!r} != {list(XI_TENTHS)}"
    close = gp.get("close_continue")
    assert close == 35 / 36 or (allow_fixed_horizon and close is None), f"close_continue {close!r} is not 35/36"
    for i in (1, 2):
        obs = config[f"obs_manager_parameters{i}"]
        assert obs["action_toks"] == DIAL_TOKENS and obs["action_strings"] == ["1", "2", "3"], f"agent {i} actions"
        assert obs["R0"] == 20 and obs.get("rules") == DIAL_RULES, f"agent {i} must render the pre-registered rules"
        ppo = config.get(f"ppo_agent_parameters{i}")
        if ppo is not None:
            assert ppo["action_toks"] == DIAL_TOKENS, f"agent {i} PPO action tokens"
    return DESIGN_RATES[gp["rate_tenths"]]
