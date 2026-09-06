"""
Tests for cpr_env.CPRDynamics against the verify_cpr.py ground-truth fixture.

Parity over the 16 constant profiles is necessary but NOT sufficient: a coverage census
over those profiles shows the scarcity branch fires in exactly one cell ((3,3) at R=4,
cap=2), `cap = 0` is never triggered, and no cell ever visits R in {1, 2}. A wrong
cap-zero path, a ceil where floor belongs, or broken behaviour at R=1 would sail through.
The explicit edge-case fixtures and the trajectory fuzz below close that gap.

Runs under pytest, or standalone: `python tests/test_cpr_env.py`
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpr_env import LOGISTIC, NO_COLLAPSE, CPRDynamics, CPRParams, logistic_growth, zero_growth_stocks
from verify_cpr import ACTIONS, const, episode, payoff_matrix

CONFIG = CPRParams(R0=20, g=2, ceiling=20, horizon=30)


def run_constant(params: CPRParams, a: int, b: int):
    """Roll one episode with both agents holding a constant request. Mirrors verify_cpr.episode."""
    env = CPRDynamics(params, n_games=1)
    ra = rb = 0
    for _ in range(params.horizon):
        out = env.step(np.array([a]), np.array([b]))
        ra += int(out.received_1[0])
        rb += int(out.received_2[0])
    return ra, rb, env.collapse_step_or_none(0)


def run_sequence(params: CPRParams, seq_a, seq_b):
    """Roll one episode from explicit per-step action sequences."""
    env = CPRDynamics(params, n_games=1)
    ra = rb = 0
    for a, b in zip(seq_a, seq_b):
        out = env.step(np.array([a]), np.array([b]))
        ra += int(out.received_1[0])
        rb += int(out.received_2[0])
    return ra, rb, env.collapse_step_or_none(0)


# ---------------------------------------------------------------- parity


def test_all_sixteen_cells_match_fixture():
    """Returns AND collapse step must match verify_cpr for every constant profile."""
    expected = payoff_matrix(CONFIG.R0, CONFIG.g, CONFIG.ceiling, CONFIG.horizon)
    for a in ACTIONS:
        for b in ACTIONS:
            assert run_constant(CONFIG, a, b) == expected[(a, b)], f"cell ({a},{b})"


def test_headline_anchors():
    """The numbers quoted throughout the plan and the writeup."""
    assert run_constant(CONFIG, 1, 1) == (30, 30, None)     # sustainable
    assert run_constant(CONFIG, 2, 2) == (18, 18, 9)        # weak NE
    assert run_constant(CONFIG, 3, 3) == (14, 14, 5)        # strict NE, fast collapse
    assert run_constant(CONFIG, 2, 1)[0] == 36              # temptation
    assert run_constant(CONFIG, 1, 2)[0] == 18              # victim


# ------------------------------------------------- edge cases the matrix misses


def test_cooperative_trap_at_r_equals_one():
    """R=1, both request 1: cap=0, both receive nothing, last unit wiped.

    Deliberate, and unreachable from any constant profile — moderation is punished
    exactly where moderation matters most."""
    env = CPRDynamics(CPRParams(R0=1, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([1]), np.array([1]))
    assert (int(out.received_1[0]), int(out.received_2[0])) == (0, 0)
    assert int(out.R_end[0]) == 0
    assert bool(out.scarcity[0]) and bool(out.depleted[0])
    assert not bool(out.masked[0]), "the collapse step itself is a real decision, never masked"


def test_odd_resource_cap_uses_floor():
    """R=3, requests (2,3): cap = 3//2 = 1, so each receives 1 and one unit is wasted."""
    env = CPRDynamics(CPRParams(R0=3, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([2]), np.array([3]))
    assert (int(out.received_1[0]), int(out.received_2[0])) == (1, 1)
    assert int(out.R_end[0]) == 0


def test_zero_is_absorbing_for_every_action_pair():
    """Once dead, no request yields anything and no regeneration occurs."""
    for a in ACTIONS:
        for b in ACTIONS:
            env = CPRDynamics(CPRParams(R0=0, g=2, ceiling=20, horizon=2), n_games=1)
            out = env.step(np.array([a]), np.array([b]))
            assert (int(out.received_1[0]), int(out.received_2[0])) == (0, 0), f"({a},{b})"
            assert int(out.R_end[0]) == 0, f"({a},{b})"
            assert bool(out.masked[0]), "a step facing a dead pool must be masked"


def test_exact_depletion_is_an_independent_collapse_path():
    """a1 + a2 == R takes the NORMAL branch, yet still collapses. Scarcity never fires."""
    env = CPRDynamics(CPRParams(R0=4, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([2]), np.array([2]))
    assert (int(out.received_1[0]), int(out.received_2[0])) == (2, 2), "full requests are paid"
    assert not bool(out.scarcity[0]), "this must not be routed through the scarcity rule"
    assert int(out.R_end[0]) == 0 and bool(out.depleted[0])
    assert env.collapse_step_or_none(0) == 1


def test_ceiling_caps_regeneration():
    """Regeneration never pushes the stock above the ceiling."""
    env = CPRDynamics(CPRParams(R0=20, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([0]), np.array([0]))
    assert int(out.R_end[0]) == 20


def test_collapse_step_is_one_indexed_and_latches():
    """Collapse records the FIRST step at which R hit zero, and is never overwritten."""
    env = CPRDynamics(CPRParams(R0=6, g=0, ceiling=20, horizon=5), n_games=1)
    for _ in range(5):
        env.step(np.array([3]), np.array([3]))
    assert env.collapse_step_or_none(0) == 1
    assert not env.survived()[0]


# ---------------------------------------------------------------- fuzz


def test_random_trajectories_match_fixture():
    """Seeded random action sequences across a parameter grid, against verify_cpr.episode."""
    rng = random.Random(0)
    grid = [(R0, g, C, T)
            for R0 in (1, 2, 3, 5, 12, 20, 33)
            for g in (0, 1, 2, 4)
            for C, T in ((R0, 12), (R0, 30), (max(R0, 40), 18))]

    for R0, g, C, T in grid:
        params = CPRParams(R0=R0, g=g, ceiling=C, horizon=T)
        for _ in range(6):
            seq_a = [rng.choice(ACTIONS) for _ in range(T)]
            seq_b = [rng.choice(ACTIONS) for _ in range(T)]
            expected = episode(lambda t, R: seq_a[t], lambda t, R: seq_b[t], R0, g, C, T)
            assert run_sequence(params, seq_a, seq_b) == expected, (R0, g, C, T, seq_a, seq_b)


# ---------------------------------------------------------------- structure


def test_symmetry_under_policy_swap():
    """Swapping the two agents' policies must swap their returns exactly.

    This is the test that would have caught the transposed r_matrix[1] in
    configs/ipd_shaping_repro.json, where r2 collapsed onto r1."""
    for a in ACTIONS:
        for b in ACTIONS:
            ra, rb, collapse = run_constant(CONFIG, a, b)
            rb_swapped, ra_swapped, collapse_swapped = run_constant(CONFIG, b, a)
            assert (ra, rb, collapse) == (ra_swapped, rb_swapped, collapse_swapped), f"({a},{b})"


def test_parallel_games_are_independent():
    """Vectorised rollout must equal n independent scalar rollouts, step for step."""
    rng = random.Random(7)
    n_games, T = 5, CONFIG.horizon
    seqs_a = [[rng.choice(ACTIONS) for _ in range(T)] for _ in range(n_games)]
    seqs_b = [[rng.choice(ACTIONS) for _ in range(T)] for _ in range(n_games)]

    env = CPRDynamics(CONFIG, n_games=n_games)
    totals = np.zeros((2, n_games), dtype=np.int64)
    for t in range(T):
        out = env.step(np.array([s[t] for s in seqs_a]), np.array([s[t] for s in seqs_b]))
        totals[0] += out.received_1
        totals[1] += out.received_2

    for i in range(n_games):
        assert (int(totals[0, i]), int(totals[1, i]), env.collapse_step_or_none(i)) \
            == run_sequence(CONFIG, seqs_a[i], seqs_b[i]), f"game {i}"


def test_reset_clears_collapse_state():
    env = CPRDynamics(CPRParams(R0=4, g=0, ceiling=20, horizon=3), n_games=2)
    env.step(np.array([3, 3]), np.array([3, 3]))
    assert not env.survived().any()
    env.reset()
    assert env.survived().all() and (env.R == 4).all() and env.t == 0
    assert (env.collapse_step == NO_COLLAPSE).all()


def test_everything_stays_integer():
    """No float may ever reach a prompt."""
    env = CPRDynamics(CPRParams(R0=5, g=2, ceiling=20, horizon=3), n_games=2)
    out = env.step(np.array([3, 1]), np.array([3, 1]))
    for name in ("R_start", "request_1", "received_1", "received_2", "R_end"):
        assert np.issubdtype(getattr(out, name).dtype, np.integer), name


# ---------------------------------------------------------------- logistic (live training rule)


def test_logistic_headlines():
    """Working logistic game: restraint lasts, greed collapses, not a flat vs-2 column."""
    assert run_constant(LOGISTIC, 1, 1) == (36, 36, None)
    assert run_constant(LOGISTIC, 2, 2) == (7, 7, 4)
    assert run_constant(LOGISTIC, 3, 3) == (5, 5, 2)


def test_logistic_growth_at_low_stock_is_positive():
    """rate=0.9, K=40: R=1 still grows. This is the freeze we refused to ship."""
    assert zero_growth_stocks(LOGISTIC.ceiling, LOGISTIC.rate_tenths) == ()
    env = CPRDynamics(CPRParams(R0=1, g=0, ceiling=40, horizon=1, rate_tenths=9), n_games=1)
    out = env.step(np.array([0]), np.array([0]))
    assert int(out.R_end[0]) == 1 + logistic_growth(1, 40, 9) == 2


def test_logistic_capacity_does_not_grow():
    env = CPRDynamics(CPRParams(R0=40, g=0, ceiling=40, horizon=1, rate_tenths=9), n_games=1)
    out = env.step(np.array([0]), np.array([0]))
    assert int(out.R_end[0]) == 40


def test_xi10_matches_old_growth_formula():
    """xi_tenths=10 is bit-identical to growth without ξ."""
    from cpr_env import round_half_even_div
    K, rate = 40, 9
    for R in range(0, K + 1):
        old = 0 if R <= 0 or R >= K else round_half_even_div(rate * R * (K - R), 10 * K)
        assert logistic_growth(R, K, rate, 10) == old


def test_logistic_preserves_absorbing_zero_and_exact_depletion():
    env = CPRDynamics(CPRParams(R0=4, g=0, ceiling=40, horizon=1, rate_tenths=9), n_games=1)
    out = env.step(np.array([2]), np.array([2]))
    assert (int(out.received_1[0]), int(out.received_2[0])) == (2, 2)
    assert not bool(out.scarcity[0])
    assert int(out.R_end[0]) == 0
    env = CPRDynamics(CPRParams(R0=0, g=0, ceiling=40, horizon=1, rate_tenths=9), n_games=1)
    out = env.step(np.array([1]), np.array([1]))
    assert int(out.R_end[0]) == 0 and bool(out.masked[0])


# ------------------------------------------- demo prompts

def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _fmt_out(out) -> str:
    return (
        f"  R_start={int(out.R_start[0])}  req=({int(out.request_1[0])},{int(out.request_2[0])})  "
        f"recv=({int(out.received_1[0])},{int(out.received_2[0])})  R_end={int(out.R_end[0])}  "
        f"scarcity={bool(out.scarcity[0])}  depleted={bool(out.depleted[0])}  "
        f"masked={bool(out.masked[0])}  step={out.step_index}"
    )

def demo_prompts() -> None:
    """Print each dynamics scenario so you can watch R / receipts change.

    Run:  python tests/test_cpr_env.py
    Later: comment out the `demo_prompts()` call in `__main__`.
    """
    # --- 0. headline constant profiles ---
    _banner("0. Headline constant profiles (full episode totals)")
    for a, b, note in [
        (1, 1, "sustainable"),
        (2, 2, "weak NE"),
        (3, 3, "strict NE, fast collapse"),
        (2, 1, "temptation for agent 1"),
        (1, 2, "victim for agent 1"),
    ]:
        ra, rb, collapse = run_constant(CONFIG, a, b)
        print(f"  ({a},{b}) → returns=({ra},{rb}), collapse={collapse}   [{note}]")

    # --- 1. walk (3,3) until collapse ---
    _banner("1. Step-by-step (3,3) until collapse (expect collapse at step 5)")
    env = CPRDynamics(CONFIG, n_games=1)
    for _ in range(8):
        out = env.step(np.array([3]), np.array([3]))
        print(_fmt_out(out))
        if bool(out.depleted[0]) and bool(out.masked[0]):
            print("  ... pool already dead; further steps stay masked at R=0")
            break
    print(f"  collapse_step = {env.collapse_step_or_none(0)}")

    # --- 2. cooperative trap at R=1 ---
    _banner("2. Cooperative trap: R=1, both request 1 → cap=0, both get 0")
    env = CPRDynamics(CPRParams(R0=1, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([1]), np.array([1]))
    print(_fmt_out(out))

    # --- 3. odd resource floor ---
    _banner("3. Odd resource floor: R=3, requests (2,3) → each gets floor(3/2)=1")
    env = CPRDynamics(CPRParams(R0=3, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([2]), np.array([3]))
    print(_fmt_out(out))

    # --- 4. dead pool is absorbing ---
    _banner("4. Dead pool (R0=0): any action pair → recv 0, masked")
    env = CPRDynamics(CPRParams(R0=0, g=2, ceiling=20, horizon=2), n_games=1)
    out = env.step(np.array([3]), np.array([3]))
    print(_fmt_out(out))

    # --- 5. exact depletion (normal branch, not scarcity) ---
    _banner("5. Exact depletion: R=4, requests (2,2) → full pay, scarcity=False, then dead")
    env = CPRDynamics(CPRParams(R0=4, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([2]), np.array([2]))
    print(_fmt_out(out))
    print(f"  collapse_step = {env.collapse_step_or_none(0)}")

    # --- 6. ceiling caps regeneration ---
    _banner("6. Ceiling: R=20, requests (0,0), g=2 → still R_end=20 (not 22)")
    env = CPRDynamics(CPRParams(R0=20, g=2, ceiling=20, horizon=1), n_games=1)
    out = env.step(np.array([0]), np.array([0]))
    print(_fmt_out(out))

    # --- 7. collapse latches at first death ---
    _banner("7. Collapse latches: R0=6, g=0, forever (3,3) → collapse stays at step 1")
    env = CPRDynamics(CPRParams(R0=6, g=0, ceiling=20, horizon=5), n_games=1)
    for _ in range(5):
        out = env.step(np.array([3]), np.array([3]))
        print(_fmt_out(out) + f"  | collapse_so_far={env.collapse_step_or_none(0)}")

    # --- 8. symmetry ---
    _banner("8. Symmetry: (2,1) vs swapped (1,2)")
    print(f"  (2,1) → {run_constant(CONFIG, 2, 1)}")
    print(f"  (1,2) → {run_constant(CONFIG, 1, 2)}")

    # --- 9. reset clears collapse ---
    _banner("9. reset() clears collapse state")
    env = CPRDynamics(CPRParams(R0=4, g=0, ceiling=20, horizon=3), n_games=2)
    env.step(np.array([3, 3]), np.array([3, 3]))
    print(f"  after collapse: survived={env.survived().tolist()}  R={env.R.tolist()}")
    env.reset()
    print(f"  after reset:    survived={env.survived().tolist()}  R={env.R.tolist()}  t={env.t}")

    # --- 10. small matrix sample ---
    _banner("10. Sample of the 16-cell payoff matrix (a,b) → (ra, rb, collapse)")
    for a in ACTIONS:
        row = []
        for b in ACTIONS:
            ra, rb, c = run_constant(CONFIG, a, b)
            row.append(f"({ra},{rb},{c})")
        print(f"  a={a}: " + "  ".join(row))

    print("\n" + "=" * 72)
    print("Demo done.")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    demo_prompts()

    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok ] {name}")
            passed += 1
    print(f"\nAll {passed} tests passed.")
