#!/usr/bin/env python3
"""
Verification script for the two-agent CPR environment design.

Reproduces every number quoted in the design.

Environment rules implemented:
  - 2 agents, discrete actions {0,1,2,3}, fixed horizon T
  - order within a step: harvest, then regenerate
  - scarcity rule: if a1 + a2 > R, each agent receives min(a_i, floor(R/2)),
    the remainder is wasted, and R is set to 0
  - absorbing zero: once R == 0, regeneration stops permanently
  - ceiling C: regeneration cannot push R above C
  - reward = units harvested that step (nothing else)
"""

from functools import lru_cache

ACTIONS = (0, 1, 2, 3)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def episode(policy_a, policy_b, R0, g, C, T):
    """Run one deterministic episode.

    policy_a / policy_b are callables (t, R) -> action.
    Returns (return_a, return_b, collapse_step or None).
    collapse_step is 1-indexed: the step at which R first became 0.
    """
    R = R0
    pa = pb = 0
    collapse = None

    for t in range(T):
        a, b = policy_a(t, R), policy_b(t, R)

        if a + b <= R:                       # normal harvest
            pa += a
            pb += b
            R -= (a + b)
        else:                                # scarcity rule
            cap = R // 2
            pa += min(a, cap)
            pb += min(b, cap)
            R = 0

        if R > 0:
            R = min(C, R + g)                # regenerate, capped at ceiling
        elif collapse is None:
            collapse = t + 1                 # absorbing: no regeneration ever again

    return pa, pb, collapse


def const(x):
    """Constant policy."""
    return lambda t, R: x


def payoff_matrix(R0, g, C, T):
    return {(a, b): episode(const(a), const(b), R0, g, C, T)
            for a in ACTIONS for b in ACTIONS} 


# --------------------------------------------------------------------------
# game-theoretic analysis
# --------------------------------------------------------------------------

def pure_nash(M, strict=False):
    """Pure-strategy Nash equilibria.

    strict=False -> weak equilibria included (ties count as best responses)
    strict=True  -> both deviations must be strictly worse
    """
    out = []
    for a in ACTIONS:
        for b in ACTIONS:
            best_a = max(M[(x, b)][0] for x in ACTIONS)
            best_b = max(M[(a, y)][1] for y in ACTIONS)
            if strict:
                ok = (M[(a, b)][0] == best_a
                      and all(M[(x, b)][0] < M[(a, b)][0] for x in ACTIONS if x != a)
                      and M[(a, b)][1] == best_b
                      and all(M[(a, y)][1] < M[(a, b)][1] for y in ACTIONS if y != b))
            else:
                ok = M[(a, b)][0] == best_a and M[(a, b)][1] == best_b
            if ok:
                out.append((a, b))
    return out


def best_response_margins(M):
    """For each opponent action, the gap between best and second-best reply.
    A margin of 0 means the learner has no gradient in that column."""
    margins = {}
    for b in ACTIONS:
        col = sorted((M[(x, b)][0] for x in ACTIONS), reverse=True)
        margins[b] = col[0] - col[1]
    return margins


def social_optimum(R0, g, C, T):
    """Max achievable JOINT return, by dynamic programming over joint harvest."""
    @lru_cache(None)
    def f(R, t):
        if t == T or R == 0:
            return 0
        best = 0
        for s in range(0, 2 * max(ACTIONS) + 1):     # joint harvest 0..6
            if s <= R:
                R2 = min(C, (R - s) + g) if (R - s) > 0 else 0
                best = max(best, s + f(R2, t + 1))
            else:
                best = max(best, 2 * (R // 2))       # wipe, then dead forever
        return best
    return f(R0, 0)


def show_matrix(R0, g, C, T, label=""):
    M = payoff_matrix(R0, g, C, T)
    print(f"\nR0={R0}  g={g}  ceiling={C}  T={T}   {label}")
    print("        " + "".join(f"   B={b:<10}" for b in ACTIONS))
    for a in ACTIONS:
        print(f"  A={a} " + "".join(f"  ({M[(a,b)][0]:4d},{M[(a,b)][1]:4d})"
                                    for b in ACTIONS))
    return M


def check(label, got, want):
    status = "ok " if got == want else "FAIL"
    print(f"  [{status}] {label:<52} got {got!r:>10}  expected {want!r}")
    assert got == want, f"{label}: got {got}, expected {want}"


# --------------------------------------------------------------------------
# 1. configuration and its analysis
# --------------------------------------------------------------------------

def section_chosen_configuration():
    R0, g, C, T = 20, 2, 20, 30
    print("=" * 78)
    print("1. CONFIGURATION:  R0=20, g=2, ceiling=20, T=30")
    print("=" * 78)
    M = show_matrix(R0, g, C, T, "<-- configuration")

    print("\n  headline payoffs")
    check("sustainable (1,1), per agent",        M[(1, 1)][0], 30)
    check("Nash (2,2), per agent",               M[(2, 2)][0], 18)
    check("Nash (3,3), per agent",               M[(3, 3)][0], 14)
    check("temptation: deviate to 2 vs coop 1",  M[(2, 1)][0], 36)
    check("victim of that deviation",            M[(1, 2)][0], 18)

    print("\n  collapse timing (step at which R first hits 0; None = survives)")
    check("(1,1) collapse step", M[(1, 1)][2], None)
    check("(2,2) collapse step", M[(2, 2)][2], 9)
    check("(3,3) collapse step", M[(3, 3)][2], 5)

    print("\n  equilibria")
    weak = pure_nash(M, strict=False)
    strict = pure_nash(M, strict=True)
    print(f"        weak (ties count):   {[(n, M[n][0]) for n in weak]}")
    print(f"        strict:              {[(n, M[n][0]) for n in strict]}")
    check("(1,1) is NOT an equilibrium -> real dilemma", (1, 1) in weak, False)
    check("symmetric equilibria present", [(2, 2), (3, 3)] == [n for n in weak if n[0] == n[1]], True)

    print("\n  best-response margins (0 = no learning gradient in that column)")
    margins = best_response_margins(M)
    for b, m in margins.items():
        flag = "   <-- indifference" if m == 0 else ""
        print(f"        vs B={b}: {m}{flag}")
    check("indifference exists only against B=2",
          [b for b, m in margins.items() if m == 0], [2])

    print("\n  welfare benchmarks")
    opt = social_optimum(R0, g, C, T)
    check("social optimum, joint (DP)", opt, 78)
    # Regeneration after the final harvest is never harvestable, so total inflow
    # available to the agents is R0 + g*(T-1), not R0 + g*T. The DP attains it exactly:
    # the social optimum exhausts every unit that ever enters the pool.
    check("closed-form upper bound R0 + g*(T-1)", R0 + g * (T - 1), 78)
    check("bound is exactly tight (optimum exhausts total inflow)", opt == R0 + g * (T - 1), True)
    for prof in [(1, 1), (2, 2), (3, 3)]:
        joint = M[prof][0] + M[prof][1]
        print(f"        efficiency of {prof}: {joint}/{opt} = {joint/opt:.0%}")
    check("efficiency of (1,1) rounds to 77%", round(100 * 60 / opt), 77)
    check("efficiency of (2,2) rounds to 46%", round(100 * 36 / opt), 46)
    check("efficiency of (3,3) rounds to 36%", round(100 * 28 / opt), 36)

    print("\n  stock trace under (3,3)")
    R, trace = R0, []
    for _ in range(8):
        R = 0 if 6 > R else R - 6
        R = min(C, R + g) if R > 0 else 0
        trace.append(R)
    check("stock trace, both take 3", trace[:6], [16, 12, 8, 4, 0, 0])


# --------------------------------------------------------------------------
# 2. why ceiling == R0 kills the abstain-then-liquidate exploit
# --------------------------------------------------------------------------

def section_ceiling():
    print("\n" + "=" * 78)
    print("2. CEILING = R0 REMOVES THE LIQUIDATION EXPLOIT")
    print("=" * 78)
    g, T = 2, 30

    def wait_then_dump(R0, C, k):
        pol = lambda t, R: 0 if t < k else 3
        return episode(pol, pol, R0, g, C, T)[0]

    print("\n  'abstain k steps, then extract at max', per agent:")
    for C, tag in [(20, "ceiling = R0 = 20  (configuration)"),
                    (30, "ceiling = R0 > 30  (test #1, exploit alive)"),
                   (60, "ceiling = 60 > R0  (test #2, exploit alive)")]:
        vals = {k: wait_then_dump(20, C, k) for k in (0, 3, 5, 10, 15, 20)}
        steady = payoff_matrix(20, g, C, T)[(1, 1)][0]
        print(f"        {tag}")
        print(f"          {'  '.join(f'k={k}:{v}' for k, v in vals.items())}"
              f"   | steady (1,1) = {steady}")

    check("with ceiling=R0, waiting never helps",
          {wait_then_dump(20, 20, k) for k in range(0, 21)}, {14})
    check("with ceiling=60, waiting beats steady moderate play",
          max(wait_then_dump(20, 60, k) for k in range(0, 26)) > 30, True)


# --------------------------------------------------------------------------
# 3. comparison against a previous config
# --------------------------------------------------------------------------

def section_comparison():
    print("\n" + "=" * 78)
    print("3. CONFIGURATION 1 vs CONFIGURATION 2")
    print("=" * 78)
    rows = []
    for (R0, g, C, T, name) in [(20, 2, 20, 30, "R0=20 T=30"),
                                (30, 2, 30, 40, "R0=30 T=40")]:
        M = payoff_matrix(R0, g, C, T)
        coop, nash = M[(1, 1)][0], M[(3, 3)][0]
        dev2, dev3 = M[(2, 1)][0], M[(3, 1)][0]
        best_dev = max(M[(x, 1)][0] for x in ACTIONS)
        rows.append((name, coop, nash, dev2, dev3, best_dev,
                     coop / nash, (best_dev - coop) / coop))
    print(f"\n  {'config':<12}{'coop':>6}{'(3,3)':>7}{'dev->2':>8}{'dev->3':>8} |"
          f"{'ratio':>7}{'best tempt':>12}")
    for n, c, p, d2, d3, bd, r, tm in rows:
        print(f"  {n:<12}{c:>6}{p:>7}{d2:>8}{d3:>8} |{r:>7.2f}{tm:>11.0%}")

    check("R0=20,T=30 ratio",           round(rows[0][6], 2), 2.14)
    check("R0=30,T=40 ratio",           round(rows[1][6], 2), 1.90)
    check("R0=20,T=30 best temptation", round(100 * rows[0][7]), 20)
    check("R0=30,T=40 best temptation", round(100 * rows[1][7]), 40)
    print("\n  NOTE: best deviation against a cooperator is action 2, NOT action 3.")
    print("        Scoring by action 3 understates it: 27 vs 36, and 42 vs 56.")

    print("\n  The trade-off is governed by T/R0 -- ratio and temptation move oppositely:")
    print(f"  {'R0':>4}{'T':>4}{'T/R0':>7} | {'coop':>5}{'(3,3)':>7}{'best dev':>10} |"
          f"{'ratio':>7}{'temptation':>12}")
    for R0, T in [(20, 24), (20, 28), (20, 30), (24, 32), (30, 36), (30, 40), (30, 44)]:
        M = payoff_matrix(R0, 2, R0, T)
        coop, nash = M[(1, 1)][0], M[(3, 3)][0]
        bd = max(M[(x, 1)][0] for x in ACTIONS)
        print(f"  {R0:>4}{T:>4}{T/R0:>7.2f} | {coop:>5}{nash:>7}{bd:>10} |"
              f"{coop/nash:>7.2f}{(bd-coop)/coop:>11.0%}")


# --------------------------------------------------------------------------
# 4. horizon ceiling: where the temptation disappears
# --------------------------------------------------------------------------

def section_horizon():
    print("\n" + "=" * 78)
    print("4. HOW LONG CAN THE HORIZON BE BEFORE DEFECTION STOPS PAYING?")
    print("=" * 78)
    R0, g, C = 20, 2, 20
    print(f"\n  {'T':>4}{'coop (1,1)':>12}{'deviate to 2':>14}{'pays?':>8}")
    last_ok = None
    for T in range(20, 45, 2):
        M = payoff_matrix(R0, g, C, T)
        coop, temp = M[(1, 1)][0], M[(2, 1)][0]
        pays = temp > coop
        if pays:
            last_ok = T
        print(f"  {T:>4}{coop:>12}{temp:>14}{'yes' if pays else 'NO':>8}")
    print(f"\n  -> temptation survives up to T = {last_ok}; decision stays at T = 30")
    check("defection still pays at T=30",
          payoff_matrix(R0, g, C, 30)[(2, 1)][0] > payoff_matrix(R0, g, C, 30)[(1, 1)][0], True)
    check("defection no longer pays at T=40",
          payoff_matrix(R0, g, C, 40)[(2, 1)][0] > payoff_matrix(R0, g, C, 40)[(1, 1)][0], False)


# --------------------------------------------------------------------------
# 5. to test the early claim: unconditional regrowth destroys the dilemma
# --------------------------------------------------------------------------

def section_absorbing():
    print("\n" + "=" * 78)
    print("5. WHY ABSORBING ZERO IS LOAD-BEARING")
    print("=" * 78)

    def episode_nonabsorbing(a, b, R0, g, C, T):
        """Same env but regeneration continues even at R == 0."""
        R, pa, pb = R0, 0, 0
        for _ in range(T):
            if a + b <= R:
                pa += a; pb += b; R -= (a + b)
            else:
                cap = R // 2
                pa += min(a, cap); pb += min(b, cap); R = 0
            R = min(C, R + g)
        return pa, pb

    print("\n  R0=20, g=2, T=30, no ceiling effect  (current configuration)")
    for rule, fn in [("absorbing zero",
                      lambda a, b: episode(const(a), const(b), 20, 2, 20, 30)[:2]),
                     ("unconditional regrowth",
                      lambda a, b: episode_nonabsorbing(a, b, 20, 2, 20, 30))]:
        c = fn(1, 1)[0]; d = fn(3, 3)[0]
        verdict = "moderation wins" if c > d else "GREED WINS -> no dilemma"
        print(f"        {rule:<24} (1,1)={c:>3}   (3,3)={d:>3}   {verdict}")

    check("absorbing: moderation wins", episode(const(1), const(1), 30, 2, 30, 40)[0]
          > episode(const(3), const(3), 30, 2, 30, 40)[0], True)
    check("unconditional: greed wins", episode_nonabsorbing(3, 3, 30, 2, 30, 40)[0]
          > episode_nonabsorbing(1, 1, 30, 2, 30, 40)[0], True)


# --------------------------------------------------------------------------
# 6. the parameter sweep that produced the chosen configuration
# --------------------------------------------------------------------------

def section_sweep(top=12):
    print("\n" + "=" * 78)
    print("6. PARAMETER SWEEP")
    print("=" * 78)
    print("""
  Filters applied:
    (a) sustainable profile (1,1) must NOT be an equilibrium   -> real dilemma
    (b) a strict equilibrium exists, and it is worse than (1,1)  -> tragedy present
    (c) (3,3) must collapse before 40% of the horizon          -> visible failure
    (d) deviation to 2 must beat cooperation by >= 10%         -> learnable temptation
  Ranked by ratio (1,1)/(3,3), tie-broken by shorter horizon (cheaper episodes).
""")
    rows = []
    for R0 in range(10, 41, 2):
        for T in range(16, 41, 2):
            g, C = 2, R0
            M = payoff_matrix(R0, g, C, T)
            coop, nash33 = M[(1, 1)][0], M[(3, 3)][0]
            eq = pure_nash(M, strict=True)
            if (1, 1) in pure_nash(M):              continue   # deviation must pay
            if not eq:                              continue   # need a well-defined trap
            if any(M[e][0] >= coop for e in eq):    continue   # trap must be worse
            if M[(3, 3)][2] is None or M[(3, 3)][2] > 0.4 * T:  continue
            temp = (max(M[(x, 1)][0] for x in ACTIONS) - coop) / coop
            if temp < 0.10:                         continue
            rows.append((coop / nash33, -T, R0, T, coop, nash33,
                         M[(2, 1)][0], temp, M[(3, 3)][2]))
    rows.sort(reverse=True)

    print(f"  {'rank':>4}{'R0=C':>6}{'T':>4} | {'coop':>5}{'(3,3)':>7}{'tempt':>7} |"
          f"{'ratio':>7}{'margin':>8}{'collapse':>10}")
    for i, r in enumerate(rows[:top], 1):
        mark = "  <-- chosen" if (r[2], r[3]) == (20, 30) else ""
        print(f"  {i:>4}{r[2]:>6}{r[3]:>4} | {r[4]:>5}{r[5]:>7}{r[6]:>7} |"
              f"{r[0]:>7.2f}{r[7]:>7.0%}{r[8]:>10}{mark}")

    print(f"\n  {len(rows)} of {len(range(10,41,2))*len(range(16,41,2))} "
          f"configurations passed all four filters.")
    chosen = [r for r in rows if (r[2], r[3]) == (20, 30)]
    check("R0=20, T=30 passes every filter", len(chosen) == 1, True)


if __name__ == "__main__":
    section_chosen_configuration()
    section_ceiling()
    section_comparison()
    section_horizon()
    section_absorbing()
    section_sweep()
    print("\n" + "=" * 78)
    print("All assertions passed.")
    print("=" * 78)
