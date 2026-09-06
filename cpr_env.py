"""
Deterministic Common-Pool Resource dynamics, vectorised over parallel games.

Single source of truth for CPR transitions. `verify_cpr.py` is the ground-truth fixture:
this module must reproduce its 16-cell payoff matrix exactly (see tests/test_cpr_env.py).

Rules, in order, per step:
  1. Both agents request a_i in {0, 1, 2, 3}.
  2. If a1 + a2 <= R   -> each receives its request;         R -= (a1 + a2)
     else (scarcity)   -> each receives min(a_i, R // 2);    R  = 0, remainder wasted
  3. If R > 0          -> regenerate, capped at ceiling (see CPRParams)
     else              -> R stays 0 permanently (absorbing zero)
Reward = units actually received. Episodes run the full horizon; a dead pool simply pays 0.

Regeneration is linear (`R + g`) when `rate_tenths` is None, and integer logistic
when it is set:
growth = round_half_even(rate_tenths * xi_tenths * R * (ceiling - R) / (100 * ceiling)).
`xi_tenths=10` is deterministic (bit-identical to the old formula without ξ).
Stage B multiplies the **growth increment** by ξ ∈ {0.7, 1.0, 1.3} (`xi_tenths` 7/10/13).
The live training configs use logistic; linear remains so the original fixtures still run.

Two edge cases that are easy to lose on a reimplementation, both deliberate:

  * Exact depletion is an independent collapse path. `a1 + a2 == R` takes the *normal*
    branch, drives R to 0, and the absorbing rule then kills it — the scarcity rule is
    never involved. Regenerating before checking `R > 0` silently loses this.

  * Scarcity destroys the pool even when a partial harvest was feasible. At R = 1, two
    *cooperative* agents both requesting 1 hit `cap = 0` and receive nothing, wiping the
    last unit. This is what makes the trap sharp: moderation is punished exactly where
    moderation matters most.

Conventions:
  * `collapse_step` is 1-indexed — the step at which R first reached 0 — matching
    `verify_cpr.episode`. `None`/-1 means the pool survived the episode.
  * All arithmetic is integer. No float ever reaches a prompt.
  * A step is *masked* for training iff `R_start == 0`, i.e. the agents faced an already
    dead pool. The collapse step itself, and the cap=0 case above, are real decisions and
    are NOT masked.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

NO_COLLAPSE = -1
# Mean-one three-point multiplier on the growth increment (Stage B).
# 10/10 = 1 is deterministic and bit-identical to the formula without ξ.
XI_TENTHS: Tuple[int, ...] = (7, 10, 13)


def round_half_even_div(num: int, den: int) -> int:
    """Nearest integer to num/den, ties to even. num >= 0, den > 0. Integer only."""
    q, r = divmod(num, den)
    two_r = r * 2
    if two_r > den:
        return q + 1
    if two_r < den:
        return q
    return q if q % 2 == 0 else q + 1


def round_half_even_div_vec(num: np.ndarray, den: int) -> np.ndarray:
    """Vectorised round_half_even_div. num >= 0, den > 0."""
    q, r = np.divmod(np.asarray(num, dtype=np.int64), den)
    two_r = r * 2
    up = two_r > den
    tie = two_r == den
    odd = (q % 2) != 0
    return q + up.astype(np.int64) + (tie & odd).astype(np.int64)


def logistic_growth(R: int, K: int, rate_tenths: int, xi_tenths: int = 10) -> int:
    """Integer logistic increment at post-harvest stock R. Zero at R=0 and R=K.

    ξ is multiplied **inside** the Fraction before rounding. xi_tenths=10 is
    bit-identical to the old (no-ξ) formula.
    """
    if R <= 0 or R >= K:
        return 0
    return round_half_even_div(rate_tenths * xi_tenths * R * (K - R), 100 * K)


def logistic_growth_vec(R: np.ndarray, K: int, rate_tenths: int, xi_tenths) -> np.ndarray:
    """Per-game growth. `xi_tenths` is a scalar or an array of shape R.shape."""
    R = np.asarray(R, dtype=np.int64)
    xi = np.broadcast_to(np.asarray(xi_tenths, dtype=np.int64), R.shape)
    growth = np.zeros_like(R)
    live = (R > 0) & (R < K)
    if not np.any(live):
        return growth
    num = rate_tenths * xi[live] * R[live] * (K - R[live])
    growth[live] = round_half_even_div_vec(num, 100 * K)
    return growth


def make_noise_table(seed: int, n_epochs: int, n_slots: int, t_max: int,
                     xi_tenths: Sequence[int] = XI_TENTHS) -> np.ndarray:
    """CRN table of ξ tenths, shape (n_epochs, n_slots, t_max).

    Spawned from SeedSequence(seed) so it is independent of the training RNG
    stream. Same (seed, shape, xi set) → same table on every arm.
    """
    ss = np.random.SeedSequence(int(seed)).spawn(1)[0]
    rng = np.random.default_rng(ss)
    xi = tuple(int(x) for x in xi_tenths)
    assert xi and all(x > 0 for x in xi), f"xi_tenths must be positive; got {xi}"
    return rng.choice(xi, size=(n_epochs, n_slots, t_max)).astype(np.int64)


def zero_growth_stocks(K: int, rate_tenths: int) -> tuple:
    """Interior stocks 1..K-1 at which rounding makes recovery impossible."""
    return tuple(R for R in range(1, K) if logistic_growth(R, K, rate_tenths) == 0)


@dataclass(frozen=True)
class CPRParams:
    """Environment parameters.

    Linear defaults match verify_cpr.py (`R0=20, g=2, ceiling=20, T=30`).
    Set `rate_tenths` to switch regeneration to logistic; `g` is then unused.
    Live training uses R0=8, ceiling=40, horizon=36, rate_tenths=9 (rate 0.9).
    """
    R0: int = 20
    g: int = 2
    ceiling: int = 20
    horizon: int = 30
    n_actions: int = 4
    rate_tenths: Optional[int] = None

    def __post_init__(self):
        assert self.R0 >= 0 and self.g >= 0 and self.horizon > 0, "R0, g must be >= 0; horizon > 0"
        assert self.ceiling >= self.R0, (
            f"ceiling ({self.ceiling}) below R0 ({self.R0}) would truncate the initial stock"
        )
        assert self.n_actions >= 2, f"need at least 2 actions; got {self.n_actions}"
        if self.rate_tenths is not None:
            assert self.rate_tenths > 0, f"rate_tenths must be > 0; got {self.rate_tenths}"

    @property
    def logistic(self) -> bool:
        return self.rate_tenths is not None

    @property
    def max_action(self) -> int:
        return self.n_actions - 1


# Live training configuration. Linear fixtures in tests keep the CPRParams defaults.
LOGISTIC = CPRParams(R0=8, g=0, ceiling=40, horizon=36, n_actions=4, rate_tenths=9)


@dataclass
class StepOutcome:
    """Per-step results, one entry per parallel game. All arrays have shape (n_games,)."""
    R_start: np.ndarray       # resource at the beginning of the step, before the decision
    request_1: np.ndarray
    request_2: np.ndarray
    received_1: np.ndarray    # equals reward for agent 1
    received_2: np.ndarray
    R_end: np.ndarray         # resource after harvest and regeneration
    scarcity: np.ndarray      # bool: the scarcity rule fired this step
    depleted: np.ndarray      # bool: pool is dead going into the next step (R_end == 0)
    masked: np.ndarray        # bool: R_start == 0, so this step is excluded from the PPO loss
    step_index: int           # 1-indexed step within the episode


class CPRDynamics:
    """CPR resource, held as one integer level per parallel game.

    Optional per-step `xi_tenths` multiplies the logistic growth increment.
    Default 10 is deterministic.
    """

    def __init__(self, params: CPRParams, n_games: int):
        assert n_games >= 1, f"n_games must be >= 1; got {n_games}"
        self.params, self.n_games = params, n_games
        self.R = np.empty(n_games, dtype=np.int64)
        self.reset()

    def reset(self) -> np.ndarray:
        """Start a fresh episode: every game back to R0, collapse bookkeeping cleared."""
        self.R.fill(self.params.R0)
        self.t = 0
        self.collapse_step = np.full(self.n_games, NO_COLLAPSE, dtype=np.int64)
        return self.R.copy()

    def step(self, request_1: np.ndarray, request_2: np.ndarray,
             xi_tenths=10) -> StepOutcome:
        """Advance every parallel game by one step. Requests must be in [0, n_actions-1].

        `xi_tenths` is a scalar or an array of shape (n_games,). 10 = deterministic.
        Never applied when post-harvest stock is 0 (absorbing zero stays absorbing).
        """
        a1 = np.asarray(request_1, dtype=np.int64)
        a2 = np.asarray(request_2, dtype=np.int64)
        assert a1.shape == a2.shape == (self.n_games,), (
            f"expected requests of shape ({self.n_games},); got {a1.shape} and {a2.shape}"
        )
        assert ((0 <= a1) & (a1 <= self.params.max_action)).all(), f"illegal request in {a1}"
        assert ((0 <= a2) & (a2 <= self.params.max_action)).all(), f"illegal request in {a2}"

        R_start = self.R.copy()
        total = a1 + a2
        scarcity = total > R_start

        # Scarcity: each agent is capped at floor(R/2) and the remainder is wasted.
        # Note this needs no special case for R_start == 0 — cap is 0, so both receive 0.
        cap = R_start // 2
        received_1 = np.where(scarcity, np.minimum(a1, cap), a1)
        received_2 = np.where(scarcity, np.minimum(a2, cap), a2)
        R_harvested = np.where(scarcity, 0, R_start - total)

        # Regeneration only while the pool is strictly alive; zero is absorbing.
        alive = R_harvested > 0
        if self.params.logistic:
            growth = logistic_growth_vec(
                R_harvested, self.params.ceiling, self.params.rate_tenths, xi_tenths
            )
            grown = R_harvested + growth
        else:
            grown = R_harvested + self.params.g
        R_end = np.where(alive, np.minimum(self.params.ceiling, grown), 0)

        self.t += 1
        now_dead = R_end == 0
        newly_dead = now_dead & (self.collapse_step == NO_COLLAPSE)
        self.collapse_step[newly_dead] = self.t
        self.R = R_end

        return StepOutcome(
            R_start=R_start,
            request_1=a1, request_2=a2,
            received_1=received_1, received_2=received_2,
            R_end=R_end.copy(),
            scarcity=scarcity,
            depleted=R_end == 0,
            masked=R_start == 0,
            step_index=self.t,
        )

    def survived(self) -> np.ndarray:
        """Bool per game: the pool is still alive."""
        return self.collapse_step == NO_COLLAPSE

    def collapse_step_or_none(self, game: int) -> Optional[int]:
        """1-indexed collapse step for one game, or None if it survived (fixture convention)."""
        step = int(self.collapse_step[game])
        return None if step == NO_COLLAPSE else step
