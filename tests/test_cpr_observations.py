"""
Tests for CPRObservationManager.

Each of the first three tests corresponds to a bug that was reproduced in the old
regex-based observation manager and that this module is designed to make impossible.

Runs under pytest, or standalone: `python tests/test_cpr_observations.py`
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpr_observation_managers import CPRObservationManager, CPRObservationManagerConfig

DIGIT_TOKS = [235276, 235274, 235284, 235304]   # "0", "1", "2", "3" for gemma-2-2b-it
DIGITS = ["0", "1", "2", "3"]


def make(is_shaper=False, n_games=1):
    return CPRObservationManager(
        CPRObservationManagerConfig(
            action_toks=DIGIT_TOKS, action_strings=DIGITS, is_shaper=is_shaper, R0=20,
        ),
        n_games=n_games,
    )


def step(om, R, own, opp, own_recv, opp_recv, inner_t, outer_t=0):
    return om.build_observations([R], [own], [opp], [own_recv], [opp_recv], inner_t, outer_t)


# ------------------------------------------- regression: the state-freeze bug


def test_previous_round_line_changes_every_step():
    """The old manager substituted with a letters-only regex, so numeric actions never
    matched and this line froze at round 1 for the entire episode, silently."""
    om = make()
    lines = []
    for t, (own, opp) in enumerate([(3, 1), (0, 2), (2, 2)], start=1):
        obs = step(om, 14, own, opp, own, opp, inner_t=t)[0]
        lines.append(next(l for l in obs.splitlines() if l.startswith("In the previous round")))

    assert len(set(lines)) == 3, f"previous-round line is stale: {lines}"
    assert lines[0] == ("In the previous round you requested 3 and received 3; "
                        "the other agent requested 1 and received 1.")
    assert lines[2] == ("In the previous round you requested 2 and received 2; "
                        "the other agent requested 2 and received 2.")


def test_requests_and_receipts_are_reported_separately():
    """Under scarcity they differ. Reporting only the request would leave the prompt
    contradicting the reward with nothing to explain the gap."""
    obs = step(make(), 0, own=3, opp=3, own_recv=0, opp_recv=0, inner_t=5)[0]
    assert "you requested 3 and received 0" in obs
    assert "the other agent requested 3 and received 0" in obs


# ------------------------------------------- regression: the count-collision bug


def test_counts_are_structured_not_parsed():
    """The old manager read counts back out of the prompt with `:(\\d+)`, so a resource
    line could shift every count by one. Counts here never round-trip through text."""
    om = make(is_shaper=True)
    scripted = [(1, 1), (2, 2), (1, 1), (3, 0), (2, 2), (1, 1), (0, 0), (3, 3), (2, 2), (1, 1)]
    for t, (own, opp) in enumerate(scripted, start=1):
        obs = step(om, 14, own, opp, own, opp, inner_t=t)[0]

    expected = {"11": 4, "22": 3, "30": 1, "00": 1, "33": 1}
    for label, want in expected.items():
        assert f"{label}: {want}" in obs, f"{label} should be {want}\n{obs}"
    assert sum(om.counts[0]) == len(scripted)
    assert "Current resource level: 14." in obs, "resource must coexist with counts"


def test_counts_survive_an_episode_boundary_and_reset_at_trial_start():
    om = make(is_shaper=True)
    for t, (own, opp) in enumerate([(1, 1), (2, 2)], start=1):
        step(om, 14, own, opp, own, opp, inner_t=t)
    assert sum(om.counts[0]) == 2

    # Episode boundary inside a trial: inner_t wraps to 0 but outer_t advances.
    om.record_episode_end([5], [0])
    obs = step(om, 20, 3, 3, 0, 0, inner_t=0, outer_t=1)[0]
    assert sum(om.counts[0]) == 3, "counts must carry across an episode boundary"
    assert "episode 1 collapsed at round 5 (final resource 0)" in obs
    assert "In the previous round" not in obs, "a fresh episode has no previous round"
    assert "Current resource level: 20." in obs

    # Trial boundary: both counters wrap.
    obs = step(om, 20, 1, 1, 1, 1, inner_t=0, outer_t=0)[0]
    assert sum(om.counts[0]) == 0, "trial start must clear counts"
    assert om.episode_summaries[0] == [], "trial start must clear episode summaries"
    assert obs == om.game_description + om.instruction_prompt


def test_naive_learner_prompt_resets_every_episode():
    om = make(is_shaper=False)
    step(om, 14, 1, 1, 1, 1, inner_t=1)
    obs = step(om, 20, 2, 2, 2, 2, inner_t=0, outer_t=1)[0]
    assert obs == om.game_description + om.instruction_prompt
    assert "Joint request counts" not in obs


# ------------------------------------------- template and duck-typed surface


def test_gemma_turn_tags_are_present():
    """Building from scratch without the turn tags puts gemma-2-it off-template and
    shifts exactly the digit priors we log before training."""
    om = make()
    reset = om.game_description + om.instruction_prompt
    assert reset.startswith("<start_of_turn>user\n")
    assert reset.endswith("\n<end_of_turn>\n<start_of_turn>model\n")

    mid = step(om, 14, 1, 2, 1, 2, inner_t=3)[0]
    assert mid.startswith("<start_of_turn>user\n")
    assert mid.endswith("\n<end_of_turn>\n<start_of_turn>model\n")


def test_reset_observation_is_a_valid_step_zero_prompt():
    """outer_rollout builds resets as game_description + instruction_prompt, one shared
    string for all games. It must already carry the initial resource level."""
    om = make()
    reset = om.game_description + om.instruction_prompt
    assert "Current resource level: 20." in reset
    assert "In the previous round" not in reset
    assert "Reply with only one number: 0, 1, 2 or 3." in reset


def test_parallel_games_render_independently():
    om = make(is_shaper=True, n_games=3)
    obs = om.build_observations([5, 12, 20], [3, 1, 0], [0, 1, 2], [3, 1, 0], [0, 1, 2],
                                inner_t=4, outer_t=0)
    assert len(obs) == 3 and len(set(obs)) == 3
    assert "Current resource level: 5." in obs[0]
    assert "Current resource level: 12." in obs[1]
    assert om.counts[0][3 * 4 + 0] == 1 and om.counts[1][1 * 4 + 1] == 1
    assert om.counts[0][1 * 4 + 1] == 0, "counts must not leak between parallel games"


def test_length_mismatch_is_rejected():
    om = make(n_games=2)
    try:
        om.build_observations([5], [1], [1], [1], [1], inner_t=1, outer_t=0)
    except ValueError as e:
        assert "expected 2" in str(e)
    else:
        raise AssertionError("mismatched batch lengths must raise")


# ------------------------------------------- demo prompts

def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_prompts() -> None:
    """Print each scenario so that can check the prompts by eye.

    Run:  python tests/test_cpr_observations.py
    Later: comment out the `demo_prompts()` call in `__main__`.
    """
    # --- 0. one shaper round ---
    _banner("0. Shaper after ONE round: own=3, opp=1  →  expect 31: 1, rest 0")
    om = make(is_shaper=True)
    obs = step(om, R=14, own=3, opp=1, own_recv=3, opp_recv=1, inner_t=1)[0]
    print(obs)

    # --- 1. previous-round line updates ---
    _banner("1. Previous-round line changes every step (look at the three lines)")
    om = make()
    for t, (own, opp) in enumerate([(3, 1), (0, 2), (2, 2)], start=1):
        obs = step(om, 14, own, opp, own, opp, inner_t=t)[0]
        prev = next(l for l in obs.splitlines() if l.startswith("In the previous round"))
        print(f"  t={t}  {prev}")

    # --- 2. scarcity: request ≠ received ---
    _banner("2. Scarcity: requested 3 but received 0")
    obs = step(make(), 0, own=3, opp=3, own_recv=0, opp_recv=0, inner_t=5)[0]
    print(obs)

    # --- 3. shaper counts after a scripted sequence ---
    _banner("3. Shaper counts after 10 scripted rounds (expect 11:4, 22:3, …)")
    om = make(is_shaper=True)
    scripted = [(1, 1), (2, 2), (1, 1), (3, 0), (2, 2), (1, 1), (0, 0), (3, 3), (2, 2), (1, 1)]
    for t, (own, opp) in enumerate(scripted, start=1):
        obs = step(om, 14, own, opp, own, opp, inner_t=t)[0]
    print(obs)
    print(f"  (total counts = {int(sum(om.counts[0]))}, expected {len(scripted)})")

    # --- 4. episode boundary keeps memory; trial start clears it ---
    _banner("4a. Episode boundary (inner_t=0, outer_t=1): counts + episode summary stay")
    om = make(is_shaper=True)
    for t, (own, opp) in enumerate([(1, 1), (2, 2)], start=1):
        step(om, 14, own, opp, own, opp, inner_t=t)
    om.record_episode_end([5], [0])
    obs = step(om, 20, 3, 3, 0, 0, inner_t=0, outer_t=1)[0]
    print(obs)
    print(f"  (total counts = {int(sum(om.counts[0]))}, expected 3)")

    _banner("4b. Trial start (inner_t=0, outer_t=0): memory wiped → bare reset")
    obs = step(om, 20, 1, 1, 1, 1, inner_t=0, outer_t=0)[0]
    print(obs)
    print(f"  (total counts = {int(sum(om.counts[0]))}, expected 0)")

    # --- 5. naive agent has no history across episodes ---
    _banner("5. Naive at episode boundary: bare reset, no joint counts")
    om = make(is_shaper=False)
    step(om, 14, 1, 1, 1, 1, inner_t=1)
    obs = step(om, 20, 2, 2, 2, 2, inner_t=0, outer_t=1)[0]
    print(obs)

    # --- 6. gemma turn tags ---
    _banner("6. Gemma turn tags on reset and mid-episode (check start/end)")
    om = make()
    reset = om.game_description + om.instruction_prompt
    mid = step(om, 14, 1, 2, 1, 2, inner_t=3)[0]
    print("--- RESET ---")
    print(repr(reset[:40]), "...", repr(reset[-50:]))
    print("--- MID ---")
    print(repr(mid[:40]), "...", repr(mid[-50:]))

    # --- 7. reset observation ---
    _banner("7. Reset observation (resource 20, no previous round)")
    om = make()
    print(om.game_description + om.instruction_prompt)

    # --- 8. three parallel games ---
    _banner("8. Three parallel games (different R / actions → different prompts)")
    om = make(is_shaper=True, n_games=3)
    obs_list = om.build_observations(
        [5, 12, 20], [3, 1, 0], [0, 1, 2], [3, 1, 0], [0, 1, 2],
        inner_t=4, outer_t=0,
    )
    for i, obs in enumerate(obs_list):
        print(f"\n--- game {i} ---")
        print(obs)

    # --- 9. length mismatch ---
    _banner("9. Length mismatch raises ValueError")
    om = make(n_games=2)
    try:
        om.build_observations([5], [1], [1], [1], [1], inner_t=1, outer_t=0)
    except ValueError as e:
        print(f"  caught: {e}")
    else:
        print("  ERROR: expected ValueError")

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
