#!/usr/bin/env python3
"""
Trace the CPR training spine, live, with print statements.

    python scripts/trace_spine.py          # all three acts
    python scripts/trace_spine.py --act 1  # just one

This runs the REAL classes — CPRObservationManager, TokenToActionMapper, CPRDynamics,
CPRGame, TrajectoryData — so what you see printed is what actually happens in training.

The ONLY thing stubbed out is the LLM itself (steps 2-3 of the spine): loading
gemma-2-2b-it takes minutes and a lot of RAM, and it would tell you nothing you can't
read off the config. Everywhere the real agent would emit a token, this script emits the
same token id by hand. Every other stage is genuine code.

THE SPINE — one agent, one step:

    1. a prompt exists as a string          cpr_observation_managers.py
    2. string -> tokens                     agents.py:119   tokenize_observation   [STUBBED]
    3. tokens -> one masked token           agents.py:148   take_action            [STUBBED]
    4. token -> int in {0,1,2,3}            environment.py:89   TokenToActionMapper.map
    5. ints -> game rules -> new R, rewards cpr_env.py:95   CPRDynamics.step
    6. outcome -> reward + NEXT prompt      cpr_game.py:130  CPRGame.step
    7. save (prompt, token, reward)         environment.py:62  TrajectoryData._update

    x36 steps  = one episode                environment.py:263  inner_rollout
    x5 episodes = one trial                 environment.py:296  outer_rollout
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpr_env import LOGISTIC
from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import CPRObservationManagerConfig
from environment import EnvState, TrajectoryData

# Gemma-2-2b-it digit ids for "0", "1", "2", "3" (same as live configs).
ACTION_TOKS = [235276, 235274, 235284, 235304]
ACTION_STRS = ["0", "1", "2", "3"]
N_GAMES = 2


# ---------------------------------------------------------------------------- helpers

def banner(n, title, where=""):
    print(f"\n\033[1m{'=' * 76}\n {n}  {title}\033[0m")
    if where:
        print(f" \033[2m{where}\033[0m")
    print("=" * 76)


def stage(n, title, where):
    print(f"\n\033[1m--- STEP {n}: {title}\033[0m")
    print(f"    \033[2m{where}\033[0m")


def show_prompt(text, indent="      "):
    for line in text.replace("<start_of_turn>user\n", "").split("\n"):
        print(f"{indent}\033[36m|\033[0m {line}" if line else f"{indent}\033[36m|\033[0m")


def fake_llm(action_per_game):
    """Stand in for agents.py:148 `take_action`.

    The real thing runs the model, hard-masks every logit outside ACTION_TOKS to
    finfo.min (AllowedTokensLogitsProcessor), samples one token, and returns a list of
    (1,)-tensors. We skip straight to the return value.
    """
    return [torch.tensor([ACTION_TOKS[a]]) for a in action_per_game]


def make_game(shaper_is_agent2=True, t_max=None, e_max=5, n_games=N_GAMES):
    t_max = LOGISTIC.horizon if t_max is None else t_max
    common = dict(action_toks=ACTION_TOKS, action_strings=ACTION_STRS, R0=LOGISTIC.R0,
                  model_name="gemma-2b")
    return CPRGame(
        CPRGameParams(
            t_max=t_max, e_max=e_max, n_games=n_games,
            R0=LOGISTIC.R0, g=LOGISTIC.g, ceiling=LOGISTIC.ceiling,
            n_actions=LOGISTIC.n_actions, rate_tenths=LOGISTIC.rate_tenths,
        ),
        CPRObservationManagerConfig(is_shaper=False, **common),
        CPRObservationManagerConfig(is_shaper=shaper_is_agent2, **common),
    )


# ============================================================================ ACT 1

def act1():
    banner("ACT 1", "ONE STEP, UNDER A MICROSCOPE",
           "two parallel games; agent 1 will ask for 2, agent 2 will ask for 3")

    game = make_game()
    obs_mgr1 = game.obs_managers["agent_1"]
    obs_mgr2 = game.obs_managers["agent_2"]
    env_state = EnvState(inner_t=0, outer_t=0)

    # ---- STEP 1 -----------------------------------------------------------------
    stage(1, "A PROMPT EXISTS AS A STRING",
          "cpr_observation_managers.py:110 game_description + :96 instruction_prompt")
    print("\n    This is exactly what outer_rollout builds at environment.py:301.")
    print("    One shared string for all games, because every reset starts at R0=8.\n")
    obs1 = [obs_mgr1.game_description + obs_mgr1.instruction_prompt] * N_GAMES
    obs2 = [obs_mgr2.game_description + obs_mgr2.instruction_prompt] * N_GAMES
    show_prompt(obs1[0])
    print(f"\n    len = {len(obs1[0])} chars, identical for both of the {N_GAMES} games")

    # ---- STEPS 2-3 --------------------------------------------------------------
    stage("2-3", "STRING -> TOKENS -> ONE MASKED TOKEN   [STUBBED]",
          "agents.py:119 tokenize_observation | agents.py:148 take_action")
    print(f"""
    The real path: tokenize the prompt, forward pass, then
    AllowedTokensLogitsProcessor (utils/training_utils.py:50) sets EVERY logit
    outside these four ids to finfo.min before sampling:

        "0" -> {ACTION_TOKS[0]}      "1" -> {ACTION_TOKS[1]}
        "2" -> {ACTION_TOKS[2]}      "3" -> {ACTION_TOKS[3]}

    So the model cannot emit anything else. Out of gemma's 256k vocabulary,
    exactly 4 tokens have non-zero probability.

    (agents.py:124 _resample_illegal then redraws anyway - torch.multinomial on
     MPS returns a zero-probability token ~0.9% of the time. Platform bug, not ours.)

    We skip the forward pass and hand back the tokens directly:""")

    resp1 = fake_llm([2, 2])   # both games: agent 1 asks for 2
    resp2 = fake_llm([3, 3])   # both games: agent 2 asks for 3
    print(f"\n      agent 1 responses: {[r.tolist() for r in resp1]}   (token {ACTION_TOKS[2]} = \"2\")")
    print(f"      agent 2 responses: {[r.tolist() for r in resp2]}   (token {ACTION_TOKS[3]} = \"3\")")

    # ---- STEP 4 -----------------------------------------------------------------
    stage(4, "TOKEN -> INT", "environment.py:89 TokenToActionMapper.map")
    mapper = game.token_action_maps["agent_1"]
    a1 = mapper.map(resp1)
    a2 = game.token_action_maps["agent_2"].map(resp2)
    print(f"\n      map({[r.item() for r in resp1]}) -> {a1.tolist()}")
    print(f"      map({[r.item() for r in resp2]}) -> {a2.tolist()}")
    print(f"\n    Any token NOT in the legal list maps to {mapper.illegal_action} (= n_actions),")
    print("    which cpr_game.py:144 asserts can never happen. It's a wiring check.")

    # ---- STEP 5 -----------------------------------------------------------------
    stage(5, "INTS -> THE GAME RULES", "cpr_env.py CPRDynamics.step  (locked logistic)")
    print("""
    Live lock: R0=8, K=40, T=36, rate_tenths=9. Linear verify_cpr.py is the fixture
    only; training refuses anything but this point.

      scarcity = total > R_start
      R_harvested = 0 if scarce else R_start - total
      received = min(a, R_start // 2) if scarce else a
      growth = round_half_even(rate_tenths * R * (K - R) / (10 * K))   if R > 0
      R_end = 0 if R_harvested == 0 else min(K, R_harvested + growth)
""")
    R_before = game.dynamics.R.copy()
    print(f"      R before the step:  {R_before.tolist()}")
    print(f"      requests:           agent1={a1.tolist()}  agent2={a2.tolist()}   (sum = 5)")
    print("\n    5 <= 8, so this is the NORMAL branch. No scarcity.")
    print("      R after harvest:    8 - 5 = 3")
    print("      logistic growth at 3: 2   ->  R_end = 5")

    # ---- STEP 6 -----------------------------------------------------------------
    stage(6, "OUTCOME -> REWARD + THE NEXT PROMPT", "cpr_game.py:130 CPRGame.step")
    print("\n    CPRGame.step does 4, 5 and 6 together. Calling it for real now:\n")
    result = game.step(obs1, obs2, resp1, resp2, env_state)

    print(f"      reward agent 1: {[float(r.item()) for r in result.r1]}   (it asked 2, received 2)")
    print(f"      reward agent 2: {[float(r.item()) for r in result.r2]}   (it asked 3, received 3)")
    print(f"      R now:          {game.dynamics.R.tolist()}")
    print(f"      env_state:      inner_t={result.new_env_state.inner_t}, "
          f"outer_t={result.new_env_state.outer_t}")
    print("\n    And here is the NEXT prompt agent 1 will see. Note the two new lines:\n")
    show_prompt(result.new_obs1[0])
    print("""
    The resource line moved 8 -> 5, and a previous-round line appeared.
    It reports BOTH the request and the receipt (cpr_observation_managers.py:124).
    Under scarcity those two differ, and if the prompt showed only the request the
    agent could never infer the scarcity rule from what it observes.""")

    # ---- STEP 7 -----------------------------------------------------------------
    stage(7, "SAVE THE TRIPLE", "environment.py:62 TrajectoryData._update")
    traj = TrajectoryData(last_observation=obs1)
    traj._update(new_queries=[torch.tensor([1, 2, 3])] * N_GAMES,   # stand-in for tokenized prompts
                 new_responses=resp1,
                 new_ids=list(range(N_GAMES)),
                 new_rewards=result.r1)
    print(f"\n      stored {len(traj.rewards[0])} transitions, rewards = "
          f"{[float(r.item()) for r in traj.rewards[0]]}")
    print("""
    That's the whole loop body. Everything PPO ever sees is a list of these triples:
    (tokenized prompt, the one token it emitted, the scalar reward it got).

    Note `new_ids` — the game index. utils/training_utils.py:245 groups by it so the
    GAE recursion never bleeds one parallel game into another.""")

    banner("ACT 1 DONE", "you have now seen a full pass through steps 1-7")


# ============================================================================ ACT 2

def act2():
    banner("ACT 2", "x36 STEPS = ONE EPISODE", "environment.py:263 inner_rollout")
    print("""
 Both agents hold 2 every step. On the locked logistic, (2,2) pays 7 each and
 dies at round 4 (LIVE_FACTS). Watch it happen.
""")
    game = make_game(n_games=1)
    obs1 = [game.obs_managers["agent_1"].game_description
            + game.obs_managers["agent_1"].instruction_prompt]
    obs2 = list(obs1)
    env_state = EnvState(inner_t=0, outer_t=0)
    traj = TrajectoryData(last_observation=obs1)

    print(f"  {'step':>4} {'R_start':>8} {'a1':>3} {'a2':>3} {'recv1':>6} {'recv2':>6}"
          f" {'R_end':>6} {'scarce':>7} {'MASKED':>7}   {'reward to PPO':>14}")
    print("  " + "-" * 74)

    total1 = 0
    collapse_step = None
    for t in range(game.t_max):
        resp1, resp2 = fake_llm([2]), fake_llm([2])
        R_start = int(game.dynamics.R[0])
        result = game.step(obs1, obs2, resp1, resp2, env_state)
        obs1, obs2, env_state = result.new_obs1, result.new_obs2, result.new_env_state

        traj._update(new_queries=[torch.tensor([1])], new_responses=resp1,
                     new_ids=[0], new_rewards=result.r1)

        i = len(game.records["step"]) - 1
        rec = {k: game.records[k][i] for k in game.records}
        total1 += rec["reward_1"]
        # Read collapse off the records: CPRGame.step resets the dynamics at the episode
        # boundary, so game.dynamics.collapse_step is already cleared by the time we exit.
        if collapse_step is None and rec["depleted"]:
            collapse_step = rec["step"]

        # Steps after collapse are identical dead-pool rows.
        if 5 <= t <= game.t_max - 2:
            if t == 5:
                print("  " + "." * 62 + "  (remaining steps: identical, dead pool)")
            continue

        r_ppo = float(result.r1[0].item())
        flag = "  <-- masked" if rec["masked"] else ""
        note = "nan  (dropped)" if np.isnan(r_ppo) else f"{r_ppo:.1f}"
        print(f"  {t+1:>4} {R_start:>8} {rec['request_1']:>3} {rec['request_2']:>3}"
              f" {rec['received_1']:>6} {rec['received_2']:>6} {rec['R_end']:>6}"
              f" {str(rec['scarcity']):>7} {str(rec['masked']):>7}{flag}   {note:>14}")

    kept = sum(len(x) for x in traj.rewards)
    print(f"""
  \033[1mRead that table against LIVE_FACTS (logistic (2,2)):\033[0m

    agent 1 total reward      = {total1}   LIVE_FACTS says (2,2) -> 7   {'MATCH' if total1 == 7 else 'MISMATCH'}
    collapse step             = {collapse_step}    LIVE_FACTS says ~round 4     {'MATCH' if collapse_step == 4 else 'MISMATCH'}
    steps kept for PPO        = {kept}/{game.t_max}
    steps masked away         = {game.t_max - kept}/{game.t_max} = {(game.t_max-kept)/game.t_max:.0%}

  \033[1mThree things to understand here:\033[0m

  1. \033[1mSTEP 4 IS SCARCITY, NOT EXACT DEPLETION.\033[0m R_start = 2, both ask for 2, so
     4 > 2. Cap = 1; each receives 1; remainder wasted; R = 0. Exact depletion
     (a1+a2 == R, scarce=False) still exists as a rule and has its own test, but
     this (2,2) path hits scarcity first.

  2. \033[1mSTEP 4 IS NOT MASKED.\033[0m It was a real decision against a live pool, so it
     trains. Masking starts at step 5, where R_start == 0.

  3. \033[1mMASKED STEPS BECOME nan, AND nan MEANS DELETED.\033[0m CPRGame writes nan into
     the reward tensor; TrajectoryData._update drops query, response, id AND reward.

     Those steps carry 0 reward and 0 continuation, so deleting them changes no GAE
     target. What it costs is signal — {(game.t_max-kept)/game.t_max:.0%} of this
     (2,2) rollout teaches the model nothing.
""")


# ============================================================================ ACT 3

def act3():
    banner("ACT 3", "x5 EPISODES = ONE TRIAL, AND WHAT MAKES A SHAPER A SHAPER",
           "environment.py:296 outer_rollout | finetuning_cpr.py:77")
    print("""
 Agent 1 is naive, agent 2 is the shaper. Both play the same actions the whole time.
 The ONLY differences between them are (a) when PPO is called and (b) what their
 prompt remembers. Watch both.
""")
    game = make_game(shaper_is_agent2=True, e_max=5, n_games=1)
    obs1 = [game.obs_managers["agent_1"].game_description
            + game.obs_managers["agent_1"].instruction_prompt]
    obs2 = [game.obs_managers["agent_2"].game_description
            + game.obs_managers["agent_2"].instruction_prompt]
    env_state = EnvState(inner_t=0, outer_t=0)

    snapshot = None
    for episode in range(game.e_max):
        for t in range(game.t_max):
            result = game.step(obs1, obs2, fake_llm([2]), fake_llm([2]), env_state)
            obs1, obs2, env_state = result.new_obs1, result.new_obs2, result.new_env_state

        print(f"\n  \033[1m--- end of episode {episode+1} "
              f"(env_state: inner_t={env_state.inner_t}, outer_t={env_state.outer_t})\033[0m")
        print("      environment.py:308  agent 1 is NOT a shaper -> "
              "\033[33mPPO UPDATE NOW\033[0m, then its trajectory is reset")
        if episode < game.e_max - 1:
            print("      environment.py:308  agent 2 IS a shaper    -> no update; "
                  "keeps accumulating")
        else:
            print("      finetuning_cpr.py trial over               -> "
                  "\033[33mSHAPER PPO UPDATE NOW\033[0m (once, over all 180 steps)")

        # Grab the prompts entering the LAST episode of the trial — that is where the
        # shaper's memory is deepest. One step later `outer_t` wraps to 0 and
        # cpr_observation_managers.py:210 calls reset_trial(), wiping it.
        if episode == game.e_max - 2:
            snapshot = (obs1[0], obs2[0])

    naive_prompt, shaper_prompt = snapshot
    print("\n\n  \033[1mThe naive agent's prompt entering episode 5:\033[0m\n")
    show_prompt(naive_prompt)
    print("\n  \033[1mThe shaper's prompt at that same moment:\033[0m\n")
    show_prompt(shaper_prompt)
    print("""
  (If you run the trial one step further, the shaper's extra lines vanish — outer_t
   wraps to 0 and reset_trial() clears the memory. Trial memory does NOT survive a
   trial boundary. That is deliberate: each trial is a fresh meta-episode.)""")

    print("""
  \033[1mThat gap is the whole thesis mechanism.\033[0m

  The shaper's prompt carries two extra lines (cpr_observation_managers.py:137, :141):
  the joint-request counts, and a summary of how every episode this trial ended. The
  naive agent's prompt resets to bare rules every episode — it cannot even tell which
  episode it is in.

  So the two agents differ along exactly two axes:

                            naive (agent 1)        shaper (agent 2)
    PPO update every        episode (36 steps)     trial (180 steps)
    GAE recursion spans     36 steps               180 steps
    prompt remembers        nothing across eps     counts + episode outcomes

  \033[1mWhy those two axes, in theory:\033[0m

  A naive learner maximises its own return with the opponent held FIXED:

        theta_i  <-  theta_i + alpha * grad_i R_i(theta_i, theta_j)

  A shaper instead treats the opponent's LEARNING as part of the environment. A trial
  is E episodes during which the opponent updates between each one, and the shaper
  maximises the whole trial:

        max over theta_s of   sum over e=1..E of  R_s(theta_s, theta_o^(e))
        where                 theta_o^(e+1) = theta_o^(e) + eta * grad R_o

  You cannot optimise that objective unless (a) your credit assignment reaches across
  episode boundaries — hence the trial-length GAE — and (b) your policy can observe
  where the opponent currently is in its learning — hence the trial memory in the
  prompt. Those are precisely the two axes above. That's M-FOS / ShapeLLM.
""")


# ============================================================================ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--act", type=int, choices=[1, 2, 3], help="run only one act")
    args = p.parse_args()

    acts = {1: act1, 2: act2, 3: act3}
    for n in ([args.act] if args.act else [1, 2, 3]):
        acts[n]()

    if not args.act:
        banner("THE WHOLE SPINE", "one line, seven stops")
        print("""
   1. prompt string                cpr_observation_managers.py
   2. string -> tokens             agents.py:119
   3. tokens -> one masked token   agents.py:148
   4. token -> int                 environment.py:89
   5. int -> game rules -> R       cpr_env.py            locked logistic
   6. -> reward + next prompt      cpr_game.py
   7. -> save the triple           environment.py:62

   x36  = episode    environment.py:263   naive agent updates here
   x5   = trial      environment.py:296   shaper updates here (finetuning_cpr.py)

   Everything else in the repo is inherited from the paper's codebase, unchanged.
""")


if __name__ == "__main__":
    main()
