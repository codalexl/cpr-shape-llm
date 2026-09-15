#!/usr/bin/env python3
"""Why the 14 September gate said NO GO, and what restores a learnable restraint signal (docs/GATE_DIAGNOSIS.md).

    python scripts/dial_learnability.py

A learner holds its measured prior: a fixed mix of takes in every state. For each setting this prints the exact
expected advantage of one restraint sample (Q - V) and its standard deviation (Monte Carlo return minus the state
value, within and between states). From these it prints the epochs a naive learner (3 games, one update per episode,
5 episodes an epoch) needs before its summed restraint signal reaches two standard deviations. The pond's Test A,
which this training stack did learn, calibrates the scale. The script also prints the solver's invariants for
stationary play under each way of ending an episode, for the design pre-registered on 14 September. The last two
tables cover the amended design (takes 1-2 over a fixed horizon, five games): learnability at restraint priors in the
measured range, and both margins across horizons.
"""
from __future__ import annotations

import json
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cpr_dial as cd  # noqa: E402
from cpr_xi import harvest_and_grow  # noqa: E402

ENDS = {  # continuation probability after each round; the episode closes when it is not continued
    "random end (pre-registered)": [35 / 36] * 108,
    "fixed 36 rounds": [1.0] * 35 + [0.0],
    "24 + geometric(1/12)": [1.0] * 24 + [11 / 12] * 84,
}
HARVEST_BOT = lambda last: 2
TIT_FOR_TAT = lambda last: 1 if last == 1 else 2


def prior(restrain: float, grab: float = 0.10) -> dict:
    return {1: restrain, 2: 1 - grab - restrain, 3: grab}


def restraint_signal(K, rate, S0, prior_mix, partner, conts, actions=(1, 2, 3), xis=(7, 10, 13),
                     n_games=3, episodes_per_epoch=5) -> dict:
    """partner: a function of the learner's last take, or a fixed mix {take: probability}."""
    A = list(actions); ix = {a: i for i, a in enumerate(A)}; nA = len(A); nS = (K + 1) * nA
    s_of = lambda R, last: R * nA + ix[last]
    r, P = np.zeros((nS, nA)), np.zeros((nS, nA, nS))
    for R in range(1, K + 1):
        for last in A:
            s, mix = s_of(R, last), (partner if isinstance(partner, dict) else {partner(last): 1.0})
            for a in A:
                for b, pb in mix.items():
                    for xi in xis:
                        r1, _, Rn = harvest_and_grow(R, a, b, xi, K=K, rate_tenths=rate)
                        P[s, ix[a], s_of(Rn, a)] += pb / len(xis)
                    r[s, ix[a]] += pb * r1
    pi, k = np.array([prior_mix.get(a, 0.0) for a in A]), ix[1]
    V, W, Q, M = [np.zeros(nS)], [np.zeros(nS)], [], []
    for c in reversed(conts):  # values and second moments of the return, backwards over rounds
        EV, EW = P @ V[0], P @ W[0]
        q, m = r + c * EV, r ** 2 + 2 * r * c * EV + c * EW
        Q.insert(0, q); M.insert(0, m); V.insert(0, q @ pi); W.insert(0, m @ pi)
    d = np.zeros(nS); d[s_of(S0, 1 if 1 in A else A[0])] = 1.0  # round 0 counts as mutual restraint
    n = mu = mu2 = within = 0.0
    for t, c in enumerate(conts):
        w = d.copy(); w[:nA] = 0.0; w *= pi[k]  # restraint samples on live rounds; an empty pool is masked
        adv = Q[t][:, k] - V[t]
        n, mu, mu2 = n + w.sum(), mu + w @ adv, mu2 + w @ adv ** 2
        within += w @ (M[t][:, k] - Q[t][:, k] ** 2)
        d = c * np.einsum("s,a,sat->t", d, pi, P)
    mean = mu / n
    var = within / n + mu2 / n - mean ** 2
    samples_needed = 4 * var / mean ** 2 if mean > 0 else float("inf")
    return {"samples_per_game_episode": n, "advantage": mean, "sd": var ** 0.5,
            "epochs": samples_needed / (n * n_games) / episodes_per_epoch}


def evaluate_end(dial, policy1, policy2, conts):
    ch, P, r, alive = cd._play(dial, policy1, policy2)
    d = np.zeros(ch.n); d[ch.start] = 1.0
    returns, survival, reach = np.zeros(2), 0.0, 1.0
    for c in conts:
        returns += reach * (d @ r)
        survival += reach * (1 - c) * float(d @ alive)
        d, reach = d @ P, reach * c
    return float(returns[0]), float(returns[1]), survival


def main() -> None:
    pond = json.loads((ROOT / "configs" / "grid" / "B_testA_whiten.json").read_text())["game_parameters"]
    s = restraint_signal(pond["ceiling"], pond["rate_tenths"], pond["R0"], {0: 0.0, 1: 0.05, 2: 0.87, 3: 0.08},
                         HARVEST_BOT, [1.0] * (pond["t_max"] - 1) + [0.0], actions=(0, 1, 2, 3),
                         xis=tuple(pond["xi_tenths"]))
    print("Epochs for a two-SD restraint signal at the epoch-1 prior")
    print(f"  pond Test A (R0 {pond['R0']}, 36 rounds, vs take-2, restraint prior 0.05): advantage {s['advantage']:+.2f} "
          f"(SD {s['sd']:.1f}), {s['epochs']:.0f} epochs; the pond learner took off at epochs 21-30")
    for end, conts in ENDS.items():
        print(f"  {end}")
        for label, rate, partner in (("G1 m=3 vs take-2", 6, HARVEST_BOT), ("G2 m=2 vs tit-for-tat", 5, TIT_FOR_TAT)):
            cells = [restraint_signal(20, rate, 20, prior(p), partner, conts) for p in (0.03, 0.10, 0.20)]
            print(f"    {label:22s} advantage {cells[0]['advantage']:+.2f} (SD {cells[0]['sd']:.1f}); epochs at restraint prior "
                  f"0.03 / 0.10 / 0.20: " + " / ".join(f"{c['epochs']:.0f}" for c in cells))
        g3 = restraint_signal(20, 5, 20, prior(0.10), prior(0.10), conts)
        print(f"    {'G3-like m=2 vs prior':22s} advantage {g3['advantage']:+.2f} at restraint prior 0.10 for both players")
    print("\nInvariants for stationary play (best responses from the geometric analysis, evaluated under each end)")
    for end, conts in ENDS.items():
        for name, dial in cd.RANDOM_END.items():
            lead = lambda partner: evaluate_end(dial, cd.best_response(dial, partner)[1], partner, conts)[1]
            harvest, tft = lead(cd.always(cd.HARVEST)), lead(cd.tit_for_tat)
            worth = (evaluate_end(dial, cd.best_response(dial, cd.always(cd.RESTRAIN))[1], cd.always(cd.RESTRAIN), conts)[0]
                     - evaluate_end(dial, cd.best_response(dial, cd.always(cd.HARVEST))[1], cd.always(cd.HARVEST), conts)[0])
            survival = evaluate_end(dial, cd.always(cd.RESTRAIN), cd.always(cd.HARVEST), conts)[2]
            print(f"  {end:28s} {name}: committed harvest {harvest:4.1f} vs tit-for-tat {tft:4.1f} "
                  f"({'commitment fails' if harvest < tft else 'commitment wins'}); restrained partner worth {worth:4.1f}; "
                  f"restrain-vs-harvest survival {survival:.2f}")

    print("\nAmended design (takes 1-2, fixed horizon, 5 games): epochs for a two-SD restraint signal")
    conts = [1.0] * (cd.HORIZON - 1) + [0.0]
    for p in (0.06, 0.10, 0.15, 0.20, 0.30):
        mix = {1: p, 2: 1 - p}
        g1 = restraint_signal(20, 6, 20, mix, HARVEST_BOT, conts, actions=(1, 2), n_games=5)
        g2 = restraint_signal(20, 5, 20, mix, TIT_FOR_TAT, conts, actions=(1, 2), n_games=5)
        g3 = restraint_signal(20, 5, 20, mix, mix, conts, actions=(1, 2), n_games=5)
        print(f"  restraint prior {p:.2f}: G1 {g1['epochs']:.1f} (advantage {g1['advantage']:+.2f}) | G2 {g2['epochs']:.1f} "
              f"(advantage {g2['advantage']:+.2f}) | G3-like, partner at the same prior {g3['epochs']:.0f} "
              f"(advantage {g3['advantage']:+.2f})")
    print("\nAmended design: margins by horizon (stationary best responses over the fixed horizon)")
    for T in (30, 36, 40, 45, 50, 55, 60, 70, 80, 100):
        rows = {}
        for actions in ((1, 2), (1, 2, 3)):
            m2, m3 = (replace(cd.DESIGN[k], horizon=T, actions=actions) for k in ("m=2", "m=3"))
            rows[actions] = (cd.committed_leader(m2, cd.tit_for_tat)[0] - cd.committed_leader(m2, cd.always(cd.HARVEST))[0],
                             cd.committed_leader(m3, cd.always(cd.HARVEST))[0] - cd.committed_leader(m3, cd.tit_for_tat)[0],
                             cd.evaluate(m3, cd.always(cd.RESTRAIN), cd.always(cd.HARVEST))[2],
                             cd.best_response(m2, cd.always(cd.RESTRAIN))[0] - cd.best_response(m2, cd.always(cd.HARVEST))[0])
        two, three = rows[(1, 2)], rows[(1, 2, 3)]
        print(f"  T={T:3d}: two takes: m=2 margin {two[0]:+5.1f}, m=3 margin {two[1]:+5.1f}, m=3 lone restrainer survives "
              f"{two[2]:.2f}, m=2 restrained partner worth {two[3]:5.1f} | three takes: m=2 {three[0]:+5.1f}, m=3 {three[1]:+5.1f} "
              f"| cross-episode credit {0.97 ** T:.2f}", flush=True)

    print("\nWider action sets on a pool of 40 (rate 0.4, peak regrowth 4; geometric end with mean 50): restrain 2, harvest 3")
    tft = lambda stock, own, other: 2 if other <= 2 else 3
    for actions in ((0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5)):
        d = cd.Dial(K=40, rate_tenths=4, actions=actions, continuation=49 / 50)
        lone = ", ".join(f"take-{h} {cd.best_survival(d, h):.2f}" for h in (3, 4, 5))
        print(f"  takes {actions[0]}-{actions[-1]}: best lone survival against a committed {lone}; committed take-3 earns "
              f"{cd.committed_leader(d, cd.always(3))[0]:.1f} against {cd.committed_leader(d, tft)[0]:.1f} for tit-for-tat (2 after 2 or less, else 3)", flush=True)


if __name__ == "__main__":
    main()
