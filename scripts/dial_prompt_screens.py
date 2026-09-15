#!/usr/bin/env python3
"""Untrained gemma-2-2b-it priors for every stochastic-CPR prompt screened on 15 September (the amendment of that date
in docs/PREREGISTRATION_STOCHASTIC_CPR.md). Needs the model; runs on CUDA, MPS or CPU.

    python scripts/dial_prompt_screens.py > results/dial/prompt_screens.txt

Probabilities are renormalised over the legal labels and listed in take order. The first round has no previous-round
line; "stock only" is the same prompt at another stock; a history "you a, other b" is one previous round in which the
players requested takes a and b, with receipts from the scarcity rule.
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr_observation_managers import (DEFAULT_INSTRUCTION, DIAL_RULES, DIAL_RULES_V2,  # noqa: E402
                                      CPRObservationManager, CPRObservationManagerConfig)

MODEL = "google/gemma-2-2b-it"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
LETTER = "\nReply with only one letter: {actions}."
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
unit = lambda n: "unit" if n == 1 else "units"
define = lambda text, rules=DIAL_RULES: rules.replace("({actions}). ", "({actions}). " + text + " ")
fmt = lambda p: "/".join(f"{x:.2f}" for x in p)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").eval().to(DEVICE)


class Manager(CPRObservationManager):
    """The environment's prompt builder, with an optional replacement for the previous-round line."""

    def __init__(self, labels, rules, instruction, R0, line=None, history=True):
        self.line, self.toks = line, []
        for s in labels:
            ids = tok.encode(s, add_special_tokens=False)
            assert len(ids) == 1, (s, ids)
            self.toks.append(ids[0])
        super().__init__(CPRObservationManagerConfig(action_toks=self.toks, action_strings=list(labels), is_shaper=False,
                                                     R0=R0, rules=rules, instruction=instruction,
                                                     show_previous_round=history), n_games=1)

    def _previous_round_line(self, own_req, opp_req, own_recv, opp_recv):
        if self.line is None:
            return super()._previous_round_line(own_req, opp_req, own_recv, opp_recv)
        return self.line(self.action_strings[own_req], int(own_recv), self.action_strings[opp_req], int(opp_recv))


def dist(prompt, toks):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits[0, -1, :].float(), dim=-1)
    legal = probs[torch.tensor(toks, device=probs.device)]
    return (legal / legal.sum()).tolist()


class Screen:
    def __init__(self, labels, takes, rules=DIAL_RULES, instruction=DEFAULT_INSTRUCTION, R0=20, line=None):
        self.takes = takes
        self.hist = Manager(labels, rules, instruction, R0, line)
        self.bare = Manager(labels, rules, instruction, R0, line, history=False)
        self.toks, self.R0 = self.hist.toks, R0

    def first(self):
        return dist(self.hist.game_description + self.hist.instruction_prompt, self.toks)

    def stock(self, R):
        return dist(self.bare.build_observations([R], [0], [0], [1], [1], inner_t=5, outer_t=0)[0], self.toks)

    def after(self, R, i, j):
        a, b = self.takes[i], self.takes[j]
        got = (a, b) if a + b <= R else (min(a, R // 2), min(b, R // 2))
        return dist(self.hist.build_observations([R], [i], [j], [got[0]], [got[1]], inner_t=5, outer_t=0)[0], self.toks)

    def report(self, title, stocks, history_stocks, pairs=None):
        n = len(self.takes)
        pairs = pairs or [(i, j) for i in range(n) for j in range(n)]
        print(f"\n{title}: labels {self.hist.action_strings} for takes {self.takes}")
        print(f"  first round (R={self.R0}): {fmt(self.first())}")
        print("  stock only: " + "  ".join(f"R={R}: {fmt(self.stock(R))}" for R in stocks))
        for R in history_stocks:
            print(f"  R={R} after " + " | ".join(f"you {self.takes[i]}, other {self.takes[j]}: {fmt(self.after(R, i, j))}"
                                              for i, j in pairs), flush=True)


print(f"{MODEL} on {DEVICE}")
print("\n(1) Three takes: labellings")
Screen(["1", "2", "3"], [1, 2, 3]).report("digits (as pre-registered)", (20, 14, 8, 4), (14, 4))
Screen(["R", "H", "G"], [1, 2, 3], define("R means restrain and take 1 unit, H means harvest and take 2 units, and G means grab and take 3 units."),
       LETTER).report("R/H/G named restrain/harvest/grab", (20, 14, 8, 4), (14, 4))
Screen(["R", "H", "G"], [1, 2, 3], define("R takes 1 unit, H takes 2 units and G takes 3 units."), LETTER).report(
    "R/H/G, amounts only", (20, 14, 8, 4), (14, 4))
Screen(["A", "B", "C"], [1, 2, 3], define("A takes 1 unit, B takes 2 units and C takes 3 units."), LETTER).report(
    "A/B/C, amounts only", (20, 14, 8, 4), (14, 4))
Screen(["C", "B", "A"], [1, 2, 3], define("C takes 1 unit, B takes 2 units and A takes 3 units."), LETTER).report(
    "C/B/A, amounts only (C = 1 unit)", (20, 14, 8, 4), (14, 4))

print("\n(2) Three takes, digit replies: rewordings of the previous-round line (mean over all nine histories; P(repeat own take))")
LINES = {
    "W0 as pre-registered": None,
    "W1 amounts in words": lambda o, orc, p, prc: f"In the previous round you requested {WORDS[int(o)]} {unit(int(o))} and received {WORDS[orc]}; the other agent requested {WORDS[int(p)]} {unit(int(p))} and received {WORDS[prc]}.",
    "W2 digits with units": lambda o, orc, p, prc: f"In the previous round you requested {o} {unit(int(o))} and received {orc} {unit(orc)}; the other agent requested {p} {unit(int(p))} and received {prc} {unit(prc)}.",
    "W3 two short lists": lambda o, orc, p, prc: f"Previous round requests: you {o}, the other agent {p}. Units received: you {orc}, the other agent {prc}.",
    "W4 asked for / got, words": lambda o, orc, p, prc: f"Last round you asked for {WORDS[int(o)]} and got {WORDS[orc]}; the other agent asked for {WORDS[int(p)]} and got {WORDS[prc]}.",
    "W5 receipts only, words": lambda o, orc, p, prc: f"In the previous round you received {WORDS[orc]} {unit(orc)} and the other agent received {WORDS[prc]} {unit(prc)}.",
}
for name, line in LINES.items():
    s = Screen(["1", "2", "3"], [1, 2, 3], line=line)
    for R in (14, 4):
        rows = {(i, j): s.after(R, i, j) for i in range(3) for j in range(3)}
        mean = [sum(r[k] for r in rows.values()) / 9 for k in range(3)]
        repeat = sum(rows[(i, j)][i] for i, j in rows) / 9
        print(f"  {name:26s} R={R:2d}: after both took 1 {fmt(rows[(0, 0)])} | mean {fmt(mean)} | P(repeat own take) {repeat:.2f}", flush=True)

print("\n(3) Two takes")
Screen(["1", "2"], [1, 2]).report("digits 1/2, rules with the random-end sentence", (20, 14, 8, 4), (20, 14, 8, 4, 1))
Screen(["1", "2"], [1, 2], DIAL_RULES_V2).report("digits 1/2, amended rules (no random-end sentence)", (20, 14, 8, 4), (20, 14, 8, 4, 1))
Screen(["C", "D"], [1, 2], define("C takes 1 unit and D takes 2 units."), LETTER).report("C/D, amounts only", (20, 14, 8, 4), (14,))
Screen(["C", "D"], [1, 2], define("C means cooperate and take 1 unit, and D means defect and take 2 units."), LETTER).report(
    "C/D named cooperate/defect", (20, 14, 8, 4), (14,))

print("\n(4) Five and six digit takes (a pool of 40)")
Screen([str(k) for k in range(6)], list(range(6)), R0=40).report("digits 0-5", (40, 28, 16, 8), (28,), pairs=[(k, k) for k in (1, 2, 3, 5)])
Screen([str(k) for k in range(1, 6)], list(range(1, 6)), R0=40).report("digits 1-5", (40, 28, 16, 8), (28,), pairs=[(k - 1, k - 1) for k in (1, 2, 3, 5)])
