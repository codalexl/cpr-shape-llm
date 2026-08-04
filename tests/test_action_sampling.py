"""
Tests for PPOAgent._resample_illegal, the workaround for the MPS multinomial bug.

Measured on torch 2.13.0: with a correctly masked distribution (exactly len(legal)
non-zero entries, total mass 1.0), torch.multinomial on MPS still returns a
zero-probability index for ~0.03% of draws at a 1k vocabulary and ~0.9% at gemma's 256k
one. CPU and CUDA are exact. Over a 30-step episode with parallel games and two agents
that is a near-certain crash, so generation needs a rejection-sampling guard.

Skipped where the real `agents` module cannot be imported (it needs trl 0.11.4).

Runs under pytest, or standalone: `python tests/test_action_sampling.py`
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents import PPOAgent
    AGENTS_AVAILABLE = True
except Exception:                                # pragma: no cover - environment dependent
    AGENTS_AVAILABLE = False

LEGAL = [235276, 235274, 235284, 235304]


class FlakyTrainer:
    """Stands in for CustomPPOTrainer: emits a bad token on the first N draws per index."""

    def __init__(self, bad_draws_per_index):
        self.bad_left = dict(bad_draws_per_index)
        self.calls = 0
        self.generated = 0

    def generate(self, query_tensors, return_prompt=False, **kwargs):
        self.calls += 1
        self.generated += len(query_tensors)
        out = []
        for q in query_tensors:
            index = int(q.item())
            if self.bad_left.get(index, 0) > 0:
                self.bad_left[index] -= 1
                out.append(torch.tensor([140162]))   # 'Threats', an actual observed failure
            else:
                out.append(torch.tensor([LEGAL[index % len(LEGAL)]]))
        return out


def make_agent(trainer):
    agent = PPOAgent.__new__(PPOAgent)               # bypass the heavy model-loading __init__
    agent.legal_tokens = list(LEGAL)
    agent.max_sampling_retries = 4
    agent.trainer = trainer
    return agent


def run(bad_draws, n=4):
    trainer = FlakyTrainer(bad_draws)
    agent = make_agent(trainer)
    queries = [torch.tensor([i]) for i in range(n)]
    responses = trainer.generate(queries)
    return agent, trainer, queries, agent._resample_illegal(queries, responses, {})


def test_illegal_draw_is_redrawn():
    if not AGENTS_AVAILABLE:
        print("  [skip] agents unavailable (needs trl 0.11.4)"); return
    _, _, _, responses = run({2: 1})
    assert all(r.item() in LEGAL for r in responses), "an illegal draw survived"


def test_only_the_offending_entries_are_redrawn():
    """Redrawing the whole batch would resample healthy entries and waste generation."""
    if not AGENTS_AVAILABLE:
        print("  [skip] agents unavailable"); return
    _, trainer, _, responses = run({1: 1})
    assert all(r.item() in LEGAL for r in responses)
    assert trainer.generated == 4 + 1, f"expected 4 initial + 1 redraw, got {trainer.generated}"


def test_repeated_failures_are_retried_until_the_budget_runs_out():
    if not AGENTS_AVAILABLE:
        print("  [skip] agents unavailable"); return
    _, trainer, _, responses = run({0: 3})
    assert all(r.item() in LEGAL for r in responses)
    assert trainer.calls == 4, "one initial draw plus three retries"


def test_clean_batch_costs_nothing():
    if not AGENTS_AVAILABLE:
        print("  [skip] agents unavailable"); return
    _, trainer, _, responses = run({})
    assert all(r.item() in LEGAL for r in responses)
    assert trainer.calls == 1, "a clean batch must not trigger any retry"


def test_persistent_failure_still_reaches_the_assert():
    """If retries are exhausted the illegal token must survive, so take_action's assert
    fires rather than a corrupted action reaching the environment silently."""
    if not AGENTS_AVAILABLE:
        print("  [skip] agents unavailable"); return
    _, _, _, responses = run({3: 99})
    assert any(r.item() not in LEGAL for r in responses), \
        "exhausted retries must not silently substitute a legal action"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok ] {name}")
            passed += 1
    print(f"\nAll {passed} tests passed.")
