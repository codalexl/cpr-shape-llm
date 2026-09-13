"""Trial-batched naive control (plan amendment of 13 Sep; thesis ch. 6 ladder rung "update schedule").

The shaper differs from a naive learner in two things at once: its advantages
pass across episode boundaries inside a trial (the shaping term), and it updates
once per trial from a five-episode batch. The trial-batched naive control keeps
the second and drops the first: it buffers E episodes, then takes one PPO step on
the concatenated batch with env ids made unique per (episode, game), so that the
per-game backward pass of CustomPPOTrainer.compute_advantages resets at every
episode boundary. Pure functions here; the agent hook is PPOAgent.update_parameters.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def remap_env_ids(ids: Sequence[int]) -> List[int]:
    """Map arbitrary hashable ids onto 0..n-1 preserving grouping (compute_advantages loops range(n))."""
    table = {v: i for i, v in enumerate(sorted(set(ids)))}
    return [table[v] for v in ids]


def concat_trial(buffer: Sequence[Tuple[list, list, list, list]]) -> Tuple[list, list, list, List[int]]:
    """Flatten a buffer of (queries, responses, rewards, env_ids) episodes.

    Env ids are offset by episode so that (episode, game) pairs are distinct,
    then remapped to 0..n-1. Returns (queries, responses, rewards, ids).
    """
    q, r, w, ids = [], [], [], []
    for e, (qe, re, we, ie) in enumerate(buffer):
        assert len(qe) == len(re) == len(we) == len(ie), "episode lists must align"
        q += list(qe); r += list(re); w += list(we)
        width = max((int(i) for i in ie), default=-1) + 1
        ids += [e * max(width, 1) + int(i) for i in ie]
    return q, r, w, remap_env_ids(ids)


def first_index_by_id(ids: Sequence[int]) -> Dict[int, int]:
    """Opening step of each sequence in a flattened batch (pure twin of training_utils.first_index_by_env_id)."""
    first: Dict[int, int] = {}
    for i, gid in enumerate(ids):
        first.setdefault(int(gid), i)
    return first
