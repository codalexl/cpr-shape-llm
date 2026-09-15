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

from typing import Dict, List, Optional, Sequence, Tuple


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


def split_credit(rewards_by_round: Sequence[Sequence[float]], ids_by_round: Sequence[Sequence[int]], episodes: int,
                 weight: float = 1.0, baseline: Optional[Sequence[float]] = None) -> Tuple[List[int], List[float], List[float]]:
    """Split cross-episode credit for one shaper trial (PREREGISTRATION_STOCHASTIC_CPR.md, 15 September).

    Inputs are laid out as TrajectoryData stores a shaper's trial: one list per round, holding the reward and the game id
    of each live step; the rounds divide evenly into `episodes`. Returns, in the same flattened order:

    * env ids unique per (episode, game), so CustomPPOTrainer.compute_advantages runs GAE within each episode;
    * each step's cross-episode term, weight * (F[g][e] - baseline[e]), where F[g][e] is the shaper's return in game g's
      episodes after e and baseline[e] is the mean of F at position e over previous trials (None: the term is zero);
    * this trial's mean of F at each position over its games, for the caller's baseline history.

    The baseline has to come from earlier trials. The parallel games share one learner, so whatever this trial's actions do
    to the learner appears in every game's later returns; a baseline from this trial's other games would cancel it.
    """
    rounds = len(rewards_by_round)
    assert rounds % episodes == 0 and len(ids_by_round) == rounds, f"{rounds} rounds do not split into {episodes} episodes"
    per = rounds // episodes
    games = sorted({int(g) for ids in ids_by_round for g in ids})
    returns = {g: [0.0] * episodes for g in games}
    for r, (rw, ids) in enumerate(zip(rewards_by_round, ids_by_round)):
        assert len(rw) == len(ids), "each round's rewards and ids must align"
        for w, g in zip(rw, ids):
            returns[int(g)][r // per] += float(w)
    future = {g: [sum(returns[g][e + 1:]) for e in range(episodes)] for g in games}
    means = [sum(future[g][e] for g in games) / len(games) if games else 0.0 for e in range(episodes)]
    width = (max(games) + 1) if games else 1
    ids_out, term = [], []
    for r, ids in enumerate(ids_by_round):
        e = r // per
        for g in ids:
            ids_out.append(e * width + int(g))
            term.append(0.0 if baseline is None else weight * (future[int(g)][e] - baseline[e]))
    return remap_env_ids(ids_out), term, means


def first_index_by_id(ids: Sequence[int]) -> Dict[int, int]:
    """Opening step of each sequence in a flattened batch (pure twin of training_utils.first_index_by_env_id)."""
    first: Dict[int, int] = {}
    for i, gid in enumerate(ids):
        first.setdefault(int(gid), i)
    return first
