"""
Prompt construction for the deterministic CPR game.

Deliberately NOT an extension of observation_managers.ObservationManager. That class
rewrites the previous prompt string with regular expressions, which has three failure
modes this game triggers immediately:

  * `_update_state` substitutes with a letters-only pattern (`[A-Za-z]+`), so numeric
    action strings never match and the previous-round line silently freezes at round 1
    for the whole episode.
  * `_extract_current_counts` reads counts back out of the prompt with `:(\\d+)`; a
    resource line written without a space after the colon shifts every count by one.
  * `state_prompt` is interpolated into a regex unescaped, so a parenthesis in the
    template makes count extraction return nothing.

All three are artefacts of storing state *in the prompt*. Here every prompt is rendered
from structured state held on this object, and **no prompt string is ever parsed**. That
is the invariant; keep it.

Duck-typed surface used by the reused rollout machinery (environment.inner_rollout /
outer_rollout):

  * `game_description` and `instruction_prompt` — concatenated by those functions to
    build the reset observation. `game_description` therefore ends with the initial
    resource line, which is correct because every reset starts at R0.
  * `is_shaper` — decides whether trial memory survives an episode boundary.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RULES = (
    "You are interacting with another agent. Each round you choose how many units to take "
    "from a shared resource ({actions}). You receive exactly the number of units you "
    "successfully take. If the two requests together exceed the resource, both of you are "
    "cut back and the remainder is lost. The resource regrows over time, but once it reaches "
    "zero it never recovers. Your goal is to maximise the total units you collect."
)

DEFAULT_INSTRUCTION = "\nReply with only one number: {actions}."

# Stage B prompt clause. Do not splice in until cpr_preflight.py has been run on
# both the deterministic rules and rules+this sentence.
XI_GROWTH_CLAUSE = " Regrowth varies by up to 30 percent each round."


@dataclass
class CPRObservationManagerConfig:
    """Configuration for one agent's CPR prompts."""
    action_toks: List[int]
    action_strings: List[str]
    is_shaper: bool
    R0: int = 20
    rules: str = DEFAULT_RULES
    instruction: str = DEFAULT_INSTRUCTION
    formatting_tags: Dict = field(default_factory=lambda: {
        "start_user_tag": "<start_of_turn>user\n",
        "start_model_tag": "\n<start_of_turn>model\n",
        "end_tag": "\n<end_of_turn>",
    })
    model_name: str = "gemma-2b"
    transmit_info: bool = True

    def __post_init__(self):
        assert len(self.action_toks) == len(self.action_strings) >= 2, (
            f"need matching action_toks/action_strings of length >= 2; "
            f"got {len(self.action_toks)} and {len(self.action_strings)}"
        )
        assert len(set(self.action_toks)) == len(self.action_toks), "duplicate action token ids"


class CPRObservationManager:
    """Renders CPR prompts from structured state. Never parses a prompt."""

    def __init__(self, config: CPRObservationManagerConfig, n_games: int):
        self.config, self.n_games = config, n_games
        self.action_strings = list(config.action_strings)
        self.n_actions = len(self.action_strings)
        self.n_bins = self.n_actions ** 2
        self.is_shaper = config.is_shaper
        self.transmit_info = config.transmit_info
        self.R0 = config.R0

        # Row-major over (own, opp), matching environment._build_action_pair_lookup and
        # CPRGame.outcomes: index = own * n_actions + opp.
        self.count_labels = [f"{a}{b}" for a in self.action_strings for b in self.action_strings]

        actions_phrase = ", ".join(self.action_strings[:-1]) + f" or {self.action_strings[-1]}"
        tags = config.formatting_tags
        rules_body = config.rules.format(actions=actions_phrase)

        # gemma-2-it is turn-tagged; dropping the tags puts the model off-template and
        # shifts exactly the digit priors we log before training.
        if config.model_name == "gemma-2b":
            self._rules = tags["start_user_tag"] + rules_body
        else:
            self._rules = rules_body + tags["start_user_tag"]
        self.instruction_prompt = (
            config.instruction.format(actions=actions_phrase)
            + tags["end_tag"] + tags["start_model_tag"]
        )
        self.reset_trial()

    # ------------------------------------------------------------------ trial memory

    def reset_trial(self) -> None:
        """Clear trial-scoped memory. Counts and episode summaries do not survive a trial."""
        self.counts = np.zeros((self.n_games, self.n_bins), dtype=np.int64)
        self.episode_summaries: List[List[str]] = [[] for _ in range(self.n_games)]

    @property
    def game_description(self) -> str:
        """Rules plus the initial resource line.

        `outer_rollout` and `TrajectoryData._reset` build reset observations as
        `game_description + instruction_prompt`, one shared string for all parallel games.
        That is correct here precisely because every reset starts at R0.
        """
        return f"{self._rules}\n{self._resource_line(self.R0)}"

    # ------------------------------------------------------------------ rendering

    def _resource_line(self, R: int) -> str:
        return f"Current resource level: {int(R)}."

    def _previous_round_line(self, own_req, opp_req, own_recv, opp_recv) -> str:
        """Requests and receipts together.

        Under scarcity these differ, and reporting only the request would leave the prompt
        contradicting the reward with nothing to explain the gap — the scarcity rule would
        be uninferable from observation.
        """
        return (
            f"In the previous round you requested {self.action_strings[own_req]} and received "
            f"{int(own_recv)}; the other agent requested {self.action_strings[opp_req]} and "
            f"received {int(opp_recv)}."
        )

    def _counts_line(self, game: int) -> str:
        joined = ", ".join(f"{lab}: {c}" for lab, c in zip(self.count_labels, self.counts[game]))
        return f"Joint request counts so far this trial: {joined}."

    def _episodes_line(self, game: int) -> str:
        if not self.episode_summaries[game]:
            return ""
        return "Episodes so far this trial: " + "; ".join(self.episode_summaries[game]) + "."

    def _render(self, game: int, R: int, previous: Optional[str]) -> str:
        parts = [self._rules]
        if self.is_shaper and self.transmit_info:
            parts.append(self._counts_line(game))
            if episodes := self._episodes_line(game):
                parts.append(episodes)
        parts.append(self._resource_line(R))
        if previous is not None:
            parts.append(previous)
        return "\n".join(parts) + self.instruction_prompt

    def _reset_observations(self) -> List[str]:
        return [self.game_description + self.instruction_prompt] * self.n_games

    # ------------------------------------------------------------------ update

    def record_step(self, own_requests: Sequence[int], opp_requests: Sequence[int]) -> None:
        """Fold one round's joint requests into the trial counts."""
        for game, (own, opp) in enumerate(zip(own_requests, opp_requests)):
            self.counts[game, int(own) * self.n_actions + int(opp)] += 1

    def record_episode_end(self, collapse_steps: Sequence[Optional[int]],
                           final_resources: Sequence[int]) -> None:
        """Append one episode's outcome to the trial summary, per game."""
        for game, (collapse, final_R) in enumerate(zip(collapse_steps, final_resources)):
            index = len(self.episode_summaries[game]) + 1
            fate = "survived" if collapse is None else f"collapsed at round {int(collapse)}"
            self.episode_summaries[game].append(
                f"episode {index} {fate} (final resource {int(final_R)})"
            )

    def build_observations(
        self,
        resource: Sequence[int],
        own_requests: Sequence[int],
        opp_requests: Sequence[int],
        own_received: Sequence[int],
        opp_received: Sequence[int],
        inner_t: int,
        outer_t: int,
    ) -> List[str]:
        """Render the next observation for every parallel game.

        `resource` is the level the agent will face on the coming step. `inner_t`/`outer_t`
        are the *new* environment counters, matching IteratedMatrixGame.step's convention:
        `inner_t == 0` means the episode just ended.

        Call ordering, which CPRGame must respect: when `inner_t == 0`, call
        `record_episode_end` BEFORE this method, or the closing episode will be missing
        from the summary the shaper sees for the rest of the trial. The joint-request
        counts need no such care — this method folds in the round it is describing.
        """
        for name, seq in (("resource", resource), ("own_requests", own_requests),
                          ("opp_requests", opp_requests), ("own_received", own_received),
                          ("opp_received", opp_received)):
            if len(seq) != self.n_games:
                raise ValueError(f"{name} has length {len(seq)}, expected {self.n_games}")

        self.record_step(own_requests, opp_requests)

        if inner_t == 0:
            # Trial boundary: memory is cleared and everyone starts from the bare prompt.
            if outer_t == 0 or not self.is_shaper or not self.transmit_info:
                if outer_t == 0:
                    self.reset_trial()
                return self._reset_observations()
            # Episode boundary inside a trial: the shaper keeps its memory, but the
            # previous-round line is dropped because a fresh episode has no previous round.
            return [self._render(g, self.R0, previous=None) for g in range(self.n_games)]

        return [
            self._render(
                g, resource[g],
                self._previous_round_line(
                    int(own_requests[g]), int(opp_requests[g]),
                    own_received[g], opp_received[g],
                ),
            )
            for g in range(self.n_games)
        ]
