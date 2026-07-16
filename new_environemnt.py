"""
Iterated Rock-Paper-Scissors (3×3) for the ShapeLLM pipeline.

Minimal extension beyond 2×2 matrix games:
  - one extra legal action token (R / P / S)
  - 3×3 payoff matrix
  - 9-bin state-visitation counts in the prompt

Everything else (trial structure, PPO+QLoRA loop, NL observations) stays the same.
Use with configs/rps_shaping.json and the generalized IteratedMatrixGame.
"""

from typing import Dict, List, Tuple

# Gemma-2 single-character token IDs (google/gemma-2-2b-it)
RPS_TOKEN_IDS = {
    "a1_tok": 235294,  # R
    "a2_tok": 235295,  # P
    "a3_tok": 235277,  # S
}

RPS_ACTION_STRINGS: Tuple[str, str, str] = ("R", "P", "S")

# Zero-sum RPS. Shape (2, 3, 3): [player][own_action][opp_action]
# Rows/cols ordered R, P, S.
# Both players share the same skew-symmetric matrix M (M = -Mᵀ). The env looks up
# player 2 via (own=a2, opp=a1), so skew-symmetry yields r2 = -r1 automatically.
RPS_R_MATRIX: List[List[List[float]]] = [
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
]

# 9 outcome bins in row-major own×opp order (matches environment outcome indices 0..8)
RPS_HISTORY_LABELS: Tuple[str, ...] = (
    "RR", "RP", "RS",
    "PR", "PP", "PS",
    "SR", "SP", "SS",
)


def rps_history_counts() -> Dict[str, int]:
    """Empty 9-bin visitation counter (same format as the 2×2 CC/CD/… summary)."""
    return {label: 0 for label in RPS_HISTORY_LABELS}


def format_rps_history(counts: Dict[str, int]) -> str:
    """Format visitation counts as 'RR: 2, RP: 1, …' for prompts / logging."""
    return ", ".join(f"{label}: {counts.get(label, 0)}" for label in RPS_HISTORY_LABELS)


def rps_obs_manager_fields(is_shaper: bool, perspective: str = "p1") -> Dict:
    """
    Fields for ObservationManagerConfig for iterated RPS.

    perspective='p1' uses RPS_R_MATRIX as-is;
    perspective='p2' swaps players so each agent sees own payoffs first (ShapeLLM convention).
    """
    if perspective == "p1":
        reward_matrix = RPS_R_MATRIX
    elif perspective == "p2":
        reward_matrix = [RPS_R_MATRIX[1], RPS_R_MATRIX[0]]
    else:
        raise ValueError(f"perspective must be 'p1' or 'p2', got {perspective!r}")

    return {
        **RPS_TOKEN_IDS,
        "a1_string": "R",
        "a2_string": "P",
        "a3_string": "S",
        "reward_matrix": reward_matrix,
        "is_shaper": is_shaper,
        "game_description": "table",
        "instruction_prompt": (
            "\nChoose an action for the current round. "
            "Reply only with {a1_tok}, {a2_tok}, or {a3_tok}."
        ),
        "model_name": "gemma-2b",
        "additional_info_type": "state_occurrence" if is_shaper else "state_only",
    }
