"""Which round tapes a readout is allowed to open.

The five chained arms were trained with whole-trial GAE: at an episode join
the bootstrap is the value of the next episode. Those tapes, and the probes
and frozen evaluations of those policies, are the 18 September archive
``results/dial/pod_pull_20260918/dial_records_final.tgz`` (SHA-256
566ffbd9cbf8a804674bad017ff1ef28c0ef87b2fe23eb55e8466dca1db64b0d), extracted
to ``checkpoints/dial_v1``.

``checkpoints/dial`` under the same chained names is a later execution. It
sets that bootstrap to zero and keeps the lambda-chain. Arms that do not
chain were not rerun; their tapes in ``checkpoints/dial`` are the runs the
thesis reports.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "checkpoints" / "dial"
UNCUT_ROOT = ROOT / "checkpoints" / "dial_v1"

# Training arms whose credit chains across episodes.
CHAINED = {
    "m2_shaper_matched",
    "m2_shaper_slow",
    "m2_shapellm",
    "m2_shapellm_history_off",
    "m3_shapellm",
}


def is_uncut_tape(name: str) -> bool:
    """A chained training arm, or a probe or frozen evaluation of one."""
    if name in CHAINED:
        return True
    for arm in CHAINED:
        stage, short = arm.split("_", 1)
        if name == f"{stage}_probe_of_{arm}":
            return True
        for kind in ("e1_transfer", "e2_replay", "e3_frozen_partner", "e4_untrained_partner"):
            if name in (f"{stage}_{kind}_{short}", f"{stage}_probe_of_{stage}_{kind}_{short}"):
                return True
    return False


def tape_dir(name: str, other: Path | None = None) -> Path:
    """Directory of round tapes for ``name``.

    ``other`` is the root for arms that do not chain. Chained names ignore it
    and always open the uncut archive.
    """
    if is_uncut_tape(name):
        return UNCUT_ROOT / name
    return (HYBRID_ROOT if other is None else Path(other)) / name
