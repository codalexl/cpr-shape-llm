"""Every table opens chained arms from the uncut archive and the rest from checkpoints/dial."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from dial_records import HYBRID_ROOT, UNCUT_ROOT, is_uncut_tape, tape_dir

CHAINED = (
    "m2_shaper_matched",
    "m2_shaper_slow",
    "m2_shapellm",
    "m2_shapellm_history_off",
    "m3_shapellm",
)
ON_DIAL = (
    "m2_naive",
    "m2_slow",
    "m3_naive",
    "m3_slow",
    "m2_tbn_matched",
    "m2_tbn_slow",
    "m2_shaper_matched_split",
)


def test_uncut_routing_sends_the_five_chained_arms_to_dial_v1():
    for name in CHAINED:
        assert is_uncut_tape(name)
        assert tape_dir(name) == UNCUT_ROOT / name
        assert tape_dir(name, Path("/tmp/elsewhere")) == UNCUT_ROOT / name
    for name in ON_DIAL:
        assert not is_uncut_tape(name)
        assert tape_dir(name) == HYBRID_ROOT / name
    assert tape_dir("m2_probe_of_m2_shapellm") == UNCUT_ROOT / "m2_probe_of_m2_shapellm"
    assert tape_dir("m2_e1_transfer_shaper_matched") == UNCUT_ROOT / "m2_e1_transfer_shaper_matched"
    assert tape_dir("m2_e3_frozen_partner_shapellm") == UNCUT_ROOT / "m2_e3_frozen_partner_shapellm"
    assert tape_dir("m2_e4_untrained_partner_shaper_slow") == UNCUT_ROOT / "m2_e4_untrained_partner_shaper_slow"
    assert tape_dir("m2_probe_of_m2_e2_replay_shapellm") == UNCUT_ROOT / "m2_probe_of_m2_e2_replay_shapellm"
    assert tape_dir("m3_probe_of_m3_shapellm") == UNCUT_ROOT / "m3_probe_of_m3_shapellm"
    assert tape_dir("m2_probe_of_m2_naive") == HYBRID_ROOT / "m2_probe_of_m2_naive"
    assert tape_dir("m2_naive", Path("/tmp/elsewhere")) == Path("/tmp/elsewhere") / "m2_naive"
