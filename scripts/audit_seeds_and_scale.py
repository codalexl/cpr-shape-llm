#!/usr/bin/env python3
"""Two audits of the executed runs that the Method and Limitations chapters cite.

1. Seed identity.  TRL 0.11.4's PPOTrainer.__init__ calls set_seed(config.seed)
   with the PPOConfig default seed=0, after the launcher's own set_seed(seed).
   Every experiment therefore sampled actions from the seed-0 stream; runs that
   were nominally different seeds differ only through the CRN noise table
   (Stage B) and non-deterministic kernels.  This audit prints, per epoch, the
   fraction of the 15 opening actions that are identical between two runs.

2. Per-update advantage scale.  The batch standard deviation sigma_B of the raw
   GAE advantages, per PPO update, from the live_adv logs.  Whitening divides by
   this number; centring does not.  The thesis quotes the median and IQR.

Usage:  python scripts/audit_seeds_and_scale.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"


def load(p: Path):
    return json.loads(p.read_text())


def opening_identity(rec_a: dict, rec_b: dict, n_epochs: int = 15):
    """Per-epoch fraction of identical agent-1 openings between two record files."""
    out = []
    for e in range(n_epochs):
        def openings(rec):
            ep = np.array(rec["epoch"]); st = np.array(rec["step"]); rq = np.array(rec["request_1"])
            return rq[(ep == e) & (st == 1)]
        oa, ob = openings(rec_a), openings(rec_b)
        n = min(len(oa), len(ob))
        out.append(float((oa[:n] == ob[:n]).mean()) if n else float("nan"))
    return out


def sigma_b_per_update(live_adv_rows):
    """Median / IQR / min / max of the per-update std of raw advantages."""
    by = defaultdict(list)
    for r in live_adv_rows:
        by[(r["epoch"], r["update"])].append(r["A_raw"])
    sig = np.array([np.std(v) for v in by.values()])
    return dict(n_updates=len(by), median=float(np.median(sig)),
                q25=float(np.percentile(sig, 25)), q75=float(np.percentile(sig, 75)),
                min=float(sig.min()), max=float(sig.max()))


PAIRS = [
    ("Test A whitened, seeds 1 vs 2", "cpr_log_testA_whiten_s12/exp1_cpr_records", "cpr_log_testA_whiten_s12/exp2_cpr_records"),
    ("Test A centred, seeds 1 vs 2", "cpr_log_testA_center_s12/exp1_cpr_records", "cpr_log_testA_center_s12/exp2_cpr_records"),
    ("Test A centred, seeds 0 vs 1", "cpr_log_testA_center/exp1_cpr_records", "cpr_log_testA_center_s12/exp1_cpr_records"),
    ("Test A whitened vs centred, seed 0", "cpr_log_testA_a0/exp1_cpr_records", "cpr_log_testA_center/exp1_cpr_records"),
    ("Naive-naive, seeds 1 vs 2", "cpr_log_naive_naive_center_s12/exp1_cpr_records", "cpr_log_naive_naive_center_s12/exp2_cpr_records"),
    ("Naive-naive xi, seeds 0 vs 1", "cpr_log_naive_naive_center_xi/exp1_cpr_records", "cpr_log_naive_naive_center_xi/exp2_cpr_records"),
    ("Naive-shaper xi, seeds 0 vs 1", "cpr_log_naive_shaper_center_xi/exp1_cpr_records", "cpr_log_naive_shaper_center_xi/exp2_cpr_records"),
    ("Test A xi, seeds 0 vs 1", "cpr_log_testA_center_xi/exp1_cpr_records", "cpr_log_testA_center_xi/exp2_cpr_records"),
    ("Test A xi, seeds 1 vs 2", "cpr_log_testA_center_xi/exp2_cpr_records", "cpr_log_testA_center_xi/exp3_cpr_records"),
    ("Naive-naive xi, seeds 1 vs 2", "cpr_log_naive_naive_center_xi/exp2_cpr_records", "cpr_log_naive_naive_center_xi/exp3_cpr_records"),
    ("Naive-shaper xi, seeds 1 vs 2", "cpr_log_naive_shaper_center_xi/exp2_cpr_records", "cpr_log_naive_shaper_center_xi/exp3_cpr_records"),
    ("Slow-LR xi, seeds 1 vs 2", "cpr_log_naive_naive_slow2_center_xi/exp2_cpr_records", "cpr_log_naive_naive_slow2_center_xi/exp3_cpr_records"),
    # Seed-fixed packet (8–9 Sep). After set_seed(seed) post PPOAgent().
    ("RESEED Test A centre, seeds 0 vs 1", "cpr_log_testA_center_reseed/exp1_cpr_records", "cpr_log_testA_center_reseed/exp2_cpr_records"),
    ("RESEED Test A centre, seeds 1 vs 2", "cpr_log_testA_center_reseed/exp2_cpr_records", "cpr_log_testA_center_reseed/exp3_cpr_records"),
    ("RESEED Test A whitened, seeds 0 vs 1", "cpr_log_testA_whiten_reseed/exp1_cpr_records", "cpr_log_testA_whiten_reseed/exp2_cpr_records"),
    ("RESEED Test A whitened, seeds 1 vs 2", "cpr_log_testA_whiten_reseed/exp2_cpr_records", "cpr_log_testA_whiten_reseed/exp3_cpr_records"),
    ("RESEED NS xi, seeds 0 vs 1", "cpr_log_naive_shaper_center_xi_reseed/exp1_cpr_records", "cpr_log_naive_shaper_center_xi_reseed/exp2_cpr_records"),
    ("RESEED NS xi, seeds 1 vs 2", "cpr_log_naive_shaper_center_xi_reseed/exp2_cpr_records", "cpr_log_naive_shaper_center_xi_reseed/exp3_cpr_records"),
    ("RESEED slow2 xi, seeds 0 vs 1", "cpr_log_naive_naive_slow2_center_xi_reseed/exp1_cpr_records", "cpr_log_naive_naive_slow2_center_xi_reseed/exp2_cpr_records"),
    ("RESEED slow2 xi, seeds 1 vs 2", "cpr_log_naive_naive_slow2_center_xi_reseed/exp2_cpr_records", "cpr_log_naive_naive_slow2_center_xi_reseed/exp3_cpr_records"),
    ("RESEED vs pre-fix NS xi seed 0", "cpr_log_naive_shaper_center_xi_reseed/exp1_cpr_records", "cpr_log_naive_shaper_center_xi/exp1_cpr_records"),
    ("RESEED vs pre-fix slow2 xi seed 0", "cpr_log_naive_naive_slow2_center_xi_reseed/exp1_cpr_records", "cpr_log_naive_naive_slow2_center_xi/exp1_cpr_records"),
]

LIVE_ADV = [
    ("Test A whitened s1", "cpr_log_testA_whiten_s12/exp1_live_adv"),
    ("Test A whitened s2", "cpr_log_testA_whiten_s12/exp2_live_adv"),
    ("Test A centred s0", "cpr_log_testA_center/exp1_live_adv"),
    ("Test A centred s1", "cpr_log_testA_center_s12/exp1_live_adv"),
    ("Test A centred s2", "cpr_log_testA_center_s12/exp2_live_adv"),
    ("Test A centred xi s0", "cpr_log_testA_center_xi/exp1_live_adv"),
    ("Test A centred xi s1", "cpr_log_testA_center_xi/exp2_live_adv"),
    ("Test A centred xi s2", "cpr_log_testA_center_xi/exp3_live_adv"),
]


def main():
    print("== 1. Opening identity between nominally different seeds (fraction of 15 per epoch) ==")
    for name, a, b in PAIRS:
        pa, pb = CK / a, CK / b
        if not (pa.exists() and pb.exists()):
            print(f"  {name}: missing"); continue
        ident = opening_identity(load(pa), load(pb))
        print(f"  {name:38s} " + " ".join(f"{x:.2f}" for x in ident)
              + f"   | first 5 epochs mean {np.nanmean(ident[:5]):.2f}, whole run {np.nanmean(ident):.2f}")

    print("\n== 2. Per-update batch std of raw GAE advantages (sigma_B) ==")
    for name, p in LIVE_ADV:
        pp = CK / p
        if not pp.exists():
            print(f"  {name}: missing"); continue
        s = sigma_b_per_update(load(pp))
        print(f"  {name:22s} updates={s['n_updates']:3d}  median {s['median']:5.1f}  IQR [{s['q25']:4.1f}, {s['q75']:4.1f}]  min {s['min']:4.1f}  max {s['max']:4.1f}")

    print("\n== 3. First-episode (untrained) opening distribution, agent 1, pooled over runs ==")
    from collections import Counter
    c = Counter(); n = 0
    for p in sorted(list(CK.glob("*/exp*_opening_a0")) + list(CK.glob("*/exp*_model1_opening_a0"))):
        for r in load(p):
            if r["epoch"] == 0 and r["update"] == 1:
                c[r["action"]] += 1; n += 1
    print("  counts", dict(sorted(c.items())), "n =", n,
          "(not independent: runs on one platform share the seed-0 stream)")


if __name__ == "__main__":
    main()
