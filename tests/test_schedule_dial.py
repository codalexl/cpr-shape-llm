"""The phase scheduler: job counts match the pre-registration's budget, every job names a config and the inputs its
launcher requires, the folders it writes are the folders the evaluator reads, and the queue keeps one process per GPU."""
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evaluate_dial as ed
import schedule_dial as sd

PHASES = sd.phases()
folders = lambda phase: {Path(j.folder).name for group in PHASES[phase] for j in group}


def test_job_counts_match_the_budget():
    counts = {phase: [len(group) for group in groups] for phase, groups in PHASES.items()}
    assert counts["gate"] == [7] and counts["train"] == [38]  # Section 10: 6 + 1 gate runs, 38 training runs
    assert counts["evaluate"] == [52 + 38, 26]  # E1-E4, then 64 probes: training arms first, E1 and E2 learners after
    assert counts["optional"] == [3, 3] and counts["smoke"] == [13, 15]
    g3 = [j for j in PHASES["gate"][0] if j.name == "m2_shaper_matched"]
    assert len(g3) == 1 and g3[0].vars["EPOCHS"] == "200" and PHASES["gate"][0][0] == g3[0]
    long = sd.phases(200)
    assert [len(g) for g in long["train"]] == [34] and [len(g) for g in long["evaluate"]] == [44 + 34, 22]
    assert [(j.name, j.seed) for j in long["train"][0][-4:]] == [("m2_shaper_matched", 3), ("m2_tbn_matched", 3),
                                                                ("m2_shaper_matched", 4), ("m2_tbn_matched", 4)]
    assert [(j.name, j.seed, j.vars["EPOCHS"]) for j in long["extra"][0]] == [("m2_shapellm", 3, "200"), ("m2_shapellm", 4, "200")]
    assert len(long["extra"][1]) == 2 and PHASES["extra"] == [[], []]
    assert all(j.vars["EPOCHS"] == j.vars["CKPT_FREQ"] == "200" for j in long["train"][0])
    templates = [v for j in long["evaluate"][0] for k, v in j.env if k.endswith("_TEMPLATE") and "e1_transfer" not in v and "e2_replay" not in v]
    assert templates and all(v.endswith(("checkpoint_200", "cpr_records")) for v in templates)
    assert all(v.endswith("checkpoint_100") for j in long["evaluate"][1] for k, v in j.env if k.endswith("_TEMPLATE"))
    train = PHASES["train"][0]
    assert {j.name for j in train[:4]} == {"m2_naive", "m2_tbn_matched", "m2_shaper_matched", "m2_shapellm"}
    assert [j.name for j in train[:2]] == ["m2_shaper_matched", "m2_tbn_matched"]  # the decisive contrast leads each seed
    assert all(j.seed == 0 for j in train[:10]) and len(set(train)) == 38


def test_every_job_has_a_config_and_its_launcher_inputs():
    for groups in PHASES.values():
        for j in (j for group in groups for j in group):
            v = j.vars
            assert os.path.exists(os.path.join(ROOT, "configs", "dial", f"{j.name}.json")), j.name
            needs = {"_e1_transfer_": {"PARTNER_ADAPTER_TEMPLATE"}, "_e2_replay_": {"REPLAY_RECORDS_TEMPLATE"},
                     "_e3_frozen_partner_": {"PARTNER_ADAPTER_TEMPLATE", "LEARNER_ADAPTER_TEMPLATE"},
                     "_e4_untrained_partner_": {"PARTNER_ADAPTER_TEMPLATE"}, "_probe": {"LEARNER_ADAPTER_TEMPLATE", "SUFFIX"}}
            for key, required in needs.items():
                if key in j.name:
                    assert required <= set(v), j.label
            assert all("%d" in v[k] for k in v if k.endswith("_TEMPLATE")), j.label
            assert v.get("SMOKE") == "1" or ("EPOCHS" in v and "CKPT_FREQ" in v), j.label


def test_folders_written_are_the_folders_the_evaluator_reads():
    assert folders("gate") == {folder for folder, _, _ in ed.GATE.values()}
    assert set(ed.decision_folders()) <= folders("evaluate")
    evaluated = {Path(t.split("/exp%d_")[0]).name for group in PHASES["evaluate"] for j in group
                 for k, t in j.env if k.endswith("_TEMPLATE")}
    assert evaluated <= folders("train") | folders("evaluate")  # every adapter read is written by training or E1/E2
    assert "m2_probe_of_m2_shapellm_smoke" in folders("smoke") and "m2_e2_replay_shapellm_smoke" in folders("smoke")


class FakeProcess:
    running = {}

    def __init__(self, args, cwd, env, start_new_session):
        name, seed, gpu = args[1], int(args[2]), args[3]
        assert env["WAIT"] == "1" and gpu not in FakeProcess.running.values()
        FakeProcess.running[id(self)] = gpu
        self.polls, self.returncode = 0, None
        suffix = env.get("SUFFIX", "") + ("_smoke" if env.get("SMOKE") == "1" else "")
        self.records = Path(sd.ROOT) / sd.DIAL / f"{name}{suffix}" / f"exp{seed + 1}_cpr_records"

    def poll(self):
        self.polls += 1
        if self.polls < 2:
            return None
        self.records.parent.mkdir(parents=True, exist_ok=True)
        self.records.write_text("{}")
        FakeProcess.running.pop(id(self), None)
        self.returncode = 0
        return 0


def test_queue_runs_every_job_once_per_free_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sd, "ROOT", tmp_path)
    monkeypatch.setattr(sd.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sd, "busy_gpus", lambda: {3})
    monkeypatch.setattr(sd.time, "sleep", lambda s: None)
    gate = PHASES["gate"]
    assert sd.run(gate, gpus=[0, 3, 4]) == []
    out = capsys.readouterr().out
    assert out.count("started") == 7 and "GPU 3" not in out
    assert all(j.done() for j in gate[0])
    sd.run(gate, gpus=[0])  # a rerun skips finished jobs
    assert "0 job(s) to run, 7 already done" in capsys.readouterr().out
