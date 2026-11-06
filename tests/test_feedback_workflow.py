from dataclasses import replace

import numpy as np
import pytest
import torch

from tenpercent import calibration, feedback
from tenpercent.constants import NEW_INTENTS
from tenpercent.io import read_json, sha256_file, write_json
from tenpercent.memory import FeedbackMemory, MemoryConfig, MemoryRunner


def test_prepare_calibrate_and_fresh_replay_use_only_permitted_data(tmp_path, monkeypatch):
    state_a = tmp_path / "runs/bundles/A"
    write_json(
        state_a / "manifest.json",
        {
            "state": "A",
            "feedback_events": 0,
            "mode": "greedy",
            "source_entries": 100,
            "model_hash": "frozen",
        },
    )
    write_json(tmp_path / "data/manifest.json", {"labels": ["old"]})
    teaching = [
        {"id": f"teach:{label}:{shot}", "text": f"new|{label}|{shot}", "label": label}
        for label in NEW_INTENTS
        for shot in range(10)
    ]
    validation = [
        {"id": f"validation:{label}", "text": f"new|{label}|validation", "label": label}
        for label in NEW_INTENTS
    ]
    working = [{"id": "working", "text": "old working", "label": "old"}]
    review = [
        {"id": "feedback:wrong", "text": "wrong feedback", "label": "old"},
        {"id": "feedback:correct", "text": "old correct", "label": "old"},
    ]
    datasets = {"banking/working": working, "clinc/train": teaching, "clinc/validation": validation}

    class Runner:
        labels = ["old"]
        output_labels = ["old"] + list(NEW_INTENTS)
        device = "cpu"

        def predict(self, text, mode):
            return "invalid" if text == "wrong feedback" else "old"

        def feature(self, text, layer):
            result = np.zeros(16, dtype=np.float32)
            result[NEW_INTENTS.index(text.split("|")[1]) + 1 if text.startswith("new|") else 0] = 1
            return result

        def features(self, text, layers):
            return {layer: self.feature(text, layer) for layer in layers}

        def label_logprobs(self, text):
            return torch.tensor([0.0])

    def load(path, device):
        if path == state_a:
            return Runner()
        return MemoryRunner(Runner(), FeedbackMemory.load(path / "memory.json"), "greedy")

    def split(root, name, purpose):
        assert purpose == "memory" and name in datasets
        return datasets[name]

    config = MemoryConfig(input_dim=16, layer=7, neighbors=1, blend=1)
    monkeypatch.setattr(
        calibration,
        "config_grid",
        lambda: [replace(config, radius=0.1), replace(config, radius=0.05)],
    )
    monkeypatch.setattr(
        calibration, "bundle_manifest", lambda path: read_json(path / "manifest.json")
    )
    monkeypatch.setattr(feedback, "bundle_manifest", lambda path: read_json(path / "manifest.json"))
    monkeypatch.setattr(calibration, "load_bundle", load)
    monkeypatch.setattr(feedback, "load_bundle", load)
    monkeypatch.setattr(calibration, "load_feedback", lambda *args: review)
    monkeypatch.setattr(calibration, "load_split", split)
    monkeypatch.setattr(feedback, "load_split", split)
    monkeypatch.setattr(calibration, "verify_public_commit", lambda *args: None)
    for mode in ("greedy", "constrained"):
        write_json(tmp_path / f"runs/baseline/working-1-{mode}.summary.json", {"correct": 1})
    prepared = calibration.prepare_feedback(tmp_path, state_a, device="cpu")
    assert prepared["old_corrections"] == 1
    assert prepared["events"] == 121
    assert b'"text":' not in (tmp_path / "feedback_stream.jsonl").read_bytes()
    write_json(
        tmp_path / "runs/feedback_attestation.json",
        {
            "manifest": "feedback_commitment.json",
            "manifest_sha256": sha256_file(tmp_path / "feedback_commitment.json"),
        },
    )
    result = calibration.calibrate_feedback(tmp_path, state_a, device="cpu")
    assert result["g3_calibration_pass"]
    report = read_json(tmp_path / result["report"])
    assert report["selected"]["config"]["radius"] == 0.05
    assert report["label_count"] == 13

    def snapshot(root, source, name, memory, count):
        path = root / "runs/bundles" / name
        memory.save(path / "memory.json")
        write_json(path / "manifest.json", {"state": name, "model_hash": "frozen"})
        return path

    monkeypatch.setattr(feedback, "save_snapshot", snapshot)
    before = sha256_file(state_a / "manifest.json")
    replay = feedback.replay_feedback(
        tmp_path, state_a, tmp_path / result["config"], tmp_path / result["report"], device="cpu"
    )
    assert replay["g3_pass"]
    assert replay["checkpoints"]["B"]["new"]["correct"] == 12
    assert replay["checkpoints"]["B"]["old"]["correct"] == 1
    assert replay["checkpoints"]["after-corrections"]["corrections_stored"] == 1
    assert replay["a_new"]["correct"] == 0
    assert sha256_file(state_a / "manifest.json") == before
    assert (
        feedback.replay_feedback(
            tmp_path,
            state_a,
            tmp_path / result["config"],
            tmp_path / result["report"],
            device="cpu",
            resume=True,
        )
        == replay
    )
    bad = read_json(tmp_path / result["config"])
    bad["radius"] = 0.3
    write_json(tmp_path / "different_config.json", bad)
    with pytest.raises(PermissionError, match="not selected"):
        feedback.replay_feedback(
            tmp_path,
            state_a,
            tmp_path / "different_config.json",
            tmp_path / result["report"],
            device="cpu",
        )


def test_real_snapshot_is_immutable_counted_and_source_bit_identical(tmp_path):
    from test_bundle import tiny_bundle

    from tenpercent.bundle import bundle_manifest, verify_lineage

    _, _, state_a, source, _ = tiny_bundle(tmp_path)
    config = MemoryConfig(input_dim=8, dimensions=16, layer=1)
    memory = FeedbackMemory(["a", "b"] + list(NEW_INTENTS), config)
    memory.add("one", "synthetic exemplar", NEW_INTENTS[0], np.ones(8))
    before = sha256_file(state_a / "manifest.json")
    destination = feedback.save_snapshot(tmp_path, state_a, "1-shot", memory, 12)
    manifest = bundle_manifest(destination)
    assert manifest["auxiliary_entries"] == memory.numeric_entries
    assert manifest["retained_entries"] == manifest["source_entries"] + memory.numeric_entries
    assert (
        verify_lineage(destination, source, expected_sha256=sha256_file(source))["mismatches"] == 0
    )
    assert feedback.save_snapshot(tmp_path, state_a, "1-shot", memory, 12) == destination
    assert sha256_file(state_a / "manifest.json") == before
    memory.config = replace(config, radius=0.3)
    with pytest.raises(PermissionError, match="differs"):
        feedback.save_snapshot(tmp_path, state_a, "1-shot", memory, 12)
