import shutil

import numpy as np
import pytest

from tenpercent.constants import NEW_INTENTS
from tenpercent.feedback import save_snapshot
from tenpercent.io import jsonl_bytes, sha256_file, write_json
from tenpercent.memory import FeedbackMemory, MemoryConfig
from tenpercent.validation import require_final_ready, standalone_validate


def test_all_states_lineage_isolation_synthetic_final_and_immutable_proof(tmp_path):
    from test_bundle import tiny_bundle

    model, tokenizer, bundle, source, _ = tiny_bundle(tmp_path)
    base = tmp_path / "assets/base"
    base.mkdir(parents=True)
    model.config.save_pretrained(base)
    tokenizer.save_pretrained(base)
    shutil.copyfile(source, base / "model.safetensors")
    state_a = tmp_path / "runs/bundles/A"
    state_a.parent.mkdir(parents=True)
    bundle.rename(state_a)
    shutil.copytree(state_a / "loader/tenpercent", tmp_path / "src/tenpercent")
    config = MemoryConfig(input_dim=8, dimensions=16, layer=1, blend=1)
    memory = FeedbackMemory(["a", "b"] + list(NEW_INTENTS), config)
    for name, count in (("1-shot", 12), ("5-shot", 60), ("B", 120)):
        memory.add(name, name, NEW_INTENTS[0], np.ones(8))
        save_snapshot(tmp_path, state_a, name, memory, count)
    info = {}
    rows = {
        "banking/train": [{"id": "train:0", "text": "longest training example", "label": "a"}],
        "banking/working": [{"id": "working:0", "text": "old working example", "label": "b"}],
        "clinc/validation": [
            {"id": f"validation:{label}", "text": label, "label": label} for label in NEW_INTENTS
        ],
    }
    for index, (split, values) in enumerate(rows.items()):
        path = tmp_path / f"data/{index}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(jsonl_bytes(values))
        info[split] = {
            "path": path.name,
            "count": len(values),
            "sealed": False,
            "sha256": sha256_file(path),
        }
    write_json(tmp_path / "data/manifest.json", {"labels": ["a", "b"], "files": info})
    proof = standalone_validate(tmp_path, device="cpu", expected_source_sha256=sha256_file(source))
    assert proof["passed"] and proof["synthetic_final"]["completed"]
    assert proof["synthetic_final"]["new_rows"] == 12
    assert all(
        report["lineage"]["mismatches"] == report["mismatches"] == 0
        for report in proof["reports"].values()
    )
    assert not (tmp_path / "runs/exam_receipt.json").exists()
    assert (
        standalone_validate(tmp_path, device="cpu", expected_source_sha256=sha256_file(source))
        == proof
    )
    with pytest.raises(PermissionError, match="not released"):
        require_final_ready(tmp_path, device="cpu")
    source_code = next((tmp_path / "src/tenpercent").glob("*.py"))
    source_code.write_text(source_code.read_text() + "\n# altered\n")
    with pytest.raises(PermissionError, match="no longer matches"):
        standalone_validate(tmp_path, device="cpu", expected_source_sha256=sha256_file(source))
