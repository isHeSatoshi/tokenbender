from types import SimpleNamespace

import pytest

from tenpercent import evaluation
from tenpercent.io import canonical


@pytest.mark.parametrize("torn_bytes", [0, 17, None])
def test_saved_pending_prediction_recovers_without_repeating_inference(
    tmp_path, monkeypatch, torn_bytes
):
    calls = []
    rows = [{"id": str(i), "text": str(i), "label": "a"} for i in range(3)]
    runner = SimpleNamespace(
        labels=["a"], device="cpu", predict=lambda text, mode: calls.append(text) or "a"
    )
    original = evaluation.append_record

    def interrupt(path, record):
        if record["example_id"] == "1":
            if torn_bytes is None:
                original(path, record)
            elif torn_bytes:
                with path.open("ab") as stream:
                    stream.write(canonical(record)[:torn_bytes])
            raise RuntimeError("power interrupted")
        original(path, record)

    monkeypatch.setattr(evaluation, "append_record", interrupt)
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(RuntimeError):
        evaluation.evaluate(runner, rows, output, mode="greedy", state_hash="fixed")
    monkeypatch.setattr(evaluation, "append_record", original)
    records, summary = evaluation.evaluate(runner, rows, output, mode="greedy", state_hash="fixed")
    assert calls == ["0", "1", "2"]
    assert summary["correct"] == 3 and len(records) == 3
    assert not output.with_suffix(".pending.json").exists()


def test_unrecoverable_or_unbound_journal_tail_is_refused(tmp_path):
    runner = SimpleNamespace(labels=["a"], device="cpu", predict=lambda *args, **kwargs: "a")
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b'{"example')
    with pytest.raises(ValueError, match="pending"):
        evaluation.evaluate(
            runner,
            [{"id": "0", "text": "x", "label": "a"}],
            output,
            mode="greedy",
            state_hash="fixed",
        )
