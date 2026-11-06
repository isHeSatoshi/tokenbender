import json
from types import SimpleNamespace

import numpy as np
import pytest

from tenpercent import feedback
from tenpercent.constants import NEW_INTENTS
from tenpercent.feedback import replay_events
from tenpercent.io import object_hash, read_json, write_json
from tenpercent.memory import FeedbackMemory, MemoryConfig, build_stream


def setup():
    config = MemoryConfig(input_dim=4, dimensions=16, neighbors=1, radius=0.05, blend=1)
    memory = FeedbackMemory(["old", "new"], config)
    runner = SimpleNamespace(
        labels=["old"],
        feature=lambda text, layer: (
            np.array([1.0, 0.0, 0.0, 0.0]) if text != "old" else np.array([-1.0, 0.0, 0.0, 0.0])
        ),
        predict=lambda text, mode: "old",
        label_logprobs=lambda text: (_ for _ in ()).throw(
            AssertionError("Blend one needs no model score")
        ),
    )
    events = [
        {"id": f"new:{i}", "text": f"new-{i}", "label": "new", "kind": "new"} for i in range(120)
    ]
    events.insert(12, {"id": "correction", "text": "old", "label": "old", "kind": "correction"})
    return memory, runner, events


def test_stream_is_nested_and_corrections_are_between_one_and_five_shot():
    teaching = [
        {"id": f"{label}:{shot}", "text": f"{label} {shot}", "label": label}
        for label in NEW_INTENTS
        for shot in range(10)
    ]
    events = build_stream(
        teaching,
        [{"id": "z", "text": "z", "label": "old"}, {"id": "a", "text": "a", "label": "old"}],
    )
    assert [e["label"] for e in events[:12]] == list(NEW_INTENTS)
    assert [e["id"] for e in events[12:14]] == ["a", "z"]
    assert [e["shot"] for e in events if e["kind"] == "new"] == [
        shot for shot in range(1, 11) for _ in NEW_INTENTS
    ]


def test_interrupted_replay_resumes_without_duplicate_events_or_updates(tmp_path):
    initial, runner, events = setup()
    snapshots = {}

    def checkpoint(name, memory, count):
        snapshots[name] = (len(memory.keys), count)

    def interrupt(cursor):
        if cursor == 18:
            raise RuntimeError("Interrupted")

    with pytest.raises(RuntimeError):
        replay_events(
            events,
            runner,
            initial,
            tmp_path,
            {"run": "a"},
            mode="greedy",
            source_entries=0,
            on_checkpoint=checkpoint,
            after_save=interrupt,
        )
    assert read_json(tmp_path / "replay_state.json")["next_event"] == 18
    final = replay_events(
        events,
        runner,
        initial,
        tmp_path,
        {"run": "a"},
        mode="greedy",
        source_entries=0,
        resume=True,
        on_checkpoint=checkpoint,
    )
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_bytes().splitlines()]
    assert len(rows) == len(events) == 121
    assert rows[0]["prediction_before_reveal"] == "old"
    assert rows[0]["true_label"] == "new"
    assert len(initial.keys) == 0 and len(final.keys) == 121
    assert snapshots["1-shot"] == (12, 12)
    assert snapshots["after-corrections"] == (13, 13)
    assert snapshots["5-shot"] == (61, 61)
    assert snapshots["B"] == (121, 121)
    again = replay_events(
        events,
        runner,
        initial,
        tmp_path,
        {"run": "a"},
        mode="greedy",
        source_entries=0,
        resume=True,
    )
    assert again.payload() == final.payload()
    with pytest.raises(PermissionError, match="different"):
        replay_events(
            events,
            runner,
            initial,
            tmp_path,
            {"run": "changed"},
            mode="greedy",
            source_entries=0,
            resume=True,
        )


def test_resume_recovers_pending_journal_record_before_checkpoint(tmp_path):
    initial, runner, events = setup()

    def stop(name, memory, count):
        raise RuntimeError("Snapshot interrupted before state commit")

    with pytest.raises(RuntimeError):
        replay_events(
            events[:12],
            runner,
            initial,
            tmp_path,
            {},
            mode="greedy",
            source_entries=0,
            on_checkpoint=stop,
        )
    assert read_json(tmp_path / "replay_state.json")["next_event"] == 11
    result = replay_events(
        events[:12], runner, initial, tmp_path, {}, mode="greedy", source_entries=0, resume=True
    )
    assert len(result.keys) == 12
    assert len((tmp_path / "events.jsonl").read_bytes().splitlines()) == 12


def test_stale_lock_resume_and_concurrent_replay_refusal(tmp_path, monkeypatch):
    initial, runner, events = setup()

    def stop(cursor):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        replay_events(
            events,
            runner,
            initial,
            tmp_path,
            {"x": 1},
            mode="greedy",
            source_entries=0,
            after_save=stop,
        )
    lock = feedback.ReplayLock(tmp_path / "replay.lock", {"x": 1}, True)
    with lock:
        with pytest.raises(PermissionError, match="active"):
            replay_events(
                events,
                runner,
                initial,
                tmp_path,
                {"x": 1},
                mode="greedy",
                source_entries=0,
                resume=True,
            )
    write_json(tmp_path / "replay.lock", {"pid": 1234, "identity": object_hash({"x": 1})})
    monkeypatch.setattr(feedback, "pid_alive", lambda pid: False)
    result = replay_events(
        events, runner, initial, tmp_path, {"x": 1}, mode="greedy", source_entries=0, resume=True
    )
    assert len(result.keys) == 121
    assert not (tmp_path / "replay.lock").exists()


def test_truncated_pending_journal_is_recovered_only_when_its_bytes_match(tmp_path):
    initial, runner, events = setup()
    identity = {"run": "partial"}
    replay_events(
        events[:8],
        runner,
        initial,
        tmp_path / "reference",
        identity,
        mode="greedy",
        source_entries=0,
    )
    reference_lines = (tmp_path / "reference/events.jsonl").read_bytes().splitlines(keepends=True)

    def stop(cursor):
        if cursor == 5:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        replay_events(
            events[:8],
            runner,
            initial,
            tmp_path / "interrupted",
            identity,
            mode="greedy",
            source_entries=0,
            after_save=stop,
        )
    path = tmp_path / "interrupted/events.jsonl"
    with path.open("ab") as stream:
        stream.write(reference_lines[5][:43])
    restored = replay_events(
        events[:8],
        runner,
        initial,
        tmp_path / "interrupted",
        identity,
        mode="greedy",
        source_entries=0,
        resume=True,
    )
    assert len(restored.keys) == 8
    assert path.read_bytes() == b"".join(reference_lines)
