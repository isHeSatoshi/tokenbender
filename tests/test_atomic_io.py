from concurrent.futures import ThreadPoolExecutor

import pytest

from tenpercent import io


def test_exclusive_atomic_publish_never_exposes_an_incomplete_file(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"

    def fail(*args):
        assert not path.exists()
        raise RuntimeError("interrupted before publish")

    with monkeypatch.context() as context:
        context.setattr(io.os, "link", fail)
        with pytest.raises(RuntimeError):
            io.write_json(path, {"complete": True}, exclusive=True)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []

    def publish(value):
        try:
            io.write_json(path, {"winner": value}, exclusive=True)
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, [1, 2]))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1 and io.read_json(path) == {"winner": winners[0]}
