import hashlib
import io
import json
from urllib.error import URLError

import pytest

import tenpercent.guard as guard
from tenpercent.data import seal
from tenpercent.io import canonical, jsonl_bytes, sha256_file, write_json


def fixture(tmp_path):
    payload = jsonl_bytes([{"id": "exam:1", "text": "synthetic", "label": "a"}])
    key = bytes(range(32))
    (tmp_path / ".secrets").mkdir()
    (tmp_path / ".secrets" / "exam.key").write_bytes(key)
    (tmp_path / "data").mkdir()
    encrypted = seal(payload, key, b"banking/exam")
    (tmp_path / "data" / "exam.enc").write_bytes(encrypted)
    data_manifest = {
        "files": {
            "banking/exam": {
                "path": "exam.enc",
                "count": 1,
                "sha256": hashlib.sha256(encrypted).hexdigest(),
                "plaintext_sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    write_json(tmp_path / "data" / "manifest.json", data_manifest)
    manifest = {
        "format": 1,
        "files": {},
        "data_manifest_sha256": sha256_file(tmp_path / "data" / "manifest.json"),
    }
    write_json(tmp_path / "final_manifest.json", manifest)
    attestation = {
        "commit": "abc",
        "manifest": "final_manifest.json",
        "manifest_sha256": sha256_file(tmp_path / "final_manifest.json"),
        "remote": "https://github.com/executor/proof.git",
        "branch": "main",
    }
    write_json(tmp_path / "attestation.json", attestation)
    return attestation


def test_public_commit_guard_checks_remote_and_worktree(tmp_path, monkeypatch):
    a = fixture(tmp_path)

    def fake_git(root, *args):
        return {
            "rev-parse": "abc",
            "status": "",
            "ls-remote": "abc\trefs/heads/main",
            "show": (tmp_path / "final_manifest.json").read_text().strip(),
        }[args[0]]

    monkeypatch.setattr(guard, "git", fake_git)
    monkeypatch.setattr(guard, "public_head", lambda *args: "abc")
    guard.verify_public_commit(tmp_path, a)
    monkeypatch.setattr(
        guard,
        "git",
        lambda root, *args: "modified" if args[0] == "status" else fake_git(root, *args),
    )
    with pytest.raises(PermissionError, match="clean"):
        guard.verify_public_commit(tmp_path, a)
    monkeypatch.setattr(
        guard,
        "git",
        lambda root, *args: (
            "def\trefs/heads/main" if args[0] == "ls-remote" else fake_git(root, *args)
        ),
    )
    with pytest.raises(PermissionError, match="Remote"):
        guard.verify_public_commit(tmp_path, a)


def test_public_visibility_requires_anonymous_https_read(monkeypatch):
    def response(request, timeout):
        assert request.full_url == "https://api.github.com/repos/executor/proof/commits/main"
        assert "Authorization" not in request.headers
        return io.BytesIO(json.dumps({"sha": "a" * 40}).encode())

    monkeypatch.setattr(guard, "urlopen", response)
    assert guard.public_head("https://github.com/executor/proof.git", "main") == "a" * 40
    with pytest.raises(PermissionError, match="canonical"):
        guard.public_head("/local/repository", "main")

    def unavailable(*args, **kwargs):
        raise URLError("Private or unavailable")

    monkeypatch.setattr(guard, "urlopen", unavailable)
    with pytest.raises(PermissionError, match="anonymously"):
        guard.public_head("https://github.com/executor/proof", "main")


def test_exam_starts_once_and_only_identical_unfinished_session_resumes(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(guard, "verify_public_commit", lambda *args: None)
    session = guard.ExamSession(tmp_path, tmp_path / "attestation.json")
    assert session.rows("banking/exam")[0]["text"] == "synthetic"
    with pytest.raises(PermissionError):
        guard.ExamSession(tmp_path, tmp_path / "attestation.json")
    with pytest.raises(PermissionError, match="active"):
        guard.ExamSession(tmp_path, tmp_path / "attestation.json", resume=True)
    session.release_after_error()
    resumed = guard.ExamSession(tmp_path, tmp_path / "attestation.json", resume=True)
    assert len(resumed.rows("banking/exam")) == 1
    resumed.finish({"complete": True})
    with pytest.raises(PermissionError):
        guard.ExamSession(tmp_path, tmp_path / "attestation.json", resume=True)


def test_modified_manifest_and_encrypted_exam_refused(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(guard, "verify_public_commit", lambda *args: None)
    (tmp_path / "final_manifest.json").write_bytes(canonical({"format": 1}))
    with pytest.raises(PermissionError, match="manifest"):
        guard.ExamSession(tmp_path, tmp_path / "attestation.json")
