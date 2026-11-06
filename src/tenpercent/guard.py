from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .data import unseal
from .io import canonical, read_json, safe_relative, sha256_file, write_json


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def public_head(remote: str, branch: str) -> str:
    match = re.fullmatch(
        r"https://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?", remote
    )
    if not match:
        raise PermissionError("Attestation must name a canonical public HTTPS GitHub repository")
    owner, repository = match.groups()
    request = Request(
        f"https://api.github.com/repos/{owner}/{repository}/commits/{quote(branch, safe='')}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "tenpercent-proof"},
    )
    # This request intentionally has no token or credential helper.
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (URLError, ValueError) as exc:
        raise PermissionError("Remote commit is not anonymously publicly readable") from exc
    sha = result.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise PermissionError("Public repository did not return a valid commit")
    return sha


def verify_public_commit(root: Path, attestation: dict) -> None:
    commit = attestation["commit"]
    if git(root, "rev-parse", "HEAD") != commit:
        raise PermissionError("Checkout is not the frozen commit")
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise PermissionError("Working tree must be clean before opening the exam")
    remote = attestation["remote"]
    branch = attestation["branch"]
    remote_head = git(root, "ls-remote", "--exit-code", remote, f"refs/heads/{branch}")
    if remote_head.split()[0] != commit:
        raise PermissionError("Remote branch does not match the frozen commit")
    if public_head(remote, branch) != commit:
        raise PermissionError("Public remote branch does not match the frozen commit")
    manifest_path = attestation["manifest"]
    tracked = git(root, "show", f"{commit}:{manifest_path}").encode() + b"\n"
    if hashlib.sha256(tracked).hexdigest() != attestation["manifest_sha256"]:
        raise PermissionError("Public commit does not contain the attested manifest")


class ExamSession:
    """Single-start, hash-bound session. This is procedural blinding, not a sandbox."""

    def __init__(self, root: Path, attestation: Path, *, resume: bool = False):
        self.root = root.resolve()
        self.attestation = read_json(attestation)
        verify_public_commit(self.root, self.attestation)
        self.manifest_path = safe_relative(self.root, self.attestation["manifest"])
        if sha256_file(self.manifest_path) != self.attestation["manifest_sha256"]:
            raise PermissionError("Final manifest hash mismatch")
        self.manifest = read_json(self.manifest_path)
        if self.manifest.get("format") != 1:
            raise PermissionError("Unsupported final manifest")
        for relative, digest in self.manifest["files"].items():
            if sha256_file(safe_relative(self.root, relative)) != digest:
                raise PermissionError(f"Frozen file changed: {relative}")
        self.receipt = self.root / "runs" / "exam_receipt.json"
        identity = {
            "manifest_sha256": self.attestation["manifest_sha256"],
            "commit": self.attestation["commit"],
        }
        self.identity = identity
        # Keep this inode in place. Kernel ownership survives races and clears on process death.
        self.active = self.root / "runs" / "exam_active.lock"
        self.active.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.active.open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise PermissionError("An exam process is active") from exc
        try:
            if resume:
                previous = read_json(self.receipt)
                if previous.get("status") != "started" or any(
                    previous.get(k) != v for k, v in identity.items()
                ):
                    raise PermissionError("No matching unfinished session to resume")
            else:
                try:
                    write_json(self.receipt, {**identity, "status": "started"}, exclusive=True)
                except FileExistsError as exc:
                    raise PermissionError(
                        "Exam already started; no second session is allowed"
                    ) from exc
            self._lock.seek(0)
            self._lock.truncate()
            self._lock.write(canonical({**identity, "pid": os.getpid()}))
            self._lock.flush()
            os.fsync(self._lock.fileno())
        except BaseException:
            self.release_after_error()
            raise

    def rows(self, name: str):
        if name not in {"banking/exam", "clinc/test"}:
            raise PermissionError("Final session accepts only registered test splits")
        data_manifest = read_json(self.root / "data" / "manifest.json")
        if (
            sha256_file(self.root / "data" / "manifest.json")
            != self.manifest["data_manifest_sha256"]
        ):
            raise PermissionError("Split commitment changed")
        info = data_manifest["files"][name]
        path = safe_relative(self.root / "data", info["path"])
        if sha256_file(path) != info["sha256"]:
            raise PermissionError("Sealed exam hash mismatch")
        key = (self.root / ".secrets" / "exam.key").read_bytes()
        plaintext = unseal(path.read_bytes(), key, name.encode())
        if hashlib.sha256(plaintext).hexdigest() != info["plaintext_sha256"]:
            raise PermissionError("Exam plaintext hash mismatch")
        rows = [json.loads(line) for line in plaintext.splitlines()]
        if len(rows) != info["count"]:
            raise PermissionError("Exam row count mismatch")
        return rows

    def finish(self, result: dict) -> None:
        write_json(
            self.receipt,
            {
                **self.identity,
                "status": "finished",
                "result_sha256": hashlib.sha256(canonical(result)).hexdigest(),
            },
        )
        self.release_after_error()

    def release_after_error(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
