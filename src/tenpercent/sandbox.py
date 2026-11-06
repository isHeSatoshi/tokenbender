from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sandbox_command(mounts: dict[str, Path], entrypoint: str, *, device: str) -> list[str]:
    """Mount only generic runtime files and the explicit read-only model inputs."""
    executable = shutil.which("bwrap")
    if not executable:
        raise PermissionError("Source-isolated inference requires bubblewrap")
    command = [
        executable,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--ro-bind",
        str(Path(sys.prefix).resolve()),
        "/venv",
    ]
    prefixes = {Path(sys.base_prefix).resolve()}
    python = Path(sys.executable)
    if python.is_symlink():
        target = Path(os.path.abspath(python.parent / os.readlink(python)))
        prefixes.add(target.parent.parent)
    for prefix in sorted(prefixes):
        command += ["--ro-bind", str(prefix.resolve()), str(prefix)]
    command += [
        "--ro-bind-try",
        "/etc/ld.so.cache",
        "/etc/ld.so.cache",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--chdir",
        "/work",
    ]
    if device.startswith("cuda"):
        for path in sorted(Path("/dev").glob("nvidia*")):
            command += ["--dev-bind", str(path), str(path)]
        # CUDA also resolves device nodes through /sys/dev/char and its symlinks.
        for path in ("/sys",):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
    for destination, source in sorted(mounts.items()):
        command += ["--ro-bind", str(source.resolve(strict=True)), destination]
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/tmp/home",
        "TMPDIR": "/tmp",
        "XDG_CACHE_HOME": "/tmp/cache",
        "HF_HOME": "/tmp/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "CUDA_CACHE_PATH": "/tmp/cuda",
    }
    for name in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"):
        if name in os.environ:
            environment[name] = os.environ[name]
    for name, value in environment.items():
        command += ["--setenv", name, value]
    return command + ["/venv/bin/python", "-I", entrypoint]


class IsolatedRunner:
    """A prediction-only IPC proxy. Labels and source files never travel together."""

    device = "isolated"

    def __init__(
        self,
        path: Path,
        state_hash: str,
        *,
        device="cuda",
        source=False,
        loader: Path | None = None,
        labels: list[str] | None = None,
        denied_paths: tuple[Path, ...] = (),
        timeout=180,
    ):
        path = path.resolve(strict=True)
        if source:
            if loader is None or labels is None:
                raise ValueError("The base worker needs a frozen loader and original labels")
            mounts = {"/base": path, "/runtime": loader}
            entrypoint = "/runtime/tenpercent/worker.py"
        else:
            mounts = {"/candidate": path}
            entrypoint = "/candidate/loader/tenpercent/worker.py"
        command = sandbox_command(mounts, entrypoint, device=device)
        command += [
            "--kind",
            "base" if source else "candidate",
            "--device",
            device,
            "--state-hash",
            state_hash,
        ]
        if source:
            command += ["--labels", json.dumps(labels)]
        for denied in denied_paths:
            command += ["--probe-denied", str(denied.resolve())]
        self.timeout, self._buffer, self._counter = timeout, b"", 0
        self._stderr = tempfile.TemporaryFile()
        self._started = time.monotonic()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            close_fds=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.hello = self._read()
            if self.hello.get("event") != "ready" or self.hello.get("state_hash") != state_hash:
                raise ValueError("Isolated worker identity mismatch")
            self.hello["launch_seconds"] = time.monotonic() - self._started
            self.labels = self.hello["labels"]
            self.output_labels = self.hello["output_labels"]
            self.profile = None
        except BaseException:
            self.close(graceful=False)
            raise

    def _read(self) -> dict:
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self._buffer:
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise TimeoutError("Isolated inference worker exceeded its response deadline")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self._stderr.seek(0)
                detail = self._stderr.read().decode(errors="replace")[-8000:]
                raise RuntimeError(f"Isolated inference worker exited: {detail}")
            self._buffer += chunk
            if len(self._buffer) > 1024 * 1024:
                raise ValueError("Oversized isolated worker response")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(line)
        if response.get("event") == "error":
            raise RuntimeError(f"Isolated worker rejected request: {response['error']}")
        return response

    def _request(self, message: dict) -> dict:
        self._counter += 1
        message = {**message, "request_id": self._counter}
        self.process.stdin.write(json.dumps(message, allow_nan=False).encode() + b"\n")
        self.process.stdin.flush()
        response = self._read()
        if response.get("request_id") != self._counter:
            raise ValueError("Isolated worker response is out of sequence")
        return response

    def predict(self, text: str, mode: str = "constrained") -> str:
        response = self._request({"event": "predict", "text": text, "mode": mode})
        if response.get("event") != "prediction":
            raise ValueError("Isolated worker did not return a prediction")
        predicted = response["predicted_label"]
        if response["valid"] != (predicted in self.output_labels):
            raise ValueError("Isolated worker returned an inconsistent validity flag")
        return predicted

    def close(self, *, graceful=True):
        try:
            if graceful and self.process.poll() is None:
                self.profile = self._request({"event": "close"})
                self.process.stdin.close()
                self.process.wait(timeout=10)
                if self.process.returncode:
                    raise RuntimeError("Isolated worker failed on shutdown")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout, self._stderr):
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(graceful=exc_type is None)
