import shutil
import sys

import pytest

from tenpercent.bundle import load_bundle
from tenpercent.io import sha256_file
from tenpercent.sandbox import IsolatedRunner, sandbox_command


def test_sandbox_mounts_no_checkout_cache_or_home(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    command = sandbox_command({"/candidate": bundle}, "/candidate/worker.py", device="cpu")
    assert "--unshare-all" in command
    assert "--clearenv" in command
    assert command[command.index("--chdir") + 1] == "/work"
    assert str(tmp_path) not in command
    assert sys.prefix not in command or str(tmp_path) not in sys.prefix
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(PermissionError, match="bubblewrap"):
        sandbox_command({"/candidate": bundle}, "/candidate/worker.py", device="cpu")


def test_real_cpu_bundle_runs_frozen_code_without_source_or_network(tmp_path):
    from test_bundle import tiny_bundle

    _, _, bundle, source, manifest = tiny_bundle(tmp_path)
    reference = load_bundle(bundle, "cpu")
    state_hash = sha256_file(bundle / "manifest.json")
    with IsolatedRunner(
        bundle, state_hash, device="cpu", denied_paths=(source, tmp_path / ".secrets/exam.key")
    ) as worker:
        assert worker.hello["code_root"] == "/candidate/loader"
        assert set(worker.hello["network_interfaces"]) <= {"lo"}
        assert worker.hello["denied_path_checks"] == 2
        for text in ("", "synthetic", "हिन्दी 中文 café", "long input " * 130):
            assert worker.predict(text, manifest["mode"]) == reference.predict(
                text, manifest["mode"]
            )
    assert worker.profile["host_peak_rss_bytes"] > 0
    assert worker.process.returncode == 0
    with pytest.raises(RuntimeError, match="identity"):
        IsolatedRunner(bundle, "0" * 64, device="cpu")


def test_worker_refuses_unregistered_mode_and_cleans_up(tmp_path):
    from test_bundle import tiny_bundle

    _, _, bundle, _, manifest = tiny_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="PermissionError"):
        with IsolatedRunner(bundle, sha256_file(bundle / "manifest.json"), device="cpu") as worker:
            worker.predict("synthetic", "greedy" if manifest["mode"] != "greedy" else "constrained")
    assert worker.process.poll() is not None


def test_original_base_has_a_separate_worker_and_only_registered_source(tmp_path):
    from test_bundle import tiny_bundle

    from tenpercent.model import Runner, TokenizerAdapter, load_base, load_tokenizer

    model, tokenizer, bundle, source, _ = tiny_bundle(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    model.config.save_pretrained(base)
    tokenizer.save_pretrained(base)
    shutil.copyfile(source, base / "model.safetensors")
    reference = Runner(load_base(base, "cpu"), TokenizerAdapter(load_tokenizer(base)), ["a", "b"])
    with IsolatedRunner(
        base,
        sha256_file(source),
        device="cpu",
        source=True,
        loader=bundle / "loader",
        labels=["a", "b"],
        denied_paths=(bundle / "manifest.json",),
    ) as worker:
        assert worker.hello["kind"] == "base"
        assert worker.hello["code_root"] == "/runtime"
        for mode in ("greedy", "constrained"):
            assert worker.predict("synthetic", mode) == reference.predict("synthetic", mode)
