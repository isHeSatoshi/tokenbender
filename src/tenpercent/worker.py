"""Frozen bundle entrypoint; communicate through JSON lines, never import the checkout."""

from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("candidate", "base"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--state-hash", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--probe-denied", action="append", default=[])
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    code_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_root))
    import tenpercent

    if not Path(tenpercent.__file__).resolve().is_relative_to(code_root):
        raise PermissionError("Worker imported an unfrozen loader")
    interfaces = [name for _, name in socket.if_nameindex()]
    if any(name != "lo" for name in interfaces):
        raise PermissionError("Worker has a non-loopback network interface")
    for denied in args.probe_denied:
        try:
            fd = os.open(denied, os.O_RDONLY)
        except OSError:
            continue
        os.close(fd)
        raise PermissionError("Worker can access a forbidden host path")
    import torch

    from tenpercent.io import sha256_file

    cuda = args.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    if args.kind == "candidate":
        from tenpercent.bundle import bundle_manifest, load_bundle

        path = Path("/candidate")
        manifest = bundle_manifest(path)
        if sha256_file(path / "manifest.json") != args.state_hash:
            raise PermissionError("Candidate manifest identity mismatch")
        try:
            fd = os.open(path / "manifest.json", os.O_WRONLY)
        except OSError:
            pass
        else:
            os.close(fd)
            raise PermissionError("Candidate bundle is writable")
        modes = [manifest["mode"]]
        runner = load_bundle(path, args.device)
    else:
        from tenpercent.model import Runner, TokenizerAdapter, load_base, load_tokenizer

        path = Path("/base")
        if sha256_file(path / "model.safetensors") != args.state_hash:
            raise PermissionError("Base source identity mismatch")
        modes = ["greedy", "constrained"]
        runner = Runner(
            load_base(path, args.device),
            TokenizerAdapter(load_tokenizer(path)),
            json.loads(args.labels),
        )
    if cuda:
        torch.cuda.synchronize()
    labels = getattr(runner, "output_labels", runner.labels)

    def profile():
        result = {"host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        if cuda:
            torch.cuda.synchronize()
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return result

    def send(value):
        protocol.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
        protocol.flush()

    send(
        {
            "event": "ready",
            "kind": args.kind,
            "state_hash": args.state_hash,
            "labels": runner.labels,
            "output_labels": labels,
            "code_root": str(code_root),
            "network_interfaces": interfaces,
            "denied_path_checks": len(args.probe_denied),
            "cold_load_seconds": time.monotonic() - started,
            **profile(),
        }
    )
    for line in sys.stdin:
        message = json.loads(line)
        request_id = message["request_id"]
        if message["event"] == "close":
            send({"event": "closed", "request_id": request_id, **profile()})
            return
        try:
            if message["event"] != "predict" or message["mode"] not in modes:
                raise PermissionError("Request changes the frozen prediction protocol")
            if not isinstance(message["text"], str):
                raise ValueError("Prediction text must be a string")
            start = time.monotonic()
            predicted = runner.predict(message["text"], mode=message["mode"])
            if cuda:
                torch.cuda.synchronize()
            send(
                {
                    "event": "prediction",
                    "request_id": request_id,
                    "predicted_label": predicted,
                    "valid": predicted in labels,
                    "seconds": time.monotonic() - start,
                }
            )
        except Exception as exc:
            send({"event": "error", "request_id": request_id, "error": type(exc).__name__})
            return


if __name__ == "__main__":
    main()
