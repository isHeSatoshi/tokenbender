from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import struct
from pathlib import Path

from .constants import (
    BASE_BYTES,
    BASE_PARAMETERS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
)
from .io import sha256_file, write_json


def inspect_safetensors(path: Path) -> dict:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if length > 100 * 1024 * 1024:
            raise ValueError("Unreasonable safetensors header")
        header = json.loads(handle.read(length))
    sizes = {
        "BF16": 2,
        "F32": 4,
        "F16": 2,
        "I64": 8,
        "I32": 4,
        "U8": 1,
        "I8": 1,
        "BOOL": 1,
        "U16": 2,
        "I16": 2,
    }
    tensors, offset = {}, 0
    for name, info in sorted(
        ((n, v) for n, v in header.items() if n != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    ):
        count = math.prod(info["shape"])
        start, end = info["data_offsets"]
        if start != offset or end - start != count * sizes[info["dtype"]]:
            raise ValueError(f"Invalid tensor extent: {name}")
        offset = end
        tensors[name] = {"shape": info["shape"], "dtype": info["dtype"], "entries": count}
    if 8 + length + offset != path.stat().st_size:
        raise ValueError("Safetensors payload length mismatch")
    return {"parameters": sum(t["entries"] for t in tensors.values()), "tensors": tensors}


def download_model(root: Path, *, weights: bool = True) -> dict:
    from huggingface_hub import snapshot_download

    target = root / "assets" / "base"
    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "README.md",
    ]
    if weights:
        patterns.append("model.safetensors")
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=target,
        allow_patterns=patterns,
    )
    result = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": {}}
    for path in sorted(target.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            result["files"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if weights:
        info = result["files"]["model.safetensors"]
        if info != {"sha256": MODEL_SHA256, "bytes": BASE_BYTES}:
            raise ValueError("Pinned model weights identity mismatch; no experiment is allowed")
        inspection = inspect_safetensors(target / "model.safetensors")
        if inspection["parameters"] != BASE_PARAMETERS:
            raise ValueError("Pinned model parameter count mismatch")
        if any(v["dtype"] != "BF16" for v in inspection["tensors"].values()):
            raise ValueError("Pinned checkpoint must contain BF16 values only")
        result["inspection"] = inspection
    write_json(root / "runs" / "source_verification.json", result)
    return result


def environment_report() -> dict:
    import torch

    packages = {}
    for name in (
        "torch",
        "transformers",
        "datasets",
        "safetensors",
        "numpy",
        "huggingface-hub",
        "cryptography",
        "zstandard",
    ):
        packages[name] = importlib.metadata.version(name)
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        result["vram"] = {"free": free, "total": total}
    return result
