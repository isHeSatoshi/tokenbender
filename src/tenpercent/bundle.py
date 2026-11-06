from __future__ import annotations

import math
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

from .constants import BYTE_CAP, MODEL_SHA256, PARAMETER_CAP
from .io import object_hash, read_json, safe_relative, sha256_file, write_json
from .masking import MaskedLinear


def model_fingerprint(descriptors, files):
    return object_hash(
        {
            "tensors": descriptors,
            "payloads": {k: v for k, v in files.items() if k.startswith("tensors/")},
            "config": files["config.json"],
            "vocabulary": files["vocabulary.json"],
        }
    )


def encode_words(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.bfloat16:
        raise ValueError("Only original BF16 words may be packed")
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(-1, 2)
    planes = raw.T.copy().tobytes()
    return zstd.ZstdCompressor(level=9).compress(planes)


def decode_words(payload: bytes, entries: int) -> torch.Tensor:
    planes = zstd.ZstdDecompressor().decompress(payload, max_output_size=entries * 2)
    if len(planes) != entries * 2:
        raise ValueError("Lossless value payload length mismatch")
    raw = np.frombuffer(planes, dtype=np.uint8).reshape(2, -1).T.copy()
    return torch.from_numpy(raw.reshape(-1)).view(torch.bfloat16)


def write_tensor(
    destination: Path,
    index: int,
    name: str,
    tensor: torch.Tensor,
    mask: torch.Tensor | None = None,
    group_size: int = 4,
) -> dict:
    descriptor = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": "BF16",
        "values": f"tensors/{index:04d}.values.zst",
    }
    if mask is not None:
        selected = tensor.reshape(*mask.shape, group_size)[mask].flatten()
        bits = np.packbits(mask.detach().cpu().numpy().reshape(-1), bitorder="little")
        relative = f"tensors/{index:04d}.mask.zst"
        (destination / relative).write_bytes(zstd.ZstdCompressor(level=9).compress(bits.tobytes()))
        descriptor.update(mask=relative, groups=mask.numel(), group_size=group_size)
    else:
        selected = tensor.flatten()
    descriptor["entries"] = selected.numel()
    (destination / descriptor["values"]).write_bytes(encode_words(selected))
    return descriptor


def read_tensor(root: Path, descriptor: dict, *, selected_only=False):
    values = decode_words(
        safe_relative(root, descriptor["values"]).read_bytes(), descriptor["entries"]
    )
    if "mask" not in descriptor:
        return values if selected_only else values.reshape(descriptor["shape"])
    groups = descriptor["groups"]
    packed = zstd.ZstdDecompressor().decompress(
        safe_relative(root, descriptor["mask"]).read_bytes(),
        max_output_size=(groups + 7) // 8,
    )
    if len(packed) != (groups + 7) // 8:
        raise ValueError("Invalid mask payload length")
    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:groups]
    mask = torch.from_numpy(mask.astype(np.bool_))
    size = descriptor["group_size"]
    if int(mask.sum()) * size != values.numel():
        raise ValueError("Mask count does not match selected value count")
    if selected_only:
        return values, mask
    dense = torch.zeros(groups, size, dtype=torch.bfloat16)
    dense[mask] = values.reshape(-1, size)
    return dense.reshape(descriptor["shape"])


def pack_model(
    model,
    tokenizer_source: Path,
    vocabulary_path: Path,
    labels: list[str],
    destination: Path,
    *,
    state="A",
    mode="constrained",
    memory=None,
    feedback_events=0,
    source_sha256=MODEL_SHA256,
) -> dict:
    if destination.exists():
        raise FileExistsError("Never overwrite a frozen bundle")
    if state not in {"A", "B", "1-shot", "5-shot"}:
        raise ValueError("Unknown frozen state")
    destination.mkdir(parents=True)
    (destination / "tensors").mkdir()
    (destination / "tokenizer").mkdir()
    descriptors = []
    body = {}
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            body[name + ".weight"] = module
    values = {
        name: value for name, value in model.named_parameters() if not name.endswith(".scores")
    }
    values.update({name: module.weight for name, module in body.items()})
    values.update(
        {
            name.removesuffix(".weight") + ".bias": module.bias
            for name, module in body.items()
            if module.bias is not None
        }
    )
    if any(value.requires_grad for name, value in values.items()):
        raise ValueError("Deployable source values must be frozen")
    for index, (name, value) in enumerate(sorted(values.items())):
        module = body.get(name)
        descriptor = write_tensor(
            destination,
            index,
            name,
            value,
            None if module is None else module.hard,
            4 if module is None else module.group_size,
        )
        descriptors.append(descriptor)
    config = model.config.to_dict()
    config["_name_or_path"] = ""
    write_json(destination / "config.json", config)
    shutil.copyfile(vocabulary_path, destination / "vocabulary.json")
    write_json(destination / "labels.json", labels)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (tokenizer_source / name).is_file():
            shutil.copyfile(tokenizer_source / name, destination / "tokenizer" / name)
    (destination / "LICENSES").mkdir()
    if (tokenizer_source / "README.md").is_file():
        shutil.copyfile(tokenizer_source / "README.md", destination / "LICENSES" / "MODEL_CARD.md")
    (destination / "LICENSES" / "ATTRIBUTION.txt").write_text(
        "Source model: johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent (Apache-2.0).\n"
        "Banking77: Casanueva et al., CC BY 4.0. CLINC150: Larson et al., CC BY 3.0.\n"
        "No raw exemplar text is required by this inference package.\n"
    )
    # Bundle the exact custom loader, not a reference to a mutable checkout.
    shutil.copytree(
        Path(__file__).parent,
        destination / "loader" / "tenpercent",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    source_root = Path(__file__).resolve().parents[2]
    for name in ("pyproject.toml", "uv.lock"):
        if (source_root / name).is_file():
            shutil.copyfile(source_root / name, destination / name)
    auxiliary = 0
    if memory is not None:
        memory.save(destination / "memory.json")
        auxiliary = memory.numeric_entries
    entries = sum(descriptor["entries"] for descriptor in descriptors) + auxiliary
    if entries > PARAMETER_CAP:
        raise ValueError(f"Bundle has {entries} entries, above {PARAMETER_CAP}")
    files = {
        p.relative_to(destination).as_posix(): sha256_file(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    model_hash = model_fingerprint(descriptors, files)
    manifest = {
        "format": 1,
        "state": state,
        "mode": mode,
        "source_sha256": source_sha256,
        "source_entries": entries - auxiliary,
        "auxiliary_entries": auxiliary,
        "retained_entries": entries,
        "feedback_events": feedback_events,
        "model_hash": model_hash,
        "tensors": descriptors,
        "files": files,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def bundle_manifest(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    if manifest.get("format") != 1:
        raise ValueError("Unsupported bundle")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Inference bundles cannot contain symbolic links")
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Unregistered files inside inference bundle")
    for relative, digest in manifest["files"].items():
        file = safe_relative(path, relative)
        if file.is_symlink() or sha256_file(file) != digest:
            raise ValueError("Inference bundle file hash mismatch")
    descriptors = manifest["tensors"]
    names = [t["name"] for t in descriptors]
    if len(set(names)) != len(names) or "lm_head.weight" in names:
        raise ValueError("Duplicate tensor names or separately stored tied head")
    references = []
    for tensor in descriptors:
        shape, count = tensor["shape"], tensor["entries"]
        if (
            tensor["dtype"] != "BF16"
            or not shape
            or any(type(v) is not int or v < 1 for v in shape)
            or type(count) is not int
            or count < 0
            or count > math.prod(shape)
        ):
            raise ValueError("Invalid tensor shape, dtype, or entry count")
        references.append(tensor["values"])
        if "mask" in tensor:
            size = tensor["group_size"]
            if (
                size not in (2, 4)
                or len(shape) != 2
                or shape[-1] % size
                or tensor["groups"] != math.prod(shape) // size
                or count % size
            ):
                raise ValueError("Invalid grouped-mask descriptor")
            references.append(tensor["mask"])
        elif count != math.prod(shape):
            raise ValueError("Dense tensor count differs from its shape")
    registered = {p for p in manifest["files"] if p.startswith("tensors/")}
    if len(set(references)) != len(references) or set(references) != registered:
        raise ValueError("Tensor payload inventory mismatch")
    counted_source = sum(t["entries"] for t in descriptors)
    counted = counted_source + manifest["auxiliary_entries"]
    if (
        counted_source != manifest["source_entries"]
        or counted != manifest["retained_entries"]
        or counted > PARAMETER_CAP
        or manifest["auxiliary_entries"] < 0
    ):
        raise ValueError("Inference bundle entry ledger mismatch")
    if model_fingerprint(descriptors, manifest["files"]) != manifest["model_hash"]:
        raise ValueError("Inference model fingerprint mismatch")
    if bool(manifest["auxiliary_entries"]) != ("memory.json" in manifest["files"]):
        raise ValueError("Auxiliary memory missing or uncounted")
    return manifest


def load_bundle(path: Path, device="cuda"):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM

    from .memory import FeedbackMemory, MemoryRunner
    from .model import Runner, deterministic, load_tokenizer
    from .vocabulary import CompactTokenizer, Vocabulary

    manifest = bundle_manifest(path)
    deterministic()
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not config.tie_word_embeddings:
        raise ValueError("This loader requires the registered tied embedding/head graph")
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    expected = {name for name, _ in model.named_parameters()}
    supplied = {descriptor["name"] for descriptor in manifest["tensors"]}
    # Some Transformers versions expose both tied aliases while on meta.
    expected.discard("lm_head.weight")
    supplied_without_alias = supplied - {"lm_head.weight"}
    if expected != supplied_without_alias:
        raise ValueError("Bundle tensor graph differs from configured model")
    for descriptor in manifest["tensors"]:
        set_module_tensor_to_device(
            model,
            descriptor["name"],
            device,
            value=read_tensor(path, descriptor),
        )
    model.tie_weights()
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    adapter = CompactTokenizer(
        load_tokenizer(path / "tokenizer"), Vocabulary.load(path / "vocabulary.json")
    )
    labels = read_json(path / "labels.json")
    runner = Runner(model, adapter, labels)
    from .constants import NEW_INTENTS

    runner.output_labels = labels + [label for label in NEW_INTENTS if label not in labels]
    if (path / "memory.json").exists():
        memory = FeedbackMemory.load(path / "memory.json")
        if memory.numeric_entries != manifest["auxiliary_entries"]:
            raise ValueError("Memory numeric-entry count mismatch")
        return MemoryRunner(runner, memory, manifest["mode"])
    return runner


def verify_lineage(bundle: Path, source: Path, *, expected_sha256=MODEL_SHA256) -> dict:
    from safetensors import safe_open

    from .vocabulary import Vocabulary

    if sha256_file(source) != expected_sha256:
        raise ValueError("Verifier source checkpoint hash mismatch")
    manifest = bundle_manifest(bundle)
    if manifest["source_sha256"] != expected_sha256:
        raise ValueError("Bundle declares another source checkpoint")
    vocabulary = Vocabulary.load(bundle / "vocabulary.json")
    checked = 0
    with safe_open(source, framework="pt", device="cpu") as original:
        if {t["name"] for t in manifest["tensors"]} != set(original.keys()):
            raise ValueError("Source tensor inventory differs from the registered full graph")
        for descriptor in manifest["tensors"]:
            reference = original.get_tensor(descriptor["name"])
            if descriptor["name"] == "model.embed_tokens.weight":
                reference = reference[vocabulary.retained]
            if list(reference.shape) != descriptor["shape"]:
                raise ValueError("Source tensor shape mismatch")
            if "mask" in descriptor:
                selected, mask = read_tensor(bundle, descriptor, selected_only=True)
                reference = reference.reshape(-1, descriptor["group_size"])[mask].flatten()
            else:
                selected = read_tensor(bundle, descriptor, selected_only=True)
                reference = reference.flatten()
            if not torch.equal(
                selected.view(torch.int16), reference.contiguous().view(torch.int16)
            ):
                raise ValueError(f"Non-identical retained words: {descriptor['name']}")
            checked += selected.numel()
    if checked != manifest["source_entries"]:
        raise ValueError("Lineage total differs from source-entry ledger")
    return {
        "mismatches": 0,
        "checked_source_entries": checked,
        "source_sha256": expected_sha256,
        "bundle_model_hash": manifest["model_hash"],
    }


def archive_bundle(source: Path, destination: Path) -> dict:
    bundle_manifest(source)
    if destination.exists():
        raise FileExistsError("Archive already exists")
    with destination.open("xb") as raw:
        with zstd.ZstdCompressor(level=9).stream_writer(raw) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    info = archive.gettarinfo(
                        str(path), arcname=path.relative_to(source).as_posix()
                    )
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    size = destination.stat().st_size
    return {"bytes": size, "byte_target_pass": size <= BYTE_CAP, "sha256": sha256_file(destination)}
