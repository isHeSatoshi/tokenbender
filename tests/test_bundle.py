import hashlib

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from tenpercent.bundle import (
    archive_bundle,
    bundle_manifest,
    decode_words,
    encode_words,
    load_bundle,
    pack_model,
    read_tensor,
    verify_lineage,
    write_tensor,
)
from tenpercent.io import sha256_file, write_json
from tenpercent.masking import replace_linears
from tenpercent.model import Runner, TokenizerAdapter
from tenpercent.vocabulary import Vocabulary


def test_lossless_codec_preserves_all_bf16_bit_patterns():
    words = np.arange(65536, dtype=np.uint16)
    values = torch.from_numpy(words).view(torch.bfloat16)
    recovered = decode_words(encode_words(values), len(values))
    assert np.array_equal(recovered.view(torch.uint16).numpy(), words)


def test_sparse_tensor_roundtrip_including_kept_zero(tmp_path):
    (tmp_path / "tensors").mkdir()
    values = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    values[0, 0] = -0.0
    mask = torch.tensor([[True, False], [False, True]])
    descriptor = write_tensor(tmp_path, 0, "x", values, mask, 4)
    recovered = read_tensor(tmp_path, descriptor)
    assert descriptor["entries"] == 8
    expected = torch.where(mask.repeat_interleave(4, 1), values, 0)
    assert torch.equal(recovered.view(torch.int16), expected.view(torch.int16))


def test_wholly_pruned_matrix_roundtrips_as_positive_zero(tmp_path):
    (tmp_path / "tensors").mkdir()
    descriptor = write_tensor(
        tmp_path,
        0,
        "x",
        torch.ones(2, 8, dtype=torch.bfloat16),
        torch.zeros(2, 2, dtype=torch.bool),
        4,
    )
    assert descriptor["entries"] == 0
    assert not read_tensor(tmp_path, descriptor).view(torch.int16).any()


def tiny_bundle(tmp_path):
    torch.manual_seed(1)
    config = Qwen2Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).to(torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    source = tmp_path / "source.safetensors"
    save_file({n: p.detach().clone() for n, p in model.named_parameters()}, source)
    vocabulary = Vocabulary(list(range(8)), {})
    vocab_path = tmp_path / "vocabulary.json"
    vocabulary.save(vocab_path)
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<bos>": 1, "<eos>": 2, "a": 3, "b": 4, "c": 5, "d": 6, "e": 7},
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>", eos_token="<eos>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tokenizer_path)
    modules = replace_linears(model)
    for module in modules.values():
        module.hard.flatten()[1::2] = False
    destination = tmp_path / "bundle"
    manifest = pack_model(
        model,
        tokenizer_path,
        vocab_path,
        ["a", "b"],
        destination,
        source_sha256=sha256_file(source),
    )
    return model, tokenizer, destination, source, manifest


def test_complete_bundle_lineage_standalone_load_and_deterministic_archive(tmp_path):
    model, tokenizer, directory, source, manifest = tiny_bundle(tmp_path)
    proof = verify_lineage(directory, source, expected_sha256=sha256_file(source))
    assert (
        proof["mismatches"] == 0 and proof["checked_source_entries"] == manifest["source_entries"]
    )
    loaded = load_bundle(directory, "cpu")
    adapter = TokenizerAdapter(tokenizer)
    reference = Runner(model, adapter, ["a", "b"])
    # Compare actual logits and generation, not merely file round-trip.
    ids = torch.tensor([[0, 3, 4]])
    with torch.inference_mode():
        expected = model(ids).logits
        actual = loaded.model(ids).logits
    assert torch.equal(expected, actual)
    assert loaded.predict("synthetic", "greedy") == reference.predict("synthetic", "greedy")
    first = tmp_path / "first.tar.zst"
    second = tmp_path / "second.tar.zst"
    archive_bundle(directory, first)
    archive_bundle(directory, second)
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    (directory / "unregistered.txt").write_text("not allowed")
    with pytest.raises(ValueError, match="Unregistered"):
        bundle_manifest(directory)


@pytest.mark.parametrize(
    "change, message",
    [
        ("duplicate", "Duplicate"),
        ("count", "entry count"),
        ("shape", "shape"),
        ("source_ledger", "ledger"),
        ("model_hash", "fingerprint"),
        ("symlink", "symbolic"),
    ],
)
def test_invalid_bundle_inventory_is_refused(tmp_path, change, message):
    _, _, directory, _, manifest = tiny_bundle(tmp_path)
    if change == "duplicate":
        manifest["tensors"].append(dict(manifest["tensors"][0]))
    elif change == "count":
        manifest["tensors"][0]["entries"] = -1
    elif change == "shape":
        manifest["tensors"][0]["shape"] = [1.5]
    elif change == "source_ledger":
        manifest["source_entries"] += 1
    elif change == "model_hash":
        manifest["model_hash"] = "wrong"
    elif change == "symlink":
        (directory / "alias").symlink_to(directory / "config.json")
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match=message):
        bundle_manifest(directory)
