import numpy as np
import pytest
import torch
from torch import nn

from tenpercent.masking import MaskedLinear
from tenpercent.training import (
    require_preregistration,
    restore_checkpoint,
    save_checkpoint,
    validate_search_config,
)


def test_tiny_search_updates_scores_selects_both_modes_and_resumes(tmp_path, monkeypatch):
    import copy

    from transformers import Qwen2Config, Qwen2ForCausalLM

    from tenpercent import training
    from tenpercent.accounting import Budget
    from tenpercent.constants import BODY_SUFFIXES, MODEL_SHA256
    from tenpercent.io import append_record, object_hash, read_json, sha256_file, write_json
    from tenpercent.teacher import TeacherCache, _teacher_record, commit_teacher
    from tenpercent.vocabulary import Vocabulary

    class Adapter:
        eos_token_id = 2

        def prompt(self, text):
            return [1, 3]

        def label(self, text, label):
            return [{"a": 4, "b": 5}[label], 2]

        def original_id(self, token):
            return token

        def decode(self, tokens):
            return {4: "a", 5: "b"}.get(tokens[0], "invalid")

    adapter = Adapter()
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
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(config).to(torch.bfloat16).eval().requires_grad_(False)
    body = sum(p.numel() for n, p in model.named_parameters() if n.endswith(BODY_SUFFIXES))
    dense = sum(p.numel() for p in model.parameters()) - body
    target = dense + body // 2
    monkeypatch.setattr(
        training,
        "require_preregistration",
        lambda root: {
            "search": {
                "steps": [2],
                "learning_rates": [0.003],
                "accumulations": [1],
                "source_targets": [target],
                "group_sizes": [4],
                "diagnostic_steps": [],
            }
        },
    )
    monkeypatch.setattr(training, "data_commitment", lambda root: "test")
    monkeypatch.setattr(training, "load_base", lambda *args: copy.deepcopy(model))
    monkeypatch.setattr(training, "load_tokenizer", lambda *args: None)
    monkeypatch.setattr(training, "CompactTokenizer", lambda *args: adapter)
    monkeypatch.setattr(
        training,
        "Budget",
        lambda v, g, t, **kwargs: Budget(
            v, g, t, hidden=8, fixed_body_entries=dense - 64, body_entries=body, **kwargs
        ),
    )
    rows = [{"id": "one", "text": "a", "label": "a"}, {"id": "two", "text": "b", "label": "b"}]
    monkeypatch.setattr(training, "load_split", lambda *args, **kwargs: rows)
    write_json(tmp_path / "data/manifest.json", {"labels": ["a", "b"]})
    Vocabulary(list(range(8)), {}).save(tmp_path / "runs/vocabulary.json")
    meta = {
        "format": 2,
        "source_sha256": MODEL_SHA256,
        "temperature": 2,
        "topk": 64,
        "vocabulary_sha256": sha256_file(tmp_path / "runs/vocabulary.json"),
        "data_commitment": "test",
        "examples": len(rows),
        "smoke_only": False,
    }
    cache = TeacherCache(tmp_path / "runs/teacher.sqlite", meta)
    for row in rows:
        record = _teacher_record(model, adapter, row["text"], row["label"])
        assert torch.allclose(
            record["top_probabilities"].sum(-1) + record["tail_probability"],
            torch.ones(2),
            atol=1e-6,
        )
        assert torch.equal(record["top_ids"], record["top_original_ids"])
        cache.put(row["id"], record)
    cache.close()
    commit_teacher(tmp_path, tmp_path / "runs/teacher.sqlite", meta)
    for mode in ("greedy", "constrained"):
        for row in rows:
            append_record(
                tmp_path / f"runs/baseline/working-2-{mode}.jsonl",
                {
                    "example_id": row["id"],
                    "true_label": row["label"],
                    "predicted_label": row["label"],
                },
            )
    before = training.frozen_hash(model)
    with pytest.raises(PermissionError, match="resource pilot"):
        training.run_search(tmp_path, steps=2, accumulation=1, source_target=target, device="cpu")
    pilot = training.run_search(
        tmp_path, steps=2, accumulation=1, source_target=target, device="cpu", resource_pilot=True
    )
    assert pilot["report"]["optimizer_updates"] == 3
    assert pilot["report"]["max_input_tokens"] == 3
    assert pilot["report"]["source_unchanged"]
    assert pilot["report"]["keep_group_phases"][0] > pilot["report"]["keep_group_phases"][-1]
    assert not (tmp_path / "runs/candidates.jsonl").exists()
    assert not (tmp_path / "runs/pilots" / pilot["resource_pilot"] / "selection.json").exists()
    result = training.run_search(
        tmp_path, steps=2, accumulation=1, source_target=target, device="cpu"
    )
    output = tmp_path / "runs/search" / result["candidate"]
    selection = read_json(output / "selection.json")
    assert selection["entries"] == target
    assert selection["mode"] in ("greedy", "constrained")
    assert len((output / "working_evaluations.jsonl").read_text().splitlines()) == 2
    assert (output / "selected_masks.safetensors").is_file()
    assert result == training.run_search(
        tmp_path, steps=2, accumulation=1, source_target=target, device="cpu", resume=True
    )
    assert object_hash(read_json(output / "config.json"))[:16] == result["candidate"]
    assert training.frozen_hash(model) == before


def test_search_refuses_missing_public_registration_before_loading_weights(tmp_path):
    with pytest.raises(PermissionError, match="pushed preregistration"):
        require_preregistration(tmp_path)


def test_unregistered_search_settings_are_rejected():
    registration = {
        "search": {
            "steps": [200, 2000],
            "learning_rates": [0.001, 0.003, 0.01],
            "accumulations": [32],
            "source_targets": [152800000, 463114291],
            "group_sizes": [4, 2],
            "diagnostic_steps": [200],
        }
    }
    args = dict(
        steps=200,
        learning_rate=0.003,
        accumulation=32,
        source_target=152800000,
        group_size=4,
        diagnostic_oversize=False,
    )
    validate_search_config(registration, **args)
    for changed in (
        {"learning_rate": 0.3},
        {"accumulation": 1},
        {"source_target": 150000000},
        {"steps": 17},
        {"group_size": 8},
        {"diagnostic_oversize": True},
        {"source_target": 463114291},
        {"steps": 2000, "source_target": 463114291, "diagnostic_oversize": True},
    ):
        with pytest.raises(PermissionError, match="preregistration"):
            validate_search_config(registration, **{**args, **changed})


def test_optimizer_scores_and_data_order_resume_exactly(tmp_path):
    torch.manual_seed(1)
    modules = {"x": MaskedLinear(nn.Linear(8, 4, bias=False))}
    optimizer = torch.optim.Adam([modules["x"].scores], lr=0.01, foreach=False)
    x = torch.randn(2, 8)
    modules["x"](x).sum().backward()
    optimizer.step()
    rng = np.random.default_rng(3)
    order = rng.permutation(10)
    saved_scores = modules["x"].scores.detach().clone()
    saved_weights = modules["x"].weight.clone()
    save_checkpoint(
        tmp_path / "resume.pt",
        modules,
        optimizer,
        config_hash="a",
        step=1,
        order=order,
        cursor=4,
        rng=rng,
        best_rank=(9, 8, -40),
    )
    expected_next = rng.permutation(7)
    expected_torch = torch.rand(7)
    with torch.no_grad():
        modules["x"].scores.add_(1)
    step, restored_order, cursor, restored_rng, rank = restore_checkpoint(
        tmp_path / "resume.pt",
        modules,
        optimizer,
        config_hash="a",
        device="cpu",
    )
    assert (step, cursor, rank) == (1, 4, (9, 8, -40))
    assert np.array_equal(restored_order, order)
    assert np.array_equal(restored_rng.permutation(7), expected_next)
    assert torch.equal(modules["x"].scores, saved_scores)
    assert torch.equal(modules["x"].weight, saved_weights)
    assert len(optimizer.state[modules["x"].scores]) == 3
    assert torch.equal(torch.rand(7), expected_torch)
    with pytest.raises(ValueError, match="candidate"):
        restore_checkpoint(
            tmp_path / "resume.pt", modules, optimizer, config_hash="changed", device="cpu"
        )
