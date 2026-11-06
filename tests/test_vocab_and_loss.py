import json
from types import SimpleNamespace

import pytest
import torch

from tenpercent.io import write_json
from tenpercent.model import LabelTrie
from tenpercent.teacher import distillation_loss, eda_variants, pseudo_label_variants
from tenpercent.vocabulary import Vocabulary


def test_recursive_vocabulary_preserves_pieces_and_mapping(tmp_path):
    vocabulary = Vocabulary([0, 1, 2], {3: (0, 1), 4: (3, 2)})
    assert vocabulary.expand([4, 0]) == [0, 1, 2, 0]
    assert vocabulary.encode([4]) == [0, 1, 2]
    vocabulary.save(tmp_path / "vocab.json")
    assert Vocabulary.load(tmp_path / "vocab.json").expand([4]) == [0, 1, 2]


def test_trie_allows_eos_and_continuation_for_prefix_labels():
    trie = LabelTrie([[1, 9], [1, 2, 9], [3, 9]], eos=9)
    assert trie.allowed([]) == [1, 3]
    assert trie.allowed([1]) == [2, 9]
    assert trie.allowed([1, 2]) == [9]


def test_bucket_kl_matches_full_distribution_when_each_token_is_covered():
    torch.manual_seed(3)
    teacher = torch.randn(3, 5)
    student = torch.randn(3, 5, requires_grad=True)
    p = (teacher / 2).softmax(-1)
    top_p, top_ids = p.topk(4, dim=-1)
    tail = 1 - top_p.sum(-1)
    target = torch.tensor([1, 2, 3])
    actual = distillation_loss(student, top_ids, top_p, tail, target)
    expected = 4 * torch.nn.functional.kl_div(
        (student / 2).log_softmax(-1),
        p,
        reduction="batchmean",
    ) + 0.3 * torch.nn.functional.cross_entropy(student, target)
    assert torch.allclose(actual, expected, atol=2e-6)
    actual.backward()
    assert torch.isfinite(student.grad).all()


def test_eda_is_deterministic_and_train_derived():
    train = [{"id": "banking77:train:1", "text": "one two three four five", "label": "a"}]
    variants = eda_variants(train, 8)
    assert variants == eda_variants(train, 8)
    assert len({r["text"] for r in variants}) == len(variants)
    assert all(r["parent_id"] == train[0]["id"] for r in variants)


def test_adaptation_plan_reads_only_train_and_refuses_changed_inputs(tmp_path, monkeypatch):
    from tenpercent import teacher

    rows = [{"id": "train:1", "text": "abc", "label": "a"}]

    def only_train(root, name, purpose):
        assert name == "banking/train" and purpose == "teacher"
        return rows

    monkeypatch.setattr(teacher, "load_split", only_train)
    monkeypatch.setattr(teacher, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(
        teacher,
        "TokenizerAdapter",
        lambda _: SimpleNamespace(prompt=lambda text: list(range(len(text)))),
    )
    monkeypatch.setattr(
        teacher, "CompactTokenizer", lambda *args: SimpleNamespace(label=lambda *args: [1, 2])
    )
    write_json(tmp_path / "data/manifest.json", {"labels": ["a"]})
    write_json(tmp_path / "assets/base/tokenizer.json", {})
    write_json(tmp_path / "assets/base/tokenizer_config.json", {})
    Vocabulary([0, 1, 2], {}).save(tmp_path / "runs/vocabulary.json")
    plan = teacher.prepare_adaptation(tmp_path, variants=0)
    assert plan["total_examples"] == 1
    assert plan["max_teacher_input_tokens_upper_bound"] == 4
    assert not plan["truncation"]
    assert teacher.prepare_adaptation(tmp_path, variants=0) == plan
    rows.append({"id": "train:2", "text": "ab", "label": "a"})
    with pytest.raises(PermissionError, match="amendment"):
        teacher.prepare_adaptation(tmp_path, variants=0)


def test_interrupted_pseudo_labeling_continues_only_unfinished_prefix(tmp_path):
    rows = [{"id": str(i), "text": str(i), "label": "a", "parent_id": "parent"} for i in range(3)]
    calls = []

    def first_run(text, mode):
        calls.append(text)
        if text == "1":
            raise RuntimeError("Interrupted")
        return "b"

    path = tmp_path / "adaptation.jsonl"
    with pytest.raises(RuntimeError, match="Interrupted"):
        pseudo_label_variants(path, rows, SimpleNamespace(labels=["a", "b"], predict=first_run))
    assert len(path.read_text().splitlines()) == 1
    calls.clear()

    def resumed(text, mode):
        calls.append(text)
        return "b"

    records = pseudo_label_variants(path, rows, SimpleNamespace(labels=["a", "b"], predict=resumed))
    assert calls == ["1", "2"]
    assert all(row["label"] == "b" for row in records)
    assert all(row["label"] == "a" for row in rows)
    assert len(path.read_text().splitlines()) == 3
    corrupted = json.loads(path.read_text().splitlines()[0])
    corrupted["text"] = "changed"
    path.write_text(json.dumps(corrupted) + "\n")
    with pytest.raises(ValueError, match="journal differs"):
        pseudo_label_variants(path, rows, SimpleNamespace(labels=["a", "b"], predict=resumed))
