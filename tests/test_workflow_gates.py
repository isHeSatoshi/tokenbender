import pytest

from tenpercent import workflow


def test_unreleased_final_never_starts_a_session_or_opens_sealed_rows(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Exam guard must not even be entered")

    monkeypatch.setattr(workflow, "ExamSession", forbidden)
    with pytest.raises(PermissionError, match="not released"):
        workflow.final_evaluate(tmp_path, tmp_path / "nonexistent-attestation.json", device="cpu")
    assert not (tmp_path / "runs/exam_receipt.json").exists()


def test_final_freeze_refuses_unvalidated_pipeline(tmp_path):
    with pytest.raises(PermissionError, match="source-isolated"):
        workflow.make_freeze_manifest(tmp_path)
    assert not (tmp_path / "final_manifest.json").exists()
