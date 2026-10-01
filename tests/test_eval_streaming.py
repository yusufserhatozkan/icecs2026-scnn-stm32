"""Metric integrity tests using synthetic signals; no checkpoints or dataset reads."""
import copy
import json
import sys

import numpy as np
import pytest

from evaluation import eval_streaming as evaluation


def score(utterance_id="synthetic_0", pesq=None, composite=None):
    clean = np.array([0.3, -0.1, 0.5, -0.4], dtype=np.float32)
    noisy = clean + np.array([0.1, 0.1, -0.1, 0.1], dtype=np.float32)
    enhanced = clean + np.array([-0.01, 0.02, 0.02, -0.01], dtype=np.float32)
    return evaluation._score_utterance(
        utterance_id, noisy, clean, enhanced, 16000,
        pesq or (lambda *args: 3.0), lambda *args, **kwargs: 0.9,
        composite or (lambda *args: [3.0, 2.0, 4.0, 1.0]))


@pytest.mark.parametrize("failed_side", ("noisy", "enh"))
def test_metric_exception_does_not_hide_other_side(failed_side):
    calls = []

    def pesq(*args):
        side = ("noisy", "enh")[len(calls)]
        calls.append(side)
        if side == failed_side:
            raise RuntimeError("synthetic PESQ failure")
        return 3.2

    record = score(pesq=pesq)
    results = evaluation._aggregate_records([record])
    assert calls == ["noisy", "enh"]
    assert record["failures"] == [{"metric": f"{failed_side}_pesq", "category": "RuntimeError",
                                    "message": "synthetic PESQ failure"}]
    assert results["successful_counts"][f"{failed_side}_pesq"] == 0
    assert results["paired_counts"]["pesq"] == 0
    assert results["successful_counts"][f"{failed_side}_stoi"] == 1


def test_composite_uses_each_finite_denominator():
    records = [score("a"), score("b", composite=lambda *args: [5.0, np.nan, np.inf, 3.0])]
    results = evaluation._aggregate_records(records)
    for side in ("noisy", "enh"):
        assert results[f"{side}_comp_pesq"] == 4.0
        assert results[f"{side}_comp_ovrl"] == 2.0
        assert results[f"{side}_comp_sig"] == 4.0
        assert results[f"{side}_comp_bak"] == 2.0
        assert results["successful_counts"][f"{side}_comp_ovrl"] == 1
    assert len(records[1]["failures"]) == 4
    assert all(f["category"] == "NonFiniteMetric" for f in records[1]["failures"])


def test_all_composite_failures_produce_no_aggregate():
    def composite(*args):
        raise ValueError("synthetic invalid signal")

    results = evaluation._aggregate_records([score(composite=composite)])
    assert results["noisy_comp_pesq"] is None
    assert results["successful_counts"]["noisy_comp_pesq"] == 0
    assert len(results["per_utterance"][0]["failures"]) == 8


def test_sisnr_zero_projection_is_recorded_without_changing_formula():
    clean = np.array([1.0, -1.0], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        record = evaluation._score_utterance("silent", np.zeros(2), clean, np.zeros(2),
                                             16000, lambda *a: 3.0, lambda *a, **k: 0.9,
                                             lambda *a: [3.0, 3.0, 3.0, 3.0])
    assert record["metrics"]["enh_sisnr"] is None
    assert record["failures"][0]["category"] == "NonFiniteMetric"


def test_successful_metrics_keep_original_values_and_options():
    calls = []
    clean = np.array([1.0, -1.0, 0.2], dtype=np.float32)
    noisy = clean + np.array([0.3, 0.2, -0.1], dtype=np.float32)

    def pesq(sr, reference, signal, mode):
        calls.append((sr, mode))
        np.testing.assert_array_equal(reference, clean)
        return 3.25

    def stoi(reference, signal, sr, extended):
        assert sr == 16000 and extended is False
        return 0.85

    record = evaluation._score_utterance("a", noisy, clean, noisy, 16000, pesq, stoi,
                                         lambda *args: [3.25, 2.0, 4.0, 1.0])
    assert calls == [(16000, "wb"), (16000, "wb")]
    assert record["metrics"]["noisy_sisnr"] == evaluation._sisnr(noisy, clean)
    assert record["metrics"]["enh_pesq"] == 3.25
    assert not record["failures"]


def complete_records():
    template = score()
    records = [copy.deepcopy(template) for _ in range(824)]
    for idx, record in enumerate(records):
        record["id"] = f"synthetic_{idx}"
    return records


@pytest.mark.parametrize("metric", evaluation.REQUIRED_METRICS)
@pytest.mark.parametrize("side", ("noisy", "enh"))
def test_paper_gate_requires_every_required_finite_result(metric, side):
    records = complete_records()
    records[-1]["metrics"][f"{side}_{metric}"] = None
    results = evaluation._aggregate_records(records)
    assert not results["paper_quality_eligible"]
    assert results["successful_counts"][f"{side}_{metric}"] == 823


def test_paper_gate_requires_exact_count_and_unique_ids():
    records = complete_records()
    assert evaluation._aggregate_records(records)["paper_quality_eligible"]
    assert not evaluation._aggregate_records(records[:-1])["paper_quality_eligible"]
    records[-1]["id"] = records[0]["id"]
    assert not evaluation._aggregate_records(records)["paper_quality_eligible"]


def test_empty_results_serialize_as_null(tmp_path):
    results = evaluation._aggregate_records([])
    target = tmp_path / "empty.txt"
    evaluation._save_results(results, str(target))
    text = (tmp_path / "empty.txt.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    saved = json.loads(text)
    assert saved["noisy_sisnr"] is None
    assert not saved["paper_quality_eligible"]


@pytest.mark.parametrize("suffix", ("", ".json"))
def test_cli_refuses_existing_result_before_loading_checkpoint(tmp_path, monkeypatch, suffix):
    target = tmp_path / "existing.txt"
    existing = tmp_path / (target.name + suffix)
    existing.write_bytes(b"historical evidence")
    monkeypatch.setattr(sys, "argv", ["eval", "--ckpt_path", "unused", "--hdf5_path", "unused",
                                      "--output_path", str(target)])
    monkeypatch.setattr(evaluation, "evaluate", lambda *a: pytest.fail("must not evaluate"))
    with pytest.raises(FileExistsError):
        evaluation.main()
    assert existing.read_bytes() == b"historical evidence"


def test_incomplete_cli_saves_per_utterance_evidence_then_fails(tmp_path, monkeypatch):
    target = tmp_path / "incomplete.txt"
    results = evaluation._aggregate_records([score()])
    monkeypatch.setattr(sys, "argv", ["eval", "--ckpt_path", "unused", "--hdf5_path", "unused",
                                      "--output_path", str(target)])
    monkeypatch.setattr(evaluation, "evaluate", lambda *a: results)
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
    assert json.loads((tmp_path / "incomplete.txt.json").read_text())["per_utterance"][0]["id"] == "synthetic_0"
    with pytest.raises(FileExistsError):
        evaluation._save_results(results, str(target))
