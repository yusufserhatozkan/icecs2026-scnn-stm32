"""Tests for ONNX export and numerical comparison."""
import os

import numpy as np
import pytest
import torch

from dpsnn.models.dp_binary_net import StreamSpikeNet
from export.export_to_onnx import ExportWrapper, export_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_model() -> StreamSpikeNet:
    """Minimal StreamSpikeNet for fast tests.

    Chosen so that:
      feature_steps  = (120 - 80) // 40 + 1 = 2
      context_step   = 40 // 40              = 1
      time_steps     = 2 - 1x1              = 1   (exactly 1 output frame)
      output_size    = (1 - 1) x 40 + 80   = 80  samples
    """
    return StreamSpikeNet(
        input_dim=120,
        context_dim=40,
        sr=16000,
        L=80,
        stride=40,
        N=16,
        B=16,
        H=16,
        X=1,
        learning_rate=1e-5,
    )


# ---------------------------------------------------------------------------
# ExportWrapper tests
# ---------------------------------------------------------------------------

def test_export_wrapper_output_shape():
    """ExportWrapper should return enhanced audio with the expected shape."""
    model = make_tiny_model()
    model.eval()
    wrapper = ExportWrapper(model)

    dummy_x = torch.randn(1, 120)
    with torch.no_grad():
        enhanced = wrapper(dummy_x)

    assert enhanced.shape == torch.Size([1, 80]), (
        f"Expected (1, 80), got {enhanced.shape}"
    )


def test_export_wrapper_batch_shape():
    """ExportWrapper should handle batch_size > 1."""
    model = make_tiny_model()
    model.eval()
    wrapper = ExportWrapper(model)

    dummy_x = torch.randn(2, 120)
    with torch.no_grad():
        enhanced = wrapper(dummy_x)

    assert enhanced.shape == torch.Size([2, 80]), (
        f"Expected (2, 80), got {enhanced.shape}"
    )


def test_export_wrapper_deterministic():
    """Identical inputs should not carry neuron state between calls."""
    model = make_tiny_model()
    model.eval()
    wrapper = ExportWrapper(model)

    dummy_x = torch.randn(1, 120)
    with torch.no_grad():
        out1 = wrapper(dummy_x.clone())
        out2 = wrapper(dummy_x.clone())

    assert torch.allclose(out1, out2, atol=1e-6), (
        "Two identical forward passes produced different outputs — model may be stateful."
    )


# ---------------------------------------------------------------------------
# ONNX export + validation tests
# ---------------------------------------------------------------------------

def test_onnx_export_creates_valid_file(tmp_path):
    """export_model should write a syntactically valid ONNX graph."""
    import onnx

    model = make_tiny_model()
    model.eval()
    wrapper = ExportWrapper(model)
    output_path = str(tmp_path / "test.onnx")
    dummy_x = torch.randn(1, 120)

    export_model(wrapper, dummy_x, output_path)

    assert os.path.exists(output_path), "ONNX file was not written"
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def test_onnx_output_matches_pytorch(tmp_path):
    """ONNX Runtime output should match PyTorch output within 1e-4."""
    from export.validate_onnx import compare_outputs, run_onnx_inference

    model = make_tiny_model()
    model.eval()
    wrapper = ExportWrapper(model)
    output_path = str(tmp_path / "test.onnx")
    dummy_x = torch.randn(1, 120)

    export_model(wrapper, dummy_x, output_path)

    with torch.no_grad():
        pt_output = wrapper(dummy_x).numpy()

    ort_output = run_onnx_inference(output_path, dummy_x.numpy())
    max_diff = compare_outputs(pt_output, ort_output)

    assert max_diff < 1e-4, (
        f"Max output difference {max_diff:.2e} exceeds 1e-4 — numerical mismatch."
    )


# ---------------------------------------------------------------------------
# compare_outputs unit tests
# ---------------------------------------------------------------------------

def test_compare_outputs_zero_for_identical():
    """compare_outputs returns 0.0 for identical arrays."""
    from export.validate_onnx import compare_outputs
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert compare_outputs(a, a.copy()) == 0.0


def test_compare_outputs_detects_mismatch():
    """compare_outputs returns the correct max absolute difference."""
    from export.validate_onnx import compare_outputs
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.5]], dtype=np.float32)
    diff = compare_outputs(a, b)
    assert abs(diff - 0.5) < 1e-6, f"Expected diff 0.5, got {diff}"
