"""Numerically validate an ONNX model against its PyTorch checkpoint.

Usage (from repo root):
    python export/validate_onnx.py \\
        --ckpt_path egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt \\
        --onnx_path export/dpsnn_pretrained.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export.export_to_onnx import ExportWrapper, load_from_checkpoint


def run_onnx_inference(onnx_path: str, input_array: np.ndarray) -> np.ndarray:
    """Run inference through ONNX Runtime."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"noisy_audio": input_array.astype(np.float32)})
    return outputs[0]


def compare_outputs(pt_output: np.ndarray, ort_output: np.ndarray) -> float:
    """Max absolute element-wise difference between two arrays."""
    return float(np.max(np.abs(pt_output - ort_output)))


def validate(
    ckpt_path: str,
    onnx_path: str,
    warn_threshold: float = 1e-3,
    fail_threshold: float = 1e-1,
) -> bool:
    """Validate an ONNX model numerically against its PyTorch source."""
    model = load_from_checkpoint(ckpt_path)
    wrapper = ExportWrapper(model)

    input_dim = model.hparams["input_dim"]
    dummy_input = torch.randn(1, input_dim)

    with torch.no_grad():
        pt_output = wrapper(dummy_input).numpy()

    ort_output = run_onnx_inference(onnx_path, dummy_input.numpy())
    max_diff = compare_outputs(pt_output, ort_output)

    print(f"Max absolute difference (PyTorch vs ONNX Runtime): {max_diff:.2e}")

    if max_diff < warn_threshold:
        print("PASS - outputs match within 1e-3")
        return True
    elif max_diff < fail_threshold:
        print(f"WARN - max diff {max_diff:.2e} exceeds {warn_threshold:.2e}. Investigate before deploying.")
        return True
    else:
        print(f"FAIL - max diff {max_diff:.2e} exceeds {fail_threshold:.2e}. ONNX export not numerically equivalent.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ONNX Runtime output against PyTorch checkpoint")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--onnx_path", required=True)
    args = parser.parse_args()

    success = validate(args.ckpt_path, args.onnx_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
