"""Evaluate the streaming StreamSpikeNet on the full VoiceBank-DEMAND test set.

Runs PyTorch one 80-sample frame at a time, with recurrent state carried within
each one-second chunk and reset between chunks. This is not an MCU measurement.

Usage:
    python evaluation/eval_streaming.py \\
        --ckpt_path models/n64.ckpt \\
        --hdf5_path data/voicebank/save/test.hdf5 \\
        --output_path data/eval_n64/metrics.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export.export_to_onnx import load_from_checkpoint
from export.export_streaming import StreamingWrapper


METRICS = ("sisnr", "pesq", "stoi", "comp_pesq", "comp_ovrl", "comp_sig", "comp_bak")
REQUIRED_METRICS = ("sisnr", "pesq", "stoi")
OFFICIAL_TEST_COUNT = 824


def _record_metric(record: dict, names: tuple, calculate) -> None:
    """Keep every finite result, and record each failure without changing its value."""
    try:
        values = np.atleast_1d(np.asarray(calculate(), dtype=np.float64))
        if values.shape != (len(names),):
            raise ValueError(f"Expected {len(names)} metric values, got {values.shape}")
    except Exception as exc:
        for name in names:
            record["metrics"][name] = None
            record["failures"].append({"metric": name, "category": type(exc).__name__,
                                       "message": str(exc)})
        return
    for name, value in zip(names, values):
        if np.isfinite(value):
            record["metrics"][name] = float(value)
        else:
            record["metrics"][name] = None
            record["failures"].append({"metric": name, "category": "NonFiniteMetric",
                                       "message": f"Metric returned {value}"})


def _score_utterance(utterance_id, noisy, clean, enhanced, sr,
                     eval_pesq, eval_stoi, eval_composite) -> dict:
    record = {"id": str(utterance_id), "length": len(clean), "metrics": {}, "failures": []}
    for side, signal in (("noisy", noisy), ("enh", enhanced)):
        # Evaluate each side independently: a noisy failure cannot hide an enhanced result.
        _record_metric(record, (f"{side}_sisnr",), lambda: _sisnr(signal, clean))
        _record_metric(record, (f"{side}_pesq",), lambda: eval_pesq(sr, clean, signal, "wb"))
        _record_metric(record, (f"{side}_stoi",),
                       lambda: eval_stoi(clean, signal, sr, extended=False))
        _record_metric(record, tuple(f"{side}_{name}" for name in METRICS[3:]),
                       lambda: eval_composite(clean, signal, sr))
    return record


def _aggregate_records(records: list) -> dict:
    result = {"n_utterances": len(records), "successful_counts": {}, "paired_counts": {},
              "per_utterance": records, "paper_quality_eligible": False,
              "paper_rejection_reasons": []}
    for metric in METRICS:
        for side in ("noisy", "enh"):
            key = f"{side}_{metric}"
            values = [r["metrics"][key] for r in records if r["metrics"][key] is not None]
            result["successful_counts"][key] = len(values)
            # Preserve the historical scalar mean and composite accumulation conventions.
            result[key] = (float(sum(values) / len(values)) if metric.startswith("comp_")
                           else float(np.mean(values))) if values else None
        result["paired_counts"][metric] = sum(
            r["metrics"][f"noisy_{metric}"] is not None and
            r["metrics"][f"enh_{metric}"] is not None for r in records)
    reasons = result["paper_rejection_reasons"]
    if len(records) != OFFICIAL_TEST_COUNT:
        reasons.append(f"Expected {OFFICIAL_TEST_COUNT} utterances, got {len(records)}")
    if len({r["id"] for r in records}) != len(records):
        reasons.append("Utterance IDs are not unique")
    for metric in REQUIRED_METRICS:
        for side in ("noisy", "enh"):
            key = f"{side}_{metric}"
            if result["successful_counts"][key] != OFFICIAL_TEST_COUNT:
                reasons.append(f"{key}: {result['successful_counts'][key]} finite results; "
                               f"{OFFICIAL_TEST_COUNT} required")
    result["paper_quality_eligible"] = not reasons
    return result


def _result_paths(output_path: str) -> tuple:
    path = Path(output_path)
    return path, path.with_name(path.name + ".json")


def _require_new_results(output_path: str) -> None:
    for path in _result_paths(output_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")


def _save_results(results: dict, output_path: str) -> None:
    _require_new_results(output_path)
    summary_path, details_path = _result_paths(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON records unavailable values as null, never a misleading NaN aggregate.
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    with details_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    with summary_path.open("x", encoding="utf-8") as stream:
        for key, value in results.items():
            if key not in ("per_utterance", "protocol"):
                stream.write(f"{key}={value}\n")


def _sisnr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    dot = np.sum(est * ref)
    proj = dot / (np.sum(ref ** 2) + 1e-8) * ref
    noise = est - proj
    return float(10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-8)))


def _normalize(est: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(est))
    return est / peak if peak > 0 else est


def run_streaming_chunk(wrapper: StreamingWrapper, chunk: np.ndarray) -> np.ndarray:
    """Run streaming inference on one fixed-length chunk (input_dim samples).

    Matches the batch model exactly: processes feature_steps frames with warmup
    reset at frame context_step, produces (time_steps * stride + stride) = 16000
    enhanced samples.  State is reset between chunks (same as batch model).
    """
    L = wrapper.L
    stride = wrapper.stride
    context_step = wrapper.context_step
    B = wrapper.srnn_readout.output_dim
    feature_steps = (len(chunk) - L) // stride + 1

    audio_t = torch.from_numpy(chunk).float()

    context_win = torch.zeros(1, B, context_step)
    v_plif      = torch.zeros(1, B, 1)
    mem_readout = torch.zeros(1, B)
    ola_tail    = torch.zeros(1, stride)

    out_chunks = []
    with torch.no_grad():
        for t in range(feature_steps):
            frame = audio_t[t * stride: t * stride + L].unsqueeze(0)
            enhanced, context_win, v_plif, mem_readout, ola_tail = wrapper(
                frame, context_win, v_plif, mem_readout, ola_tail)

            # After context_step warmup frames, reset membrane + OLA state.
            # Matches the batch model which initialises those to zero at local_t=0.
            if t == context_step - 1:
                v_plif      = torch.zeros(1, B, 1)
                mem_readout = torch.zeros(1, B)
                ola_tail    = torch.zeros(1, stride)

            if t >= context_step:
                out_chunks.append(enhanced.squeeze(0).numpy())

        # Flush the final OLA tail (last frame's non-overlapping half).
        out_chunks.append(ola_tail.squeeze(0).numpy())

    return np.concatenate(out_chunks)  # matches batch model output size


def _build_chunks(audio: np.ndarray, input_dim: int, output_size: int) -> np.ndarray:
    """Split utterances into padded chunks with an initial context prefix."""
    context_size = input_dim - output_size
    remainder = len(audio) % output_size
    if remainder:
        audio = np.pad(audio, (0, output_size - remainder))
    target_outputs = len(audio)
    padded = np.pad(audio, (context_size, 0))
    chunks = [padded[t:t + input_dim] for t in range(0, target_outputs, output_size)]
    return np.stack(chunks).astype(np.float32)


def evaluate(ckpt_path: str, hdf5_path: str, sr: int = 16000) -> dict:
    from pesq import pesq as eval_pesq
    from pystoi import stoi as eval_stoi
    from dpsnn.data.metrics import eval_composite

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_from_checkpoint(ckpt_path)
    model.eval()
    wrapper = StreamingWrapper(model)
    wrapper.eval()

    input_dim   = model.hparams["input_dim"]                           # 16160
    output_size = (model.time_steps - 1) * model.stride + model.L     # 16000

    print(f"Model: input_dim={input_dim}, L={model.L}, stride={model.stride}, "
          f"context_step={model.context_step}, time_steps={model.time_steps}")
    print(f"Evaluating on {hdf5_path} ...")

    records = []

    with h5py.File(hdf5_path, "r") as f:
        if sr != 16000 or f.attrs.get("sr") != sr or f.attrs.get("channels") != 1:
            raise ValueError("Streaming PESQ-WB evaluation requires a verified mono 16 kHz cache")
        total = len(f)
        for idx in range(total):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}]", flush=True)

            audio = f[str(idx)]["noisy"][()].astype(np.float32).squeeze()
            clean = f[str(idx)]["clean"][()].astype(np.float32).squeeze()
            audio_length = int(f[str(idx)].attrs["length"])
            utterance_id = f[str(idx)].attrs["ID"]
            if isinstance(utterance_id, bytes):
                utterance_id = utterance_id.decode("utf-8")
            if (audio.ndim != 1 or clean.ndim != 1 or audio_length <= 0 or
                    len(audio) != audio_length or len(clean) != audio_length or
                    not np.isfinite(audio).all() or not np.isfinite(clean).all()):
                raise ValueError(f"{utterance_id}: invalid, nonfinite, or mismatched input audio")

            # Split utterance into fixed 1-second chunks — identical to batch eval.
            chunks = _build_chunks(audio, input_dim, output_size)  # (n_chunks, input_dim)
            enhanced_chunks = np.stack([
                run_streaming_chunk(wrapper, c) for c in chunks
            ])  # (n_chunks, output_size)

            enhanced = enhanced_chunks.flatten()[:audio_length]
            noisy_trim = audio[:audio_length]
            clean_trim = clean[:audio_length]

            if len(enhanced) < audio_length:
                enhanced = np.pad(enhanced, (0, audio_length - len(enhanced)))

            enhanced = _normalize(enhanced)

            records.append(_score_utterance(utterance_id, noisy_trim, clean_trim, enhanced,
                                            sr, eval_pesq, eval_stoi, eval_composite))

    results = _aggregate_records(records)
    packages = {}
    for name in ("numpy", "torch", "pesq", "pystoi", "scipy", "h5py"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    results["protocol"] = {
        "checkpoint": str(ckpt_path), "hdf5": str(hdf5_path), "sample_rate": sr,
        "input_dim": input_dim, "output_size": output_size,
        "frame_samples": model.L, "stride_samples": model.stride,
        "context_frames": model.context_step, "package_versions": packages,
        "implementation": "chunked PyTorch streaming; four states reset per chunk",
        "alignment": "discard context frames; reset membrane/readout/OLA after warmup",
        "padding": "zero-pad initial context and final incomplete chunk; flush OLA tail",
        "cropping": "concatenate enhanced chunks and crop to stored utterance length",
        "normalization": "enhanced utterance peak only; clean/noisy unchanged",
        "sisnr": "zero-mean projection; historical 1e-8 denominator epsilons",
        "pesq_mode": "wb", "stoi_extended": False,
        "aggregation": "unweighted mean of finite per-utterance values; separate counts",
    }
    return results


def _print_results(results: dict) -> None:
    n = results["n_utterances"]
    print(f"\n{'='*55}")
    print(f"  Streaming eval  ({n} utterances)")
    print(f"{'='*55}")
    print(f"{'Metric':<22} {'Noisy':>10} {'Enhanced':>10}")
    print(f"{'-'*44}")
    for label, metric in zip(("SI-SNR (dB)", "PESQ (wb)", "STOI", "Comp PESQ", "Comp OVRL",
                              "Comp SIG", "Comp BAK"), METRICS):
        values = [results[f"{side}_{metric}"] for side in ("noisy", "enh")]
        formatted = ["unavailable" if value is None else f"{value:.3f}" for value in values]
        print(f"{label:<22} {formatted[0]:>10} {formatted[1]:>10}")
    print(f"{'='*55}")
    print(f"Finite result counts: {results['successful_counts']}")
    print(f"Paper-quality eligible: {results['paper_quality_eligible']}")
    for reason in results["paper_rejection_reasons"]:
        print(f"REJECTED: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--output_path", required=True,
                        help="New text result path; full records saved beside it as <path>.json")
    args = parser.parse_args()

    _require_new_results(args.output_path)
    results = evaluate(args.ckpt_path, args.hdf5_path)
    _print_results(results)

    _save_results(results, args.output_path)
    print(f"\nMetrics saved -> {args.output_path}")
    if not results["paper_quality_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
