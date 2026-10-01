"""Prepare an audio fixture for the STM32U585.

1. Load noisy/clean WAV pair, normalize, pad/truncate to 16000 samples.
2. Write test_utterance.h + test_utterance.c for embedding in STM32 Flash.
3. Run Python ONNX streaming inference → reference enhanced WAV + SI-SNR.

Usage (from repo root, dpsnn env):
    python tools/wav_to_c_array.py \\
        --noisy data/noisy_testset_wav_16k/p232_006.wav \\
        --clean  data/clean_testset_wav_16k/p232_006.wav  \\
        --onnx   export/dpsnn_streaming_xcubeai.onnx \\
        --out_c  ../Stm_deployment/X-CUBE-AI/App \\
        --out_wav deploy/
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import wave

import numpy as np
import onnxruntime as ort
import soundfile as sf

FRAME  = 80   # encoder window
STRIDE = 40   # hop size


# ---------------------------------------------------------------------------
# SI-SNR
# ---------------------------------------------------------------------------
def si_snr(estimate: np.ndarray, target: np.ndarray) -> float:
    est = estimate - estimate.mean()
    tgt = target   - target.mean()
    dot = np.dot(est, tgt)
    tgt_power = np.dot(tgt, tgt) + 1e-8
    s_target = (dot / tgt_power) * tgt
    e_noise  = est - s_target
    return 10.0 * math.log10(
        (np.dot(s_target, s_target) + 1e-8) /
        (np.dot(e_noise,  e_noise)  + 1e-8)
    )


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------
def save_wav(path: str, audio: np.ndarray, sr: int = 16000) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# C array generation
# ---------------------------------------------------------------------------
def write_c_array(out_dir: str, audio: np.ndarray, file_id: str) -> None:
    n_samples = len(audio)
    n_frames  = (n_samples - FRAME) // STRIDE + 1
    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, "test_utterance.h")
    c_path = os.path.join(out_dir, "test_utterance.c")

    with open(h_path, "w") as f:
        f.write(
            "#ifndef TEST_UTTERANCE_H\n"
            "#define TEST_UTTERANCE_H\n\n"
            "#include <stdint.h>\n\n"
            f"#define TEST_UTTERANCE_SAMPLES  {n_samples}U\n"
            f"#define TEST_UTTERANCE_STRIDE   {STRIDE}U\n"
            f"#define TEST_UTTERANCE_FRAME    {FRAME}U\n"
            f"#define TEST_UTTERANCE_N_FRAMES {n_frames}U\n\n"
            "extern const float test_utterance[TEST_UTTERANCE_SAMPLES];\n\n"
            "#endif /* TEST_UTTERANCE_H */\n"
        )

    with open(c_path, "w") as f:
        f.write('#include "test_utterance.h"\n\n')
        f.write(
            f"/* {file_id} (noisy, female, VoiceBank-DEMAND, 16 kHz, normalized float32) */\n"
        )
        f.write(
            "__attribute__((section(\".rodata\")))\n"
            "const float test_utterance[TEST_UTTERANCE_SAMPLES] = {\n"
        )
        vals = audio.astype(np.float32).tolist()
        per_row = 8
        for i in range(0, len(vals), per_row):
            row = vals[i : i + per_row]
            f.write("    " + ", ".join(f"{v:.8f}f" for v in row))
            if i + per_row < len(vals):
                f.write(",")
            f.write("\n")
        f.write("};\n")

    print(f"[C array]  {h_path}")
    print(f"[C array]  {c_path}  ({os.path.getsize(c_path)//1024} KB)")


# ---------------------------------------------------------------------------
# ONNX streaming inference
# ---------------------------------------------------------------------------
def run_streaming_inference(noisy: np.ndarray, onnx_path: str) -> np.ndarray:
    n_frames = (len(noisy) - FRAME) // STRIDE + 1
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Infer channel count N from the context_win input shape
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    N = inputs["context_win"][1]

    ctx  = np.zeros((1, N, 4), dtype=np.float32)
    vp   = np.zeros((1, N, 1), dtype=np.float32)
    mem  = np.zeros((1, N),    dtype=np.float32)
    tail = np.zeros((1, 40),   dtype=np.float32)

    out_chunks: list[np.ndarray] = []

    for i in range(n_frames):
        start = i * STRIDE
        frame = noisy[start : start + FRAME].astype(np.float32).reshape(1, FRAME)

        feeds = {
            "frame":       frame,
            "context_win": ctx,
            "v_plif":      vp,
            "mem_readout": mem,
            "ola_tail":    tail,
        }
        enhanced, ctx, vp, mem, tail = sess.run(None, feeds)
        out_chunks.append(enhanced[0])

        if (i + 1) % 200 == 0:
            print(f"  frame {i+1}/{n_frames}")

    return np.concatenate(out_chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy",   required=True)
    parser.add_argument("--clean",   required=True)
    parser.add_argument("--onnx",    required=True)
    parser.add_argument("--out_c",   required=True,
                        help="Directory to write test_utterance.{h,c}")
    parser.add_argument("--out_wav", default="deploy",
                        help="Directory to write reference WAVs")
    args = parser.parse_args()

    # ---- Load audio ----
    noisy_raw, sr = sf.read(args.noisy, dtype="float32")
    clean_raw, _  = sf.read(args.clean, dtype="float32")
    assert sr == 16000, f"Expected 16 kHz, got {sr}"

    noisy_raw = noisy_raw.squeeze()
    clean_raw = clean_raw.squeeze()

    # Peak-normalize noisy (same as training/eval pipeline)
    peak = np.max(np.abs(noisy_raw)) + 1e-8
    noisy = (noisy_raw / peak).astype(np.float32)
    clean = clean_raw.astype(np.float32)

    n_samples = len(noisy)
    n_frames  = (n_samples - FRAME) // STRIDE + 1
    file_id   = os.path.splitext(os.path.basename(args.noisy))[0]

    print(f"Audio loaded: {n_samples} samples ({n_samples/16000:.2f}s, {n_frames} frames)")
    print(f"Input SI-SNR (noisy vs clean): {si_snr(noisy[:n_frames*STRIDE], clean[:n_frames*STRIDE]):.2f} dB")

    # ---- Write C array ----
    write_c_array(args.out_c, noisy, file_id)

    # ---- ONNX streaming inference ----
    print(f"\nRunning ONNX streaming inference ({n_frames} frames)...")
    enhanced = run_streaming_inference(noisy, args.onnx)

    n_out     = n_frames * STRIDE
    clean_trim = clean[:n_out]
    ref_sisnr = si_snr(enhanced, clean_trim)
    print(f"\nReference SI-SNR (enhanced vs clean): {ref_sisnr:.2f} dB")

    # ---- Save WAVs ----
    os.makedirs(args.out_wav, exist_ok=True)
    save_wav(os.path.join(args.out_wav, "test_noisy.wav"),    noisy)
    save_wav(os.path.join(args.out_wav, "test_clean.wav"),    clean)
    # Peak-normalise enhanced before saving (SI-SNR is scale-invariant, but WAV
    # must be in [-1,1]; model output can have large absolute values)
    enh_peak = np.max(np.abs(enhanced)) + 1e-8
    save_wav(os.path.join(args.out_wav, "test_enhanced_ref.wav"), enhanced / enh_peak)

    # Save raw float32 for easy MCU comparison
    enhanced.astype(np.float32).tofile(
        os.path.join(args.out_wav, "test_enhanced_ref.bin")
    )
    print(f"\nSaved WAVs + ref binary to {args.out_wav}/")

    # ---- Write reference SI-SNR to file ----
    ref_path = os.path.join(args.out_wav, "reference_sisnr.txt")
    with open(ref_path, "w") as f:
        f.write(f"utterance:   {file_id}\n")
        f.write(f"n_samples:   {n_samples}\n")
        f.write(f"n_frames:    {n_frames}\n")
        f.write(f"si_snr_ref:  {ref_sisnr:.4f} dB\n")
    print(f"Reference SI-SNR written to {ref_path}")


if __name__ == "__main__":
    main()
