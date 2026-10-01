"""Receive streaming audio from the STM32U585 over UART.

Receives enhanced audio streamed from the STM32U585 over UART:
    AUDIO_START\\r\\n
    <399 * 40 * 4 bytes of float32 audio>
    TIMING:<ms/frame> ms/frame RTF:<rtf>\\r\\n
    AUDIO_END\\r\\n

Then computes SI-SNR vs the clean reference and compares to the Python
ONNX reference SI-SNR from deploy/reference_sisnr.txt.

Usage (dpsnn env):
    python tools/mcu_audio_receiver.py \\
        --port COM3 \\
        --clean data/clean_testset_wav_16k/p232_006.wav \\
        --ref_bin deploy/test_enhanced_ref.bin \\
        --out deploy/mcu_enhanced.wav
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import time
import wave

import numpy as np
import serial
import soundfile as sf

def load_n_frames(ref_bin: str) -> int:
    ref_txt = os.path.join(os.path.dirname(ref_bin), "reference_sisnr.txt")
    if os.path.exists(ref_txt):
        for line in open(ref_txt):
            if line.startswith("n_frames:"):
                return int(line.split(":")[1].strip())
    return 399  # fallback


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


def save_wav(path: str, audio: np.ndarray, sr: int = 16000) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    default="COM3",
                        help="Serial port (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud",    type=int, default=115200)
    parser.add_argument("--clean",   required=True,
                        help="Clean reference WAV (p232_006)")
    parser.add_argument("--ref_bin", required=True,
                        help="Python ONNX reference enhanced binary (.bin)")
    parser.add_argument("--out",     default="deploy/mcu_enhanced.wav")
    args = parser.parse_args()

    N_FRAMES = load_n_frames(args.ref_bin)
    N_FLOATS = N_FRAMES * 40
    N_BYTES  = N_FLOATS * 4
    print(f"Expecting {N_FRAMES} frames ({N_FLOATS} samples, {N_BYTES} bytes)")

    # ---- Load reference SI-SNR and clean audio ----
    clean_raw, sr = sf.read(args.clean, dtype="float32")
    assert sr == 16000
    clean = clean_raw.squeeze()[:N_FLOATS]   # trim to match output length

    ref_enhanced = np.fromfile(args.ref_bin, dtype=np.float32)
    ref_sisnr    = si_snr(ref_enhanced, clean)
    print(f"Python ONNX reference SI-SNR: {ref_sisnr:.2f} dB")

    # ---- Open serial port ----
    print(f"\nOpening {args.port} @ {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=120)
    time.sleep(0.5)   # let board initialise

    # ---- Wait for AUDIO_START marker ----
    print("Waiting for AUDIO_START marker (press RESET on board if needed)...")
    while True:
        line = ser.readline()
        if b"AUDIO_START" in line:
            print("Got AUDIO_START, receiving audio...")
            break

    # ---- Receive float32 audio frame-by-frame ----
    raw = b""
    received_frames = 0
    while len(raw) < N_BYTES:
        chunk = ser.read(N_BYTES - len(raw))
        raw += chunk
        new_frames = len(raw) // 160
        if new_frames > received_frames:
            received_frames = new_frames
            if received_frames % 50 == 0:
                print(f"  {received_frames}/{N_FRAMES} frames received...")

    audio = np.frombuffer(raw[:N_BYTES], dtype=np.float32)
    print(f"Received {len(audio)} samples ({len(audio)/16000:.2f} s)")

    # ---- Read timing line and END marker ----
    timing_str = ""
    for _ in range(5):
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line.startswith("TIMING:"):
            timing_str = line
        if "AUDIO_END" in line:
            break
    ser.close()

    if timing_str:
        print(f"\n{timing_str}")
    else:
        print("\n(No timing line received)")

    # ---- Compute MCU SI-SNR ----
    mcu_sisnr = si_snr(audio, clean)
    delta     = mcu_sisnr - ref_sisnr

    print(f"\n--- Results ---")
    print(f"MCU SI-SNR:    {mcu_sisnr:.2f} dB")
    print(f"Python ref:    {ref_sisnr:.2f} dB")
    print(f"Delta:         {delta:+.2f} dB  {'PASS' if abs(delta) < 0.1 else 'WARN'}")

    # ---- Save MCU enhanced WAV ----
    # Peak-normalise before saving; model output has large absolute values
    audio_peak = np.max(np.abs(audio)) + 1e-8
    save_wav(args.out, audio / audio_peak)
    print(f"\nSaved MCU enhanced audio: {args.out}")

    # ---- Write results summary ----
    summary_path = os.path.join(os.path.dirname(args.out), "mcu_test_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"utterance:      {os.path.splitext(os.path.basename(args.clean))[0]}\n")
        f.write(f"n_frames:       {N_FRAMES}\n")
        f.write(f"mcu_si_snr:     {mcu_sisnr:.4f} dB\n")
        f.write(f"ref_si_snr:     {ref_sisnr:.4f} dB\n")
        f.write(f"delta:          {delta:+.4f} dB\n")
        if timing_str:
            f.write(f"timing:         {timing_str}\n")
    print(f"Results written: {summary_path}")


if __name__ == "__main__":
    main()
