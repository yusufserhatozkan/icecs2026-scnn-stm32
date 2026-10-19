"""Receive streaming audio from the STM32U585 over UART.

Receives enhanced audio streamed from the STM32U585 over UART:
    AUDIO_START\\r\\n
    <n_frames * 40 * 4 bytes of float32 audio>
    TIMING:<ms/frame> ms/frame RTF:<rtf>\\r\\n
    AUDIO_END\\r\\n

Compares raw samples with the ONNX reference supplied with --ref_bin and
reports waveform errors and SI-SNR. Saves the raw float32 capture alongside
a peak-normalized WAV for listening. Existing capture files are never replaced.

Usage (dpsnn env):
    python tools/mcu_audio_receiver.py \\
        --port COM3 \\
        --clean deploy/n64/test_clean.wav \\
        --ref_bin deploy/n64/test_enhanced_ref.bin \\
        --out data/board_capture/mcu_enhanced.wav
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

def load_n_frames(ref_bin: str) -> int:
    size = os.path.getsize(ref_bin)
    if size == 0 or size % (40 * 4):
        raise ValueError("Reference binary must contain complete 40-sample float32 frames")
    count = size // (40 * 4)
    ref_txt = os.path.join(os.path.dirname(ref_bin), "reference_sisnr.txt")
    if os.path.exists(ref_txt):
        for line in Path(ref_txt).read_text(encoding="utf-8").splitlines():
            if line.startswith("n_frames:"):
                if int(line.split(":")[1].strip()) != count:
                    raise ValueError("Reference frame count differs from its binary length")
    return count


def compare_waveforms(audio: np.ndarray, reference: np.ndarray) -> dict:
    """Compare aligned raw samples without rescaling or shifting either waveform."""
    audio = np.asarray(audio)
    reference = np.asarray(reference)
    if (audio.ndim != 1 or audio.size == 0 or audio.shape != reference.shape or
            not np.isfinite(audio).all() or not np.isfinite(reference).all()):
        raise ValueError("Waveforms must be finite, nonempty, one-dimensional and equally sized")
    difference = audio.astype(np.float64) - reference.astype(np.float64)
    return {"sample_count": int(audio.size),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "max_abs_error": float(np.max(np.abs(difference))),
            "mean_abs_error": float(np.mean(np.abs(difference)))}


def read_audio_bytes(connection, count: int) -> bytes:
    raw = bytearray()
    while len(raw) < count:
        chunk = connection.read(count - len(raw))
        if not chunk:
            raise TimeoutError(f"Audio transfer stopped after {len(raw)} of {count} bytes")
        raw.extend(chunk)
    return bytes(raw)


def capture_paths(output_path):
    output = Path(output_path)
    if output.suffix.lower() != ".wav":
        raise ValueError("Capture output must have a .wav extension")
    return output, output.with_suffix(".bin"), output.with_name("mcu_test_results.txt")


def require_new_capture(output_path):
    for path in capture_paths(output_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite capture: {path}")


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
    with open(path, "xb") as stream:
        with wave.open(stream, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())


def save_capture(output_path, audio, reference, clean, utterance, timing_str=""):
    require_new_capture(output_path)
    comparison = compare_waveforms(audio, reference)
    compare_waveforms(audio, clean)
    mcu_sisnr, ref_sisnr = si_snr(audio, clean), si_snr(reference, clean)
    wav_path, raw_path, summary_path = capture_paths(output_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(audio, dtype="<f4").tobytes()
    with raw_path.open("xb") as stream:
        stream.write(raw)
    save_wav(str(wav_path), audio / (np.max(np.abs(audio)) + 1e-8))
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(f"utterance:      {utterance}\n")
        stream.write(f"captured_utc:   {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"n_frames:       {len(audio) // 40}\n")
        stream.write(f"mcu_si_snr:     {mcu_sisnr:.8f} dB\n")
        stream.write(f"ref_si_snr:     {ref_sisnr:.8f} dB\n")
        stream.write(f"delta:          {mcu_sisnr - ref_sisnr:+.8f} dB\n")
        for name, value in comparison.items():
            stream.write(f"{name}: {value}\n")
        stream.write(f"raw_float32:    {raw_path.name}\n")
        stream.write(f"raw_sha256:     {hashlib.sha256(raw).hexdigest()}\n")
        reference_bytes = np.asarray(reference, dtype="<f4").tobytes()
        stream.write(f"reference_sha256: {hashlib.sha256(reference_bytes).hexdigest()}\n")
        if timing_str:
            stream.write(f"timing:         {timing_str}\n")
    return {**comparison, "mcu_si_snr": mcu_sisnr, "ref_si_snr": ref_sisnr,
            "delta": mcu_sisnr - ref_sisnr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    default="COM3",
                        help="Serial port (e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud",    type=int, default=115200)
    parser.add_argument("--clean",   required=True,
                        help="Clean reference WAV paired with the embedded utterance")
    parser.add_argument("--ref_bin", required=True,
                        help="Python ONNX reference enhanced binary (.bin)")
    parser.add_argument("--out",     default="data/board_capture/mcu_enhanced.wav")
    args = parser.parse_args()
    require_new_capture(args.out)
    import serial

    N_FRAMES = load_n_frames(args.ref_bin)
    N_FLOATS = N_FRAMES * 40
    N_BYTES  = N_FLOATS * 4
    print(f"Expecting {N_FRAMES} frames ({N_FLOATS} samples, {N_BYTES} bytes)")

    # ---- Load reference SI-SNR and clean audio ----
    clean_raw, sr = sf.read(args.clean, dtype="float32")
    if sr != 16000 or clean_raw.ndim != 1 or len(clean_raw) < N_FLOATS:
        raise ValueError("Clean reference must be mono 16 kHz audio covering the full capture")
    clean = clean_raw[:N_FLOATS]

    ref_enhanced = np.fromfile(args.ref_bin, dtype="<f4")
    compare_waveforms(ref_enhanced, clean)
    ref_sisnr    = si_snr(ref_enhanced, clean)
    print(f"Python ONNX reference SI-SNR: {ref_sisnr:.2f} dB")

    # ---- Open serial port ----
    print(f"\nOpening {args.port} @ {args.baud} baud...")
    with serial.Serial(args.port, args.baud, timeout=120) as ser:
        time.sleep(0.5)
        print("Waiting for AUDIO_START marker (press RESET on board if needed)...")
        deadline = time.monotonic() + 120
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("No AUDIO_START marker received")
            line = ser.readline()
            if not line:
                raise TimeoutError("No AUDIO_START marker received")
            if b"AUDIO_START" in line:
                break
        raw = read_audio_bytes(ser, N_BYTES)
        timing_str = ""
        ended = False
        for _ in range(5):
            line = ser.readline().decode("ascii", errors="strict").strip()
            if line.startswith("TIMING:"):
                timing_str = line
            if "AUDIO_END" in line:
                ended = True
                break
        if not ended:
            raise ValueError("Audio transfer has no AUDIO_END marker")

    audio = np.frombuffer(raw, dtype="<f4")
    print(f"Received {len(audio)} samples ({len(audio)/16000:.2f} s)")

    if timing_str:
        print(f"\n{timing_str}")
    else:
        print("\n(No timing line received)")

    result = save_capture(args.out, audio, ref_enhanced, clean, Path(args.clean).stem,
                          timing_str)

    print(f"\n--- Results ---")
    print(f"MCU SI-SNR:    {result['mcu_si_snr']:.8f} dB")
    print(f"Python ref:    {ref_sisnr:.2f} dB")
    print(f"SI-SNR delta:  {result['delta']:+.8f} dB")
    print(f"Sample RMSE:   {result['rmse']:.8g}")
    print(f"Max abs error: {result['max_abs_error']:.8g}")
    for path in capture_paths(args.out):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
