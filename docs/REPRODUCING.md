# Reproduction

Run commands from the repository root. Write generated files to new directories under `data/`; the released models and results should remain unchanged.

## Environment

The archived host environment used Python 3.11.5, PyTorch 2.1.0, torchvision 0.16.0, torchaudio 2.1.0 and NumPy 1.26.4. The GPU training environment used CUDA 11.8. The remaining direct package versions are pinned in `requirements.txt`.

Install the matching PyTorch packages for your platform before installing this repository. A local compiler may be needed to install `pesq`. For UART tools, install `pyserial` separately.

```text
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python tools/verify_artifacts.py
python tools/verify_artifacts.py --models
```

## Dataset

Obtain the 28-speaker clean/noisy training archives and clean/noisy test archives from [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791). Extract the paired 48 kHz WAV folders under `data/raw/train/` and `data/raw/test/`.

```text
python -m tools.prepare_camera_ready_data --train-root data/raw/train --test-root data/raw/test --output-root data/voicebank
```

This prepares mono 16 kHz audio and separate train/validation/test caches, with 10,802/770/824 pairs. Validation speakers are p226 and p287; test speakers are p232 and p257. The command refuses an existing output directory and records file hashes in `data/voicebank/split_audit.json`.

The archived test HDF5 SHA-256 is `c3507125db82884509a2820af1a0296c77c007e612b7bad24327900944ce6092`. Different library versions can affect resampling or file serialization; investigate a hash mismatch before comparing results.

## Training

The baseline configuration uses seed 2020, 100 epochs, batch size 64, Adam with learning rate 0.01, bf16 mixed precision and gradient clipping at 1.0. Checkpoint selection minimizes validation loss. The test set is excluded from fitting and checkpoint selection. The archived epoch logs and selection metadata are in [results/training](../results/training/summary.json).

Training requires the SHA-256 of the split report produced by your preparation command. Print it with:

```text
python -c "from pathlib import Path; import hashlib; print(hashlib.sha256(Path('data/voicebank/split_audit.json').read_bytes()).hexdigest())"
```

Replace `YOUR_SPLIT_AUDIT_SHA256` below with that value. The hash includes your local file inventory; do not substitute the hash of the archived report or the test HDF5.

```text
python -m egs.voicebank.vctk_trainer --config conference.yaml -L 80 --stride 40 -N 128 -B 128 -H 128 -X 1 --scnn_only --device_num 1 --skip_test_after_fit --split_audit_sha256 YOUR_SPLIT_AUDIT_SHA256 --run_dir data/train_n128
python -m egs.voicebank.vctk_trainer --config conference.yaml -L 80 --stride 40 -N 64 -B 64 -H 64 -X 1 --scnn_only --device_num 1 --skip_test_after_fit --split_audit_sha256 YOUR_SPLIT_AUDIT_SHA256 --run_dir data/train_n64
```

The released N=128/N=64 checkpoints allow inference without retraining. N=256 is an archived upstream reference checkpoint, not a model trained with these two commands. Its original inference entry point is not included; see [model notes](../models/README.md).

## Baseline evaluation

```text
python evaluation/eval_streaming.py --ckpt_path models/n128.ckpt --hdf5_path data/voicebank/save/test.hdf5 --output_path data/eval_n128/metrics.txt
python evaluation/eval_streaming.py --ckpt_path models/n64.ckpt --hdf5_path data/voicebank/save/test.hdf5 --output_path data/eval_n64/metrics.txt
```

These commands use the archived convolution-decoder evaluation procedure, including enhanced-output peak normalization. Read the [method and archive limitations](RESULTS.md) before interpreting a rerun. They execute PyTorch streaming inference; they are not board measurements. The N=256 aggregate is retained as a reference and is not evaluated by this SCNN streaming entry point.

## ONNX and STM32

To export a baseline checkpoint into a new output directory:

```text
python export/export_streaming.py --ckpt_path models/n64.ckpt --output_path data/export_n64/streaming.onnx
```

For the original compiler input, use `models/dpsnn_n64_streaming_xcubeai.onnx` or the corresponding N=128 file. The saved reports identify ST Edge AI Core 2.2.0-20266, target `stm32u5`, balanced optimization and no weight compression. The paper lists X-CUBE-AI 10.2.0.

With the ST tool installed:

```text
stedgeai analyze --model models/dpsnn_n64_streaming_xcubeai.onnx --target stm32u5 --optimization balanced --compression none --name scnn_n64 --workspace data/st_workspace --output data/st_output
```

The board is the B-U585I-IOT02A with STM32U585 at 160 MHz. The streaming graph consumes 80 audio samples and emits 40 enhanced samples per call. All four output states must be retained for the next call. The paper describes externally managed ping-pong state buffers. The original firmware project is unavailable in this release, so an exact board-timing reproduction requires that project.

`tools/wav_to_c_array.py` prepares an embedded audio fixture and an ONNX reference. `tools/mcu_audio_receiver.py` receives the corresponding UART output at 115,200 baud by default. Their command-line help describes the expected files and serial format. UART transfer is excluded from the paper's DWT timing. Use fresh output directories under `data/`; these legacy utilities can overwrite files in a supplied output directory.

The board fixture preparation peak-normalizes the noisy input and streams the full clip with states carried between frames. Table I's evaluator instead uses one-second chunks and peak-normalizes the enhanced utterance. The two procedures should not be treated as the same test. The saved WAVs are 16-bit listening/plotting copies; only `test_enhanced_ref.bin` retains a raw float32 reference. The raw UART capture is unavailable.

`export/validate_onnx.py` is a full-chunk, single-input validator used by the export software tests. It does not accept the released five-input streaming graphs. `tools/verify_artifacts.py --models` checks their structure and checkpoint identity without running inference.

## Software checks

Install `pytest`, then run:

```text
python -c "from pathlib import Path; Path('data').mkdir(exist_ok=True)"
python -m pytest tests -q --basetemp data/pytest_check
```

Use a new disposable directory for `--basetemp`; pytest manages its contents. These tests check software behavior and do not generate paper results. The synthetic export tests exercise newly exported graphs, not the archived firmware or the full VoiceBank test set. Passing them does not resolve the discrepancies listed in [the paper comparison](RESULTS.md).
