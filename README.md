# Spiking Neural Network on Microcontroller for Low-cost Speech Enhancement

Code, models and results accompanying the IEEE ICECS 2026 paper by Yusuf Serhat Özkan, Tao Sun and Guangzhi Tang.

The paper deploys a time-domain spiking convolutional network on the STM32U585. A streaming ONNX graph exposes the temporal context, neuron states and overlap-add tail as recurrent inputs and outputs. Audio is processed at 16 kHz with an 80-sample window and a 40-sample hop.

## Try the conference example

Listen to the [noisy input](deploy/n64/test_noisy.wav), [STM32 output](deploy/n64/mcu_enhanced.wav), and [clean reference](deploy/n64/test_clean.wav) for utterance p232_009.

From the repository root, run the file and results check with Python 3.11. For the CPU demo, install the three dependencies in a virtual environment and run the saved N=64 ONNX graph on the included audio.

```text
python tools/verify_artifacts.py
python -m pip install numpy==1.26.4 onnxruntime==1.29.0 soundfile==0.14.0
python tools/wav_to_c_array.py --noisy deploy/n64/test_noisy.wav --clean deploy/n64/test_clean.wav --onnx models/dpsnn_n64_streaming_xcubeai.onnx --out_c data/demo_n64/c --out_wav data/demo_n64/audio
```

The demo runs 1,662 frames and writes a listening WAV and a float32 reference under `data/demo_n64/audio`. It runs on the host CPU. For training, evaluation on the full test set, and STM32 compilation, see [Reproduction](docs/REPRODUCING.md).

## Results reported in the paper

| Model | SI-SNR (dB) | PESQ | STOI |
|---|---:|---:|---:|
| Noisy input | 8.44 | 1.971 | 0.921 |
| Reproduced SCNN N=256 | 18.08 | 2.264 | 0.925 |
| SCNN N=128, ONNX streaming | 17.42 | 2.149 | 0.923 |
| SCNN N=64, ONNX streaming | 16.70 | 2.022 | 0.922 |

| Deployment measurement | N=128 | N=64 |
|---|---:|---:|
| Weights (Flash) | 280 KB | 91 KB |
| Activations (SRAM) | 46 KB | 23 KB |
| Reported ONNX graph nodes | 43 | 43 |
| Analyze wall-time | 16 s | 16 s |
| Per-frame latency | 6.145 ms | 2.715 ms |
| Real-time factor | 2.46 | 1.086 |

Latency was reported for the B-U585I-IOT02A board at 160 MHz, excluding UART transmission. The frame budget is 2.5 ms. Both baselines exceed that budget.

The submitted conference paper is the reference for the reported results and model labels in this repository. The tables above preserve its values. [Reported results and archive notes](docs/RESULTS.md) describe the supporting files, their compatibility and the available evidence for reproduction. Archive observations do not replace the paper's reported results.

An [ONNX Runtime reproduction check](results/onnx_verification/README.md) evaluates both frozen baseline graphs on all 824 test utterances and reproduces their displayed Table I values.

## Repository contents

This repository accompanies the conference paper with model checkpoints, ONNX exports, evaluation records, compiler reports and saved deployment summaries. Board measurements are preserved as reported in the paper. The original firmware project and raw validation captures are not included in this release.

- `dpsnn/`: network layers, model and data preparation code.
- `egs/voicebank/`: training entry point and baseline configuration.
- `evaluation/`: baseline quality evaluation with PyTorch and ONNX Runtime.
- `export/`: streaming ONNX export and X-CUBE-AI conversion.
- `models/`: N=128 and N=64 checkpoints and ONNX files, plus the upstream N=256 reference checkpoint.
- `results/`: paper tables, baseline per-utterance scores, ONNX reproduction checks, training records and compiler reports.
- `deploy/n64/`: saved test-utterance audio, reference output and board summary.
- `figures/`: the paper's pipeline and waveform figures.
- `licenses/`: license texts for the upstream components identified in the third-party notices.
- `tests/`: software checks for data splits, checkpoint selection and export.

## Citation

```bibtex
@inproceedings{ozkan2026scnnstm32,
  title={Spiking Neural Network on Microcontroller for Low-cost Speech Enhancement},
  author={Özkan, Yusuf Serhat and Sun, Tao and Tang, Guangzhi},
  booktitle={2026 IEEE 33rd International Conference on Electronics, Circuits and Systems (ICECS)},
  year={2026}
}
```

## License

The [MIT license](LICENSE) covers original contributions. The inherited code and N=256 reference checkpoint come from [DPSNN](https://github.com/tao-sun/dpsnn) by Tao Sun and Sander Bohté. That revision has no published license for those materials. See [third-party notices](THIRD_PARTY_NOTICES.md) for the rights and terms that apply to individual components.
