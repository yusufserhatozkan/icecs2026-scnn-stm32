# Spiking Neural Network on Microcontroller for Low-cost Speech Enhancement

Code, models and results accompanying the IEEE ICECS 2026 paper by Yusuf Serhat Özkan, Tao Sun and Guangzhi Tang.

The paper deploys a time-domain spiking convolutional network on the STM32U585. A streaming ONNX graph exposes the temporal context, neuron states and overlap-add tail as recurrent inputs and outputs. Audio is processed at 16 kHz with an 80-sample window and a 40-sample hop.

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

## Repository contents

- `dpsnn/`: network layers, model and data preparation code.
- `egs/voicebank/`: training entry point and baseline configuration.
- `evaluation/`: archived baseline quality-evaluation script.
- `export/`: streaming ONNX export and X-CUBE-AI conversion.
- `models/`: N=128 and N=64 checkpoints and ONNX files, plus the upstream N=256 reference checkpoint.
- `results/`: paper tables, baseline per-utterance scores, training records and compiler reports.
- `deploy/n64/`: saved test-utterance audio, reference output and board summary.
- `figures/`: the paper's pipeline and waveform figures.
- `tests/`: software checks for data splits, checkpoint selection and export.

## Use

Use Python 3.11 and PyTorch 2.1.0. Install the matching PyTorch, torchvision and torchaudio packages for your platform, then:

```text
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python tools/verify_artifacts.py
python tools/verify_artifacts.py --models
```

See [reproduction instructions](docs/REPRODUCING.md) for dataset preparation, training, evaluation and ONNX export. The dataset and STM32 toolchain must be obtained separately. The original board firmware project is not included.

## Citation

```bibtex
@inproceedings{ozkan2026scnnstm32,
  title={Spiking Neural Network on Microcontroller for Low-cost Speech Enhancement},
  author={Özkan, Yusuf Serhat and Sun, Tao and Tang, Guangzhi},
  booktitle={2026 IEEE 33rd International Conference on Electronics, Circuits and Systems (ICECS)},
  year={2026}
}
```

This work builds on [DPSNN](https://github.com/tao-sun/dpsnn) by Tao Sun and Sander Bohté. Attribution and component terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The [MIT license](LICENSE) covers the original contributions to this repository.
