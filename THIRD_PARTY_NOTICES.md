# Third-party notices

The MIT license in `LICENSE` covers original contributions to this repository. It does not replace the terms or rights attached to third-party code, checkpoints, data or tools.

## DPSNN

The network code, data utilities and training procedure derive from [DPSNN](https://github.com/tao-sun/dpsnn/tree/1c162f76732e36d0d4e23caf743be86a7b1be681) by Tao Sun and Sander Bohté. The N=256 reference checkpoint comes from that repository. The referenced revision does not provide a standalone license file; no additional license for that material is asserted here. The notices below identify upstream components inherited through DPSNN. They do not grant rights to other DPSNN code or its modifications.

Please cite:

> T. Sun and S. Bohté, "DPSNN: Spiking neural network for low-latency streaming speech enhancement," Neuromorphic Computing and Engineering, 4(4), 044008, 2024.

## Asteroid

`dpsnn/layers/sdr.py` contains Asteroid's SDR loss implementations. Its source lines match [Asteroid's `asteroid/losses/sdr.py`](https://github.com/asteroid-team/asteroid/blob/c15708a04d3d28e9a1cd50553c456299dbc6d236/asteroid/losses/sdr.py), apart from the attribution header added here. The [MIT notice](licenses/asteroid-MIT.txt), including Copyright (c) 2019 Pariente Manuel, is included with this release.

## SpikingJelly

The following files contain code from SpikingJelly, inherited through DPSNN. The source links refer to revision `50d19752c39d7b80a7103076c58ba143a63a9527`, whose [license](https://github.com/fangwei123456/spikingjelly/blob/50d19752c39d7b80a7103076c58ba143a63a9527/LICENSE) is Mozilla Public License 2.0. A copy is included in [licenses/spikingjelly-MPL-2.0.txt](licenses/spikingjelly-MPL-2.0.txt).

| Local file | Upstream source | Retained code |
|---|---|---|
| `dpsnn/layers/accelerating.py` | [`clock_driven/accelerating.py`](https://github.com/fangwei123456/spikingjelly/blob/50d19752c39d7b80a7103076c58ba143a63a9527/spikingjelly/clock_driven/accelerating.py) | Source lines match, apart from the attribution header added here. |
| `dpsnn/layers/surrogate.py` | [`clock_driven/surrogate.py`](https://github.com/fangwei123456/spikingjelly/blob/50d19752c39d7b80a7103076c58ba143a63a9527/spikingjelly/clock_driven/surrogate.py) | Surrogate functions and documentation, with changes inherited through DPSNN. |
| `dpsnn/layers/spike_neurons.py` | [`clock_driven/neuron.py`](https://github.com/fangwei123456/spikingjelly/blob/50d19752c39d7b80a7103076c58ba143a63a9527/spikingjelly/clock_driven/neuron.py) | Portions of `BaseNode` and its documentation, with changes inherited through DPSNN. Other neuron implementations also remain in this file. |

These references identify matching historical source. They do not establish the revision originally imported by DPSNN or apply the license of a later SpikingJelly release to this code.

## SpeechBrain

`dpsnn/data/voicebank_prepare.py` adapts [SpeechBrain's VoiceBank preparation recipe](https://github.com/speechbrain/speechbrain/blob/89ead74d163463d30c62329a09cfdb4c54f5abc1/recipes/Voicebank/voicebank_prepare.py). It retains the author credits for Szu-Wei Fu and Peter Plantinga. The `get_all_files` utility in `dpsnn/data/data_uitls.py` derives from [SpeechBrain's data utilities](https://github.com/speechbrain/speechbrain/blob/89ead74d163463d30c62329a09cfdb4c54f5abc1/speechbrain/utils/data_utils.py).

These components were adapted through DPSNN for its data preparation. They are not unmodified copies of the complete upstream files. SpeechBrain's [Apache License 2.0](licenses/speechbrain-Apache-2.0.txt) is included. The bracketed fields in that license's standard appendix are part of the upstream license text, not unfinished project documentation.

## Denoiser augmentation

`dpsnn/data/augment.py` retains the Facebook, Inc. copyright notice and Alexandre Défossez's `adefossez` author credit. Its batch-remixing functions were adapted through DPSNN from [Denoiser's augmentation code](https://github.com/facebookresearch/denoiser/blob/8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/augment.py). The local functions accept clean/noisy tensors instead of Denoiser's stacked sources and include a separate `remix_split` function.

Denoiser's [Creative Commons Attribution-NonCommercial 4.0 license](licenses/denoiser-CC-BY-NC-4.0.txt) accompanies the inherited augmentation code. This is separate from the MIT exception for its speech-quality metrics below. The root MIT license does not replace these component terms. Augmentation is disabled by the conference training protocol.

## Speech-quality metrics

`dpsnn/data/metrics.py` derives from the `matlab_eval.py` evaluation code described in [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser#license), adapted from [santi-pdp/segan_pytorch](https://github.com/santi-pdp/segan_pytorch). Denoiser identifies that component as MIT-licensed; its general repository license is different. The local version adapts the metric interface. The component's MIT notice is retained below. The PESQ and STOI packages retain their own terms.

```text
MIT License

Copyright (c) 2018 Santi DSP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## VoiceBank-DEMAND

The saved audio fixture in `deploy/n64/` is derived from the VoiceBank-DEMAND test utterance `p232_009`, distributed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The fixture is resampled to 16 kHz; enhanced versions are model outputs. The full dataset is not distributed here. Dataset source and end-user license: [University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791).

> C. Valentini-Botinhao, "Noisy speech database for training speech enhancement algorithms and TTS models, 2016," University of Edinburgh, 2017. DOI: 10.7488/ds/2117.

## STM32 tools

X-CUBE-AI, STM32CubeIDE and STM32CubeProgrammer are STMicroelectronics tools with their own licenses. They are external dependencies. The analysis reports are included to document the compiled baseline models; no ST runtime library is redistributed here.
