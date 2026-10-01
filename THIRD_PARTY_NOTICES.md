# Third-party notices

The MIT license in `LICENSE` covers original contributions to this repository. It does not replace the terms or rights attached to third-party code, checkpoints, data or tools.

## DPSNN

The network code, data utilities and training procedure derive from [DPSNN](https://github.com/tao-sun/dpsnn) by Tao Sun and Sander Bohté. The N=256 reference checkpoint comes from that repository. The upstream repository does not currently provide a standalone license file; no additional license for that material is asserted here.

Please cite:

> T. Sun and S. Bohté, "DPSNN: Spiking neural network for low-latency streaming speech enhancement," Neuromorphic Computing and Engineering, 4(4), 044008, 2024.

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

X-CUBE-AI, STM32CubeIDE and STM32CubeProgrammer are STMicroelectronics tools with their own licenses. They are external dependencies. The analysis reports are included to document the compiled baseline models; no ST runtime library or device firmware is redistributed here.
