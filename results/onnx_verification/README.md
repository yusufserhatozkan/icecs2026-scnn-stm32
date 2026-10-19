# ONNX Runtime reproduction

On 19 October 2026, both frozen conference baseline graphs were evaluated with ONNX Runtime on the same 824-utterance VoiceBank-DEMAND cache. These are reproduction checks of the published models, with no retraining or architecture changes.

| Model | SI-SNR (dB) | PESQ | STOI | Complete utterances |
|---|---:|---:|---:|---:|
| N=128 | 17.424370637365076 | 2.1490580890942548 | 0.9232329764126691 | 824 |
| N=64 | 16.696732334549463 | 2.021783474145584 | 0.9215905312350491 | 824 |

Every displayed baseline value in Table I is reproduced after rounding to the paper's precision. All required metrics are finite for every utterance. The earlier PyTorch records remain in `results/metrics/` and `results/per_utterance/`.

The checks execute the released `*_streaming_xcubeai.onnx` graphs with the CPU execution provider. They retain one-second chunks, context warmup, recurrent-state resets, the final overlap-add flush, utterance-length cropping and enhanced-output peak normalization. Scoring retains the composite helper's in-place operations. This comparison changes the inference backend while preserving the conference scoring procedure.

The [N=128](n128.json) and [N=64](n64.json) records contain every per-utterance score, package versions, model and dataset hashes, the evaluator hash and the record-writing time. The dataset path uses the documented repository layout. `source_record_sha256` identifies the original local result file before that path was made portable and release metadata was added.

The test-cache SHA-256 is `c3507125db82884509a2820af1a0296c77c007e612b7bad24327900944ce6092`. These host checks do not measure board latency. Commands are in the [reproduction instructions](../../docs/REPRODUCING.md).
