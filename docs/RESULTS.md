# Paper and repository comparison

## Scope

This release accompanies the submitted four-page ICECS 2026 paper, *Spiking Neural Network on Microcontroller for Low-cost Speech Enhancement*. Its SHA-256 is `4c41cb819df0c48c7ae747261e59546332a8322a632d1b865f25d46389f4c929`.

`results/table1.csv`, `results/table2.csv` and `results/deployment.csv` transcribe the paper's numerical values and model identities, with shortened labels. Packaging these records did not involve research training, dataset evaluation or board measurements.

Passing the repository checks verifies file consistency, not every scientific claim. Some entries match retained measurements; others are transcriptions with missing primary evidence. Research experiments were not rerun for this comparison; checks inspect saved artifacts and use synthetic software tests.

| Paper location | Repository evidence | Finding |
|---|---|---|
| Table I: four quality rows | [Table I](../results/table1.csv), [metric summaries](../results/metrics/n128.json), [per-utterance records](../results/per_utterance/n128.csv) | All displayed numbers match. Both baseline means round to their table entries. |
| III-B: ONNX Runtime evaluation | [Evaluator](../evaluation/eval_streaming.py), baseline JSON protocol fields | Retained 824-utterance records identify PyTorch streaming, not ONNX Runtime. Rounded agreement does not establish the claimed backend. |
| III-A: dataset partitions and training | [Split/selection summary](../results/training/summary.json), [N=128 log](../results/training/n128_progress.jsonl), [N=64 log](../results/training/n64_progress.jsonl) | Counts and speaker separation match. Each baseline log contains epochs 0-99; selected minima match the checkpoint metadata. |
| II-A/B: architecture and recurrent tensors | [Model](../dpsnn/models/dp_binary_net.py), [wrapper](../export/export_streaming.py), [manifest](../models/manifest.json) | Baseline layers and all five input/output shapes match the description. |
| Fig. 1 and Fig. 2 | [Pipeline](../figures/pipeline.png), [waveforms](../figures/waveforms.png) | Both images match the PDF's embedded images pixel for pixel. Image identity alone does not establish the provenance of every plotted sample. |
| Table II | [N=128 report](../results/compiler/n128.txt), [N=64 report](../results/compiler/n64.txt) | Exact storage is available. Reports count 43 C nodes; ONNX files have 44. Analysis elapsed times are absent. |
| III-C.2: aiValidation | Paper only | Raw 10-batch validation report unavailable. |
| III-C.3/4: N=64 clip and timing | [Board summary](../deploy/n64/mcu_test_results.txt), [timing line](../results/n64_timing.txt), saved audio | Summary matches 1,662 frames, 15.98 dB and 2.715 ms/frame. Original firmware and raw UART samples are unavailable. |
| III-C.3/4: N=128 clip and timing | [Paper transcription](../results/deployment.csv) | 15.66 dB and 6.145 ms/frame are paper-reported; baseline waveform and original firmware are unavailable. |

## Speech-quality records

The N=128 and N=64 records contain 824 utterances each. SI-SNR, wideband PESQ and STOI each have 824 finite scores. The published CSV files select these metrics from the saved evaluation records; their values have not been recomputed or changed. Source-file hashes are included in the corresponding JSON summaries.

The stored protocol uses one-second chunks, a context prefix, state resets after warmup, a final overlap-add flush and cropping to the utterance length. Enhanced utterances are peak-normalized before scoring. This occurs in numerical evaluation, not just in Fig. 2. The evaluator does not explicitly peak-normalize clean/noisy inputs, but the composite helper changes arrays as described below.

The paper labels Table I as ONNX streaming. The archived September evaluation records identify the executing implementation as the PyTorch streaming wrapper. ONNX export and numerical validation were separate steps. This release preserves that distinction.

The evaluator scores the noisy side first, including optional composite metrics. The composite helper changes the clean reference array in place before enhanced-side SI-SNR, PESQ and STOI calls. The archived protocol string saying "clean/noisy unchanged" describes the explicit normalization step; it does not account for this mutation. The released evaluator retains the saved procedure. These records must not be described as an independent, mutation-free evaluation of raw outputs.

For N=256, an aggregate result and the upstream checkpoint are available, but per-utterance records are not. The paper calls this row a reproduced SCNN reference. The upstream checkpoint contains a recurrent separator stage as well as the convolutional stage; it is preserved as supplied, rather than converted into a different architecture. Its legacy `readout_threshold` tensor is not accepted by the release loader. The checkpoint is included as an archive; reproducing that aggregate requires its original inference implementation.

The quality differences in III-B are consistent with subtraction of the printed, rounded Table I entries. Differences calculated from unrounded means can round differently in the final digit.

## Training records

The two [baseline epoch logs](../results/training/summary.json) contain 100 completed epochs each. Minimum validation loss selects stored epochs 87 and 92 (human epochs 88 and 93) for N=128 and N=64. Their source checkpoint hashes match the model manifest. These logs do not establish a 100-epoch training history for the upstream N=256 reference.

The archived split metadata records 10,802 training, 770 validation and 824 test pairs, with disjoint speaker sets. Validation speakers are p226/p287; test speakers are p232/p257. The dataset and original HDF5 caches are not distributed here.

Training uses `100 + negative SI-SDR + 0.001 * MSE`. The paper omits the constant 100; that constant shifts the recorded loss but changes neither its gradients nor checkpoint ordering. The N=128/N=64 models have 71,299/23,363 trainable parameters. The N=256 checkpoint stores 372,996 scalar tensor elements, including its legacy scalar, consistent with the approximate 5.2-fold comparison despite the architecture-label discrepancy.

## Model files

The checkpoint files contain the original tensors and model hyperparameters. Optimizer state, callbacks and machine-specific training paths have been omitted. `models/manifest.json` records source checkpoint hashes and hashes of the released files. The N=128/N=64 weight tensors were checked to be identical to those in the earlier deployment checkpoints.

The processed ONNX file MD5 hashes match the model hashes in the supplied ST analysis reports. They retain the convolution decoder and both pointwise layers. Their five inputs/outputs carry one audio tensor and four recurrent states. The readout uses the ALIF class in its non-spiking mode, with membrane state but no adaptive-threshold state. Its existing dense layer is a different component from the convolution decoder.

Table II labels 43 nodes per width as ONNX graph nodes. The archived raw ONNX files contain 73 nodes and the processed files contain 44. The ST reports explicitly record 43 generated C nodes. Thus the table number matches the C-graph count, while its label names the ONNX graph. Both the reported table and the original files are retained.

The current export code preserves equality in `GreaterOrEqual` comparisons. The archived processed files used a strict comparison at that boundary. A new export therefore need not reproduce the archived graph exactly. Use the supplied processed files when referring to the saved ST reports. Export software tests cover current exports; they do not retrospectively validate the archived board executable.

## Table II and latency arithmetic

| Quantity | N=128 | N=64 | Interpretation |
|---|---:|---:|---|
| Paper weight storage | 280 KB | 91 KB | Preserved as reported. |
| Compiled weights in ST report | 285,216 B | 93,472 B | 278.53/91.28 KiB; the paper uses coarse values without a consistent stated rounding rule for N=128. |
| Compiled activations in ST report | 47,072 B | 23,264 B | 45.97/22.72 KiB, consistent with 46/23 when rounded in KiB. |
| Paper analysis wall-time | 16 s | 16 s | No elapsed-time evidence in the retained compiler reports. |

The statement that halving channel width roughly halves both weights and activations is not supported for weights: the reports show a 3.05-fold weight reduction and a 2.02-fold activation reduction. Both footprints fit the capacities stated in the paper, but compiler activation storage is not a measurement of complete firmware SRAM usage.

The latency arithmetic checks out: 40 samples at 16 kHz gives a 2.5 ms hop; 6.145/2.5 rounds to 2.46, 2.715/2.5 is 1.086, and 6.145/2.715 rounds to a 2.26-fold speed-up. Both reported baselines exceed the hop budget. Clock configuration, DWT timing, UART exclusion and pointer-swapped state buffers remain paper descriptions without the original firmware to inspect.

## Board records

The N=64 board summary records 1,662 frames, 2.715 ms/frame and 15.9771 dB SI-SNR, which rounds to the paper's 15.98 dB. Its clean/noisy fixture, enhanced WAV and reference float32 output are included under `deploy/n64/`.

The clean/noisy WAVs contain 66,522 samples at 16 kHz (4.157625 s, rounded to 4.16 s). The 80-sample window and 40-sample hop give 1,662 complete frames. Enhanced WAVs and the reference binary contain 66,480 samples. The board helper does not append a final overlap-add tail.

The board helper peak-normalizes the noisy input and carries state across the full clip, unlike Table I's chunked procedure. The receiver scores raw float32 samples before saving peak-normalized, 16-bit PCM WAVs. Those WAVs are listening/plotting copies, not raw captures.

Two retained reference scores differ: `reference_sisnr.txt` records 15.9755 dB, while `mcu_test_results.txt` records 15.9771 dB for both reference and MCU. The reference binary scored against the saved, quantized clean WAV agrees with 15.9771 dB. Both values round to 15.98, but these files should not be compared as scores under identical reference serialization. Without raw MCU samples, the paper's raw-output agreement within 5e-4 dB cannot be independently reconstructed from saved WAVs alone.

The N=128 values of 6.145 ms/frame and 15.66 dB are preserved as paper-reported values. The original baseline firmware project, ELF/BIN images, N=128 baseline waveform capture and raw aiValidation report were not located in the available archive. The paper reports a largest per-output RMSE of 3.48e-6 across 10 random-input batches; that statement cannot be independently reconstructed from a raw validation report in this release.

The original firmware source is needed to reproduce the reported board timings exactly. The ONNX models, compiler reports and host utilities support rebuilding a deployment, but do not constitute the original firmware. The reported timings do not include live microphone acquisition or UART transfer.

## Figures

`figures/pipeline.png` and `figures/waveforms.png` are the supplied paper figures. Figure 2 displays peak-normalized waveforms for utterance `p232_009`. Plot scaling and numerical evaluation are separate steps, even though the archived evaluation also used enhanced-output peak normalization.
