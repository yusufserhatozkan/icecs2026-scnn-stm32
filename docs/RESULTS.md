# Results and artifact notes

## Scope

This release accompanies the submitted four-page ICECS 2026 paper, *Spiking Neural Network on Microcontroller for Low-cost Speech Enhancement*. Its SHA-256 is `4c41cb819df0c48c7ae747261e59546332a8322a632d1b865f25d46389f4c929`.

`results/table1.csv`, `results/table2.csv` and `results/deployment.csv` transcribe the paper's values and labels. Packaging these records did not involve training, dataset evaluation or board measurements.

## Speech-quality records

The N=128 and N=64 records contain 824 utterances each. SI-SNR, wideband PESQ and STOI each have 824 finite scores. The published CSV files select these metrics from the saved evaluation records; their values have not been recomputed or changed. Source-file hashes are included in the corresponding JSON summaries.

The stored protocol uses one-second chunks, a context prefix, state resets after warmup, a final overlap-add flush and cropping to the utterance length. Enhanced utterances are peak-normalized before scoring. Clean and noisy inputs are not peak-normalized by the evaluator.

The paper labels Table I as ONNX streaming. The archived September evaluation records identify the executing implementation as the PyTorch streaming wrapper. ONNX export and numerical validation were separate steps. This release preserves that distinction.

The archived evaluator also calls composite metrics. Those metrics are not reported here. Its composite helper changes the reference array in place before enhanced-side scoring. That behavior is retained to preserve the archived procedure; the supplied scores should be read with that limitation in mind.

For N=256, an aggregate result and the upstream checkpoint are available, but per-utterance records are not. The paper calls this row a reproduced SCNN reference. The upstream checkpoint contains a recurrent separator stage as well as the convolutional stage; it is preserved as supplied, rather than converted into a different architecture. Its legacy `readout_threshold` tensor is not accepted by the release loader. The checkpoint is included as an archive; reproducing that aggregate requires its original inference implementation.

## Model files

The checkpoint files contain the original tensors and model hyperparameters. Optimizer state, callbacks and machine-specific training paths have been omitted. `models/manifest.json` records source checkpoint hashes and hashes of the released files. The N=128/N=64 weight tensors were checked to be identical to those in the earlier deployment checkpoints.

The processed ONNX files are byte-identical to the files identified by the MD5 model hashes in the supplied ST analysis reports. They retain the convolution decoder and both pointwise layers. Their five inputs/outputs carry one audio tensor and four recurrent states.

Table II reports 43 graph nodes for each width. The archived processed ONNX files contain 44 ONNX nodes. Compiler graph counts and ONNX file counts should not be silently substituted for each other. Both the paper's reported count and the original graph files are retained.

The export code includes a threshold-boundary correction. A newly generated graph may therefore differ from the archived compiler input. Use the supplied processed files when referring to the saved ST reports.

## Board records

The N=64 board summary records 1,662 frames, 2.715 ms/frame and 15.9771 dB SI-SNR, which rounds to the paper's 15.98 dB. Its clean/noisy fixture, enhanced WAV and reference float32 output are included under `deploy/n64/`.

The N=128 values of 6.145 ms/frame and 15.66 dB are preserved as paper-reported values. The original baseline firmware project, ELF/BIN images, N=128 baseline waveform capture and raw aiValidation report were not located in the available archive. The paper reports a largest per-output RMSE of 3.48e-6 across 10 random-input batches; that statement cannot be independently reconstructed from a raw validation report in this release.

The original firmware source is needed to reproduce the reported board timings exactly. The ONNX models, compiler reports and host utilities support rebuilding a deployment, but do not constitute the original firmware. The reported timings do not include live microphone acquisition or UART transfer.

## Figures

`figures/pipeline.png` and `figures/waveforms.png` are the supplied paper figures. Figure 2 displays peak-normalized waveforms for utterance `p232_009`. Plot scaling and numerical evaluation are separate steps, even though the archived evaluation also used enhanced-output peak normalization.
