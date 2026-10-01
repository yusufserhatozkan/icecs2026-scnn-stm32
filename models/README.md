# Models

`n128.ckpt` and `n64.ckpt` contain the selected conference baseline weights. Each retains learned-threshold binarization, the pointwise bottleneck and mask layers, and the convolution decoder. The released checkpoint payloads contain only model tensors and hyperparameters.

For each baseline, `dpsnn_n*_streaming.onnx` is the raw streaming export and `dpsnn_n*_streaming_xcubeai.onnx` is the archived processed compiler input. Each graph takes one 80-sample frame and four recurrent state tensors, then returns 40 enhanced samples and four updated states.

`n256_reference.ckpt` is the upstream reference checkpoint used for the paper's N=256 row. It includes a recurrent separator stage and a legacy `readout_threshold` tensor that the release loader does not accept. It is retained as an archive, with its tensors unchanged; its original inference entry point is not included. Use the N=128/N=64 checkpoints with the streaming exporter.

`manifest.json` records source and release hashes, tensor preservation and compiler-input identity. See [artifact notes](../docs/RESULTS.md) for the distinction between the saved graphs, evaluation records and paper labels.
