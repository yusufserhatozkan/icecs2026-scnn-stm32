"""Baseline streaming and ONNX checks."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from dpsnn.models.dp_binary_net import StreamSpikeNet
from evaluation.eval_streaming import run_streaming_chunk
from export import export_streaming as streaming
from export import export_to_onnx as exporter


def make_model():
    torch.manual_seed(2020)
    return StreamSpikeNet(input_dim=400, context_dim=160, sr=16000, L=80, stride=40,
                           N=8, B=8, H=8, X=1, scnn_only=True).eval()


def wrapper_for(model):
    return streaming.StreamingWrapper(model).eval()


def initial_states(model):
    return [torch.zeros(1, model.B, model.context_step), torch.zeros(1, model.B, 1),
            torch.zeros(1, model.B), torch.zeros(1, model.stride)]


def export_synthetic(wrapper, model, path):
    with path.open("xb") as output:
        torch.onnx.export(wrapper, (torch.zeros(1, model.L), *initial_states(model)), output,
                          input_names=list(streaming.STREAMING_INPUT_NAMES),
                          output_names=list(streaming.STREAMING_OUTPUT_NAMES), opset_version=13)
    onnx.checker.check_model(onnx.load(path))


def test_chunk_warmup_reset_and_final_tail_match_batch():
    model = make_model()
    wrapper = wrapper_for(model)
    audio = torch.randn(1, 400, generator=torch.Generator().manual_seed(22))
    identifier, length = torch.zeros(1), torch.tensor(400)
    batch = ((identifier, audio, length), (identifier, torch.zeros_like(audio), length))
    with torch.no_grad():
        batch_audio, *_ = model(batch)
    actual = run_streaming_chunk(wrapper, audio.squeeze(0).numpy())
    assert actual.shape == (240,)
    np.testing.assert_allclose(actual, batch_audio.squeeze(0).numpy(), atol=1e-4, rtol=0)
    # A second utterance/chunk must start fresh, despite previous calls.
    np.testing.assert_array_equal(actual, run_streaming_chunk(wrapper, audio.squeeze(0).numpy()))
    assert streaming.verify_streaming_vs_original(model) < streaming.STREAMING_ATOL


def test_raw_and_processed_onnx_match_all_outputs_across_ping_pong_hops(tmp_path):
    model = make_model()
    wrapper = wrapper_for(model)
    raw, processed = tmp_path / "raw.onnx", tmp_path / "processed.onnx"
    export_synthetic(wrapper, model, raw)
    exporter.postprocess_for_xcubeai(str(raw), str(processed), input_dim=80, output_dim=40)
    generator = torch.Generator().manual_seed(19)
    # Include exact silence, an impulse, and deterministic random frames.
    frames = [torch.zeros(1, 80), torch.nn.functional.one_hot(torch.tensor([0]), 80).float()]
    frames += [torch.randn(1, 80, generator=generator) for _ in range(10)]
    for path in (raw, processed):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        assert [v.name for v in session.get_inputs()] == list(streaming.STREAMING_INPUT_NAMES)
        assert [v.name for v in session.get_outputs()] == list(streaming.STREAMING_OUTPUT_NAMES)
        assert all(v.type == "tensor(float)" for v in session.get_inputs() + session.get_outputs())
        expected_shapes = [[1, 80], [1, 8, 4], [1, 8, 1], [1, 8], [1, 40]]
        assert [v.shape for v in session.get_inputs()] == expected_shapes
        assert [v.shape for v in session.get_outputs()] == [[1, 40]] + expected_shapes[1:]
        for nonzero in (False, True):
            torch_states = initial_states(model)
            if nonzero:
                torch_states = [torch.rand(s.shape, generator=generator) for s in torch_states]
            banks = [[s.numpy().copy() for s in torch_states],
                     [np.empty_like(s.numpy()) for s in torch_states]]
            active = 0
            with torch.no_grad():
                for frame in frames:
                    reference = wrapper(frame, *torch_states)
                    feeds = dict(zip(streaming.STREAMING_INPUT_NAMES,
                                     [frame.numpy()] + banks[active]))
                    candidate = session.run(None, feeds)
                    for expected, actual in zip(reference, candidate):
                        assert np.isfinite(actual).all()
                        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=0)
                    # External state banks copy all four outputs before the next call.
                    for destination, state in zip(banks[1 - active], candidate[1:]):
                        np.copyto(destination, state)
                    active = 1 - active
                    torch_states = list(reference[1:])
