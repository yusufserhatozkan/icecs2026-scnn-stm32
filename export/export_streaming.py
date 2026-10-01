"""Export StreamSpikeNet as a single-frame streaming ONNX.

This exporter wraps the convolution-decoder model as one inference step:

    (frame[t], state[t]) -> (enhanced_samples[t], state[t+1])

The caller feeds state_out back as state_in for the next frame. The paper's
4.16-second board fixture uses 1,662 calls. Chunked host evaluation includes
four context frames followed by 399 output frames and a final overlap-add tail.

State tensors
-------------
context_win  : (1, B=128, context_step=4)  SCNN context window
v_plif       : (1, B=128, 1)               PLIFNode membrane potential
mem_readout  : (1, B=128)                  ALIFNode (no-spike) readout membrane
ola_tail     : (1, stride=40)              OLA overlap tail (L-stride prev samples)

Warmup note
-----------
For equivalence to chunked batch evaluation, retain context after the warmup
frames, reset PLIF/readout/OLA state, then collect output and flush the final
OLA tail. The caller is responsible for this chunk-boundary policy.

Usage
-----
python export/export_streaming.py \\
    --ckpt_path models/n64.ckpt \\
    --output_path data/export_n64/streaming.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpsnn.models.dp_binary_net import StreamSpikeNet
from export.export_to_onnx import (load_from_checkpoint, postprocess_for_xcubeai,
                                   require_new_outputs, xcubeai_output_path)

STREAMING_INPUT_NAMES = ("frame", "context_win", "v_plif", "mem_readout", "ola_tail")
STREAMING_OUTPUT_NAMES = ("enhanced", "new_context_win", "new_v_plif",
                          "new_mem_readout", "new_ola_tail")
STREAMING_ATOL = 1e-4


class StreamingWrapper(nn.Module):
    """Single-frame forward for a trained StreamSpikeNet (SCNN-only, X=1).

    Takes one audio frame plus recurrent state, returns enhanced audio samples
    plus updated state. All state tensors are explicit ONNX inputs/outputs so
    X-CUBE-AI can map them to fixed MCU SRAM locations.
    """

    def __init__(self, model: StreamSpikeNet) -> None:
        super().__init__()
        assert model.scnn_only, "Streaming export only supports scnn_only=True"
        assert model.X == 1, "Streaming export only supports X=1"
        assert model.L == 2 * model.stride, "Streaming OLA requires L == 2 * stride"
        assert model.H == model.B, "Streaming PLIF state currently requires H == B"
        assert not model.no_pointwise, "This exporter requires the conference baseline architecture"

        # Share sub-modules with the trained model (weights are not copied)
        self.encoder_1d = model.encoder_1d
        self.encoder_act = model.encoder_act
        self.ln = model.ln
        self.proj = model.proj
        self.sconv1d = model.repeats[0][0]  # block 0, module 0
        self.srnn_readout = model.srnn_readout
        self.mask = model.mask
        self.mask_act = model.mask_act
        self.decoder_1d = model.decoder_1d

        self.L = model.L
        self.stride = model.stride
        self.context_step = model.context_step

        # Cache PLIFNode hyper-parameters as scalar tensors so ONNX bakes them
        # as constants (avoids runtime attribute lookup, which doesn't export).
        plif = self.sconv1d.neuron
        self.plif_v_threshold = float(plif.v_threshold)
        self.plif_v_reset = float(plif.v_reset)  # 0.0
        self.plif_surrogate = plif.surrogate_function

        alif = self.srnn_readout.neuro
        self.alif_R_m = float(alif.R_m)  # 1.0

    def forward(
        self,
        frame: torch.Tensor,        # (1, L)
        context_win: torch.Tensor,  # (1, B, context_step)
        v_plif: torch.Tensor,       # (1, B, 1)
        mem_readout: torch.Tensor,  # (1, B)
        ola_tail: torch.Tensor,     # (1, stride)
    ):
        # ---- Encoder --------------------------------------------------------
        x = self.encoder_1d(frame)          # (1, N, 1)  [Conv1D adds channel dim]
        x = self.encoder_act(x)             # (1, N, 1)
        w = x                               # encoder skip for masking

        x = self.ln(x)                      # (1, N, 1)  norm
        x = self.proj(x)                    # (1, B, 1)  binary proj

        # ---- SCNN (SpikeConv1d with explicit PLIFNode state) ----------------
        win_w = torch.cat([context_win, x], dim=2)  # (1, B, context_step+1)
        y = self.sconv1d.dconv(win_w)               # (1, H, 1)

        # PLIFNode step — expressed without self.v so it exports as a pure
        # data-flow graph with no mutable attributes.
        # leaky integration: v_new = v + (y - (v - v_reset)) * alpha
        alpha_plif = self.sconv1d.neuron.w.sigmoid()
        v_new = v_plif + (y - (v_plif - self.plif_v_reset)) * alpha_plif
        # spike via heaviside (eval mode) — GreaterOrEqual+Cast, eliminated by
        # pipeline step 9 (replaced with Sub/Sign/ReLU for X-CUBE-AI BOOL issue)
        spike = self.plif_surrogate(v_new - self.plif_v_threshold)
        # hard voltage reset: v_reset=0, so fired neurons → 0
        # Using multiply (instead of masked_fill) avoids BOOL tensors entirely.
        new_v_plif = v_new * (1.0 - spike.detach())

        x = spike                                   # (1, H, 1)
        new_context_win = win_w[:, :, 1:]           # (1, B, context_step)

        # ---- ReadoutLayer (ALIFNode no_spiking, explicit membrane state) ----
        x_sq = x.squeeze(2)                         # (1, H)
        y_dense = self.srnn_readout.dense(x_sq)     # (1, B)
        alpha_readout = self.srnn_readout.neuro.alpha.sigmoid()
        new_mem_readout = (mem_readout * alpha_readout
                           + (1.0 - alpha_readout) * self.alif_R_m * y_dense)  # (1, B)
        x_readout = new_mem_readout.unsqueeze(2)    # (1, B, 1)

        # ---- Mask -----------------------------------------------------------
        x_mask = self.mask(x_readout)               # (1, N, 1)
        x_mask = self.mask_act(x_mask)              # sigmoid
        x_out = w * x_mask                          # (1, N, 1)  encoder skip × mask

        # ---- Decoder + OLA --------------------------------------------------
        decoded = self.decoder_1d(x_out).squeeze(1) # (1, L)

        # OLA: finalize stride samples that are now fully covered by two frames.
        # ola_tail = last (L-stride) samples of previous decoded frame.
        # enhanced = ola_tail + decoded[:stride]   (both contributions to positions
        #            [prev_t*stride, prev_t*stride+stride))
        # new_tail  = decoded[stride:]             (will overlap with next frame)
        enhanced = ola_tail + decoded[:, :self.stride]   # (1, stride)
        new_ola_tail = decoded[:, self.stride:]           # (1, L-stride)

        return enhanced, new_context_win, new_v_plif, new_mem_readout, new_ola_tail


def verify_streaming_vs_original(model: StreamSpikeNet) -> float:
    """Run the streaming wrapper for one full utterance and compare vs original.

    Use the evaluator's warmup-reset and final-tail policy; compare every sample.
    """
    model.eval()
    wrapper = StreamingWrapper(model)
    wrapper.eval()

    L, stride, context_step = model.L, model.stride, model.context_step
    B = model.B

    # Use the training input_dim so model.feature_steps (fixed at 403) is valid.
    input_dim = model.hparams["input_dim"]
    x_in = torch.randn(1, input_dim, generator=torch.Generator().manual_seed(0))
    dummy_id = torch.zeros(1)
    dummy_len = torch.tensor(input_dim)
    batch = ((dummy_id, x_in, dummy_len), (dummy_id, torch.zeros_like(x_in), dummy_len))

    with torch.no_grad():
        out_orig, *_ = model(batch)

    # Streaming: run wrapper frame-by-frame
    context_win = torch.zeros(1, B, context_step)
    v_plif = torch.zeros(1, B, 1)
    mem_readout = torch.zeros(1, B)
    ola_tail = torch.zeros(1, stride)

    streaming_chunks = []
    skip = model.X * context_step

    with torch.no_grad():
        for t in range(model.feature_steps):
            frame = x_in[:, t * stride: t * stride + L]
            enhanced, context_win, v_plif, mem_readout, ola_tail = wrapper(
                frame, context_win, v_plif, mem_readout, ola_tail)
            if t == skip - 1:
                v_plif = torch.zeros_like(v_plif)
                mem_readout = torch.zeros_like(mem_readout)
                ola_tail = torch.zeros_like(ola_tail)
            if t >= skip:
                streaming_chunks.append(enhanced)
        streaming_chunks.append(ola_tail)

    out_stream = torch.cat(streaming_chunks, dim=1)

    if out_orig.shape != out_stream.shape:
        raise RuntimeError(f"Streaming/batch output shapes differ: {out_stream.shape}, {out_orig.shape}")
    if not torch.isfinite(out_orig).all() or not torch.isfinite(out_stream).all():
        raise RuntimeError("Nonfinite streaming/batch verification output")
    diff = (out_orig - out_stream).abs()
    return float(diff.max())


def export_streaming(
    ckpt_path: str,
    output_raw: str,
    output_xcubeai: str,
) -> None:
    require_new_outputs(output_raw, output_xcubeai)
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_from_checkpoint(ckpt_path)
    model.eval()

    L = model.L
    stride = model.stride
    B = model.B
    context_step = model.context_step
    print(f"Model: N={model.N}, B={B}, L={L}, stride={stride}, "
          f"context_step={context_step}, time_steps={model.time_steps}")

    print("Verifying streaming wrapper against original model ...")
    max_diff = verify_streaming_vs_original(model)
    print(f"  max-abs-diff (complete chunk) = {max_diff:.2e}")
    if not max_diff < STREAMING_ATOL:
        raise RuntimeError(f"Streaming/batch difference {max_diff} exceeds {STREAMING_ATOL}")

    wrapper = StreamingWrapper(model)
    wrapper.eval()

    # Dummy state tensors (zeros = initial state)
    dummy_frame = torch.zeros(1, L)
    dummy_ctx = torch.zeros(1, B, context_step)
    dummy_v = torch.zeros(1, B, 1)
    dummy_mem = torch.zeros(1, B)
    dummy_tail = torch.zeros(1, stride)

    print(f"\nExporting streaming ONNX -> {output_raw}")
    os.makedirs(os.path.dirname(output_raw) or ".", exist_ok=True)
    with open(output_raw, "xb") as output:
        torch.onnx.export(
            wrapper, (dummy_frame, dummy_ctx, dummy_v, dummy_mem, dummy_tail), output,
            input_names=list(STREAMING_INPUT_NAMES), output_names=list(STREAMING_OUTPUT_NAMES),
            opset_version=13,
        )
    size_kb = os.path.getsize(output_raw) / 1024
    print(f"Raw streaming ONNX: {size_kb:.1f} KB")

    print(f"\nRunning X-CUBE-AI compatibility pipeline -> {output_xcubeai}")
    postprocess_for_xcubeai(
        onnx_path=output_raw,
        output_path=output_xcubeai,
        input_dim=L,
        output_dim=stride,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export StreamSpikeNet as streaming ONNX")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", required=True,
                        help="Path for the raw streaming ONNX")
    parser.add_argument("--xcubeai_out", default=None,
                        help="Path for the X-CUBE-AI output (default: <stem>_xcubeai.onnx)")
    args = parser.parse_args()

    xcubeai_path = args.xcubeai_out or xcubeai_output_path(args.output_path)

    try:
        export_streaming(args.ckpt_path, args.output_path, xcubeai_path)
    except Exception as exc:
        print(f"\nExport FAILED: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
