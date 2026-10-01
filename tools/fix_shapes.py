"""
Fix symbolic/dynamic dimensions in an ONNX model for X-CUBE-AI.

X-CUBE-AI requires fully static shapes. After ONNX export + QDQ insertion,
some intermediate tensors retain symbolic dimension names (e.g. 'H', 'batch_size').
This script sets the input to a concrete shape and runs ONNX shape inference
to propagate concrete integer dimensions throughout the graph.

Usage:
    python tools/fix_shapes.py --input C:/ai/dpsnn_int8_final.onnx \
                                --output C:/ai/dpsnn_int8_static.onnx \
                                --batch 1 --samples 16160
"""

import argparse
import onnx
from onnx import shape_inference, helper, TensorProto


def fix_input_shape(model: onnx.ModelProto, batch: int, samples: int) -> onnx.ModelProto:
    graph = model.graph

    # Fix the main audio input (first graph input that is not an initializer)
    init_names = {init.name for init in graph.initializer}
    for inp in graph.input:
        if inp.name in init_names:
            continue  # skip weight/bias inputs
        tt = inp.type.tensor_type
        del tt.shape.dim[:]
        d0 = tt.shape.dim.add()
        d0.dim_value = batch
        d1 = tt.shape.dim.add()
        d1.dim_value = samples
        print(f"  Set input '{inp.name}' shape to ({batch}, {samples})")
        break

    return model


def fix_output_shape(model: onnx.ModelProto, batch: int, samples_out: int) -> onnx.ModelProto:
    # Only fix the first (main audio) output. Multi-output models (e.g. streaming
    # with state tensors) have additional outputs whose shapes differ — those are
    # handled correctly by ONNX shape inference after the input is fixed.
    first_out = model.graph.output[0]
    tt = first_out.type.tensor_type
    if tt.HasField("shape"):
        del tt.shape.dim[:]
        d0 = tt.shape.dim.add()
        d0.dim_value = batch
        d1 = tt.shape.dim.add()
        d1.dim_value = samples_out
        print(f"  Set output '{first_out.name}' shape to ({batch}, {samples_out})")
    return model


def count_symbolic_dims(model: onnx.ModelProto) -> int:
    count = 0
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):  # symbolic
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16160, help="Input samples (1s @ 16kHz + 10ms context)")
    parser.add_argument("--samples_out", type=int, default=16000, help="Output samples")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    model = onnx.load(args.input)

    sym_before = count_symbolic_dims(model)
    print(f"Symbolic dims before: {sym_before}")

    print("Setting concrete input/output shapes ...")
    model = fix_input_shape(model, args.batch, args.samples)
    model = fix_output_shape(model, args.batch, args.samples_out)

    print("Running shape inference ...")
    model = shape_inference.infer_shapes(model, strict_mode=False)

    sym_after = count_symbolic_dims(model)
    print(f"Symbolic dims after:  {sym_after}")

    print("Checking model ...")
    onnx.checker.check_model(model)

    onnx.save(model, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
