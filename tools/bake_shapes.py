"""
Bake concrete shapes into every tensor in an ONNX graph.

X-CUBE-AI rejects symbolic dimensions (e.g. 'batch_size', 'H').
This script runs a single OnnxRuntime forward pass on a zero input,
collects the actual shape of every intermediate tensor, and writes
those concrete integers back into the model's value_info entries.

Usage:
    python tools/bake_shapes.py --input C:/ai/dpsnn_int8_static.onnx \
                                 --output C:/ai/dpsnn_int8_baked.onnx
"""

import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def collect_shapes(model: onnx.ModelProto, input_shape: tuple) -> dict:
    """Run ORT with all intermediate tensors as outputs, return name->shape map."""
    # Build the set of tensors actually produced by non-Constant nodes in the main graph
    valid_outputs = set()
    for node in model.graph.node:
        if node.op_type not in ("Constant", "ConstantOfShape"):
            valid_outputs.update(o for o in node.output if o)

    # Existing graph outputs are already computed — don't re-add them
    existing_outputs = {o.name for o in model.graph.output}

    # value_info names that are valid intermediate tensors
    vi_names = {vi.name for vi in model.graph.value_info}

    extra_outputs = [
        name for name in valid_outputs
        if name in vi_names and name not in existing_outputs
    ]

    # Build a name -> value_info map to preserve correct types
    vi_map = {vi.name: vi for vi in model.graph.value_info}

    augmented = onnx.ModelProto()
    augmented.CopyFrom(model)
    for name in extra_outputs:
        if name in vi_map:
            augmented.graph.output.append(vi_map[name])

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # suppress warnings
    sess = ort.InferenceSession(
        augmented.SerializeToString(),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    # Find the main input name (non-initializer)
    init_names = {init.name for init in model.graph.initializer}
    main_input = next(
        inp.name for inp in model.graph.input if inp.name not in init_names
    )

    dummy = np.zeros(input_shape, dtype=np.float32)
    outputs = sess.run(None, {main_input: dummy})

    # Map output name -> shape
    all_output_names = [o.name for o in augmented.graph.output]
    shape_map = {}
    for name, arr in zip(all_output_names, outputs):
        if arr is not None:
            shape_map[name] = list(arr.shape)

    return shape_map


def bake_shapes(model: onnx.ModelProto, shape_map: dict) -> onnx.ModelProto:
    """Replace all symbolic dims with concrete integers from shape_map."""
    def apply(vi):
        name = vi.name
        if name not in shape_map:
            return
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return
        concrete = shape_map[name]
        if len(concrete) != len(tt.shape.dim):
            return
        for dim, val in zip(tt.shape.dim, concrete):
            dim.ClearField("dim_param")
            dim.dim_value = val

    for vi in model.graph.value_info:
        apply(vi)
    for vi in model.graph.input:
        apply(vi)
    for vi in model.graph.output:
        apply(vi)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16160)
    args = parser.parse_args()

    print(f"Loading {args.input}")
    model = onnx.load(args.input)

    print(f"Running ORT forward pass (input shape: ({args.batch}, {args.samples})) ...")
    print("  This may take 2-5 minutes for 399 unrolled time steps ...")
    shape_map = collect_shapes(model, (args.batch, args.samples))
    print(f"  Collected shapes for {len(shape_map)} tensors")

    print("Baking concrete shapes from ORT pass ...")
    model = bake_shapes(model, shape_map)

    # Brute-force pass: replace any remaining symbolic dims with known values
    # All remaining symbolic dims are 'batch_size' = 1 (confirmed by inspection)
    known = {"batch_size": args.batch}
    remaining = 0
    forced = 0
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        if not vi.type.tensor_type.HasField("shape"):
            continue
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_param"):
                param = d.dim_param
                if not param:  # empty string — skip
                    continue
                if param in known:
                    d.ClearField("dim_param")
                    d.dim_value = known[param]
                    forced += 1
                else:
                    remaining += 1
    print(f"Force-replaced {forced} symbolic dims | Still symbolic: {remaining}")

    print("Checking model ...")
    onnx.checker.check_model(model)

    onnx.save(model, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
