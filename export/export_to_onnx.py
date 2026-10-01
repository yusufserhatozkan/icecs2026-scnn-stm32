"""ONNX export utilities for StreamSpikeNet.

Usage (from repo root):
    # DPSNN scnn_only after training (hparams loaded from ckpt)
    python export/export_to_onnx.py \\
        --ckpt_path egs/voicebank/<scnn_ckpt>.ckpt \\
        --output_path export/dpsnn_scnn128.onnx

By default a second file <stem>_xcubeai.onnx is also written with all
X-CUBE-AI compatibility fixes applied in sequence:
  1. Inline degenerate If nodes
  2. Remove dead nodes (Equal/Cast orphans after If inlining)
  3. Strip empty-string optional inputs (Pad crash)
  4. Set concrete input/output shapes (batch=1)
  5. ORT BASIC constant-folding (eliminates ConstantOfShape/Shape/Gather)
  6. ONNX shape inference post-ORT (annotates intermediate tensors)
  7. Strip foreign opset imports left over from ORT
  8. onnxsim (X-CUBE-AI's bundled version) — eliminates no-op Reshapes and
     fuses Concats
  9. Inflate every rank-2/3 Pad and every rank-3 ConvTranspose to 4D NCHW
     by wrapping with Unsqueeze/Squeeze.  X-CUBE-AI v2.2.0 cannot parse
     those ops at lower ranks (Pad: "Unknown dimensions: H/W",
     ConvTranspose: "TOOL ERROR: tuple index out of range").
 10. Eliminate BOOL tensors.  Replace Greater/GreaterOrEqual + Cast with
     Sub/Sign/Relu and Where(bool, A, B) with bool_float * A + (1 -
     bool_float) * B.  X-CUBE-AI emits "Unsupported operands type: BOOL"
     once the parser reaches the spike-threshold path.

Pass --no_xcubeai to skip steps 1-10 and write only the raw ONNX.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpsnn.models.dp_binary_net import StreamSpikeNet


def require_new_outputs(*paths) -> None:
    """Reject colliding or existing destinations before loading/exporting a model."""
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Export destinations must be distinct")
    for path in resolved:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing export: {path}")


def xcubeai_output_path(raw_path) -> str:
    path = Path(raw_path)
    return str(path.with_name(path.stem + "_xcubeai.onnx"))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ExportWrapper(nn.Module):
    """Single-tensor interface around StreamSpikeNet.

    Both models expect a nested batch tuple; only noisy_x is used in the
    computation. This wrapper accepts only noisy_x, builds the required
    tuple internally, and returns enhanced audio as a 2-D tensor
    (batch, output_len).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, noisy_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_x: (1, seq_len) noisy audio including the context prefix.
        Returns:
            enhanced: (1, output_len) denoised audio.
        """
        dummy_id = torch.zeros(1, device=noisy_x.device)
        dummy_len = torch.tensor(noisy_x.shape[-1], device=noisy_x.device)
        inputs = (dummy_id, noisy_x, dummy_len)
        dummy_targets = (dummy_id, torch.zeros_like(noisy_x), dummy_len)

        enhanced, _, _, _ = self.model((inputs, dummy_targets))
        return enhanced.reshape(noisy_x.shape[0], -1)


# ---------------------------------------------------------------------------
# Raw ONNX export
# ---------------------------------------------------------------------------

def load_from_checkpoint(ckpt_path: str) -> nn.Module:
    """Load a StreamSpikeNet from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = StreamSpikeNet(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def export_model(
    wrapper: ExportWrapper,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 13,
) -> None:
    """Export ExportWrapper to a raw ONNX file (no post-processing)."""
    require_new_outputs(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # No dynamic_axes: embedded target always uses batch=1 and a fixed
    # input_dim, so baking concrete shapes from the start avoids an entire
    # class of symbolic-dimension issues with X-CUBE-AI.
    with open(output_path, "xb") as output:
        torch.onnx.export(
            wrapper, dummy_input, output,
            input_names=["noisy_audio"], output_names=["enhanced_audio"],
            opset_version=opset_version,
        )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Raw ONNX exported  -> {output_path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# X-CUBE-AI post-processing pipeline
# ---------------------------------------------------------------------------

def _import_tools():
    """Add the tools/ directory to sys.path so we can import helper modules."""
    tools_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
    )
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)


def postprocess_for_xcubeai(
    onnx_path: str,
    output_path: str,
    input_dim: int = 16160,
    output_dim: int = 16000,
) -> str:
    """Run the full X-CUBE-AI compatibility pipeline on a raw ONNX file.

    Steps:
      1. Inline degenerate If nodes (PyTorch tracer artefacts).
      2. Remove dead nodes (Equal/Shape/Gather/Constant orphans after inlining).
      3. Strip empty-string optional inputs (crash source in X-CUBE-AI parser).
      4. Set concrete input/output shapes (batch=1, fixed seq length).
      5. ORT BASIC constant-folding (folds ConstantOfShape, Shape, Gather chains).
      6. ONNX shape inference post-ORT (annotates all 18K+ intermediate tensors
         with concrete shapes — eliminates 'Unknown dimensions: H' error).

    Args:
        onnx_path:   Path to the raw ONNX file produced by export_model().
        output_path: Where to write the X-CUBE-AI compatible ONNX.
        input_dim:   Input sequence length (default 16160 = 1s @16kHz + 10ms context).
        output_dim:  Output sequence length (default 16000 = 1s @16kHz).

    Returns:
        output_path
    """
    require_new_outputs(output_path)
    _import_tools()

    import onnx
    from onnx import shape_inference
    from inline_if_nodes import (
        inline_if_nodes, topological_sort,
        remove_dead_nodes, strip_empty_inputs,
    )
    from fix_shapes import fix_input_shape, fix_output_shape

    print("\n--- X-CUBE-AI post-processing pipeline ---")
    model = onnx.load(onnx_path)

    # Step 1 & 2: Inline If nodes, re-sort, remove dead nodes
    n_if = sum(1 for n in model.graph.node if n.op_type == "If")
    print(f"Step 1: Inlining {n_if} If nodes ...")
    model = inline_if_nodes(model)
    model = topological_sort(model)

    print("Step 2: Removing dead nodes ...")
    model = remove_dead_nodes(model)

    print("Step 2b: Stripping empty optional inputs ...")
    model = strip_empty_inputs(model)

    # Step 3: Concrete input/output shapes
    print(f"Step 3: Setting concrete shapes  input=(1,{input_dim})  output=(1,{output_dim}) ...")
    model = fix_input_shape(model, batch=1, samples=input_dim)
    # Only fix the first (main audio) output — multi-output models (e.g. streaming)
    # have additional state outputs whose shapes differ and must be left for shape inference.
    model = fix_output_shape(model, batch=1, samples_out=output_dim)

    # Step 4: ORT graph optimization — constant-folds Shape→Gather→ConstantOfShape
    # chains and any other runtime-but-actually-constant patterns.
    # Must run BEFORE shape inference so the folded graph is what gets annotated.
    # ORT_ENABLE_BASIC = constant folding + dead-code elimination only (no op fusion).
    # ORT_ENABLE_ALL would introduce FusedConv (ORT-internal, X-CUBE-AI rejects it).
    print("Step 4: ORT constant-folding + dead-code elimination ...")
    import onnxruntime as ort
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", dir=os.path.dirname(onnx_path) or ".", delete=False) as tmp:
        tmp_in = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", dir=os.path.dirname(onnx_path) or ".", delete=False) as tmp:
        tmp_out = tmp.name
    onnx.save(model, tmp_in)
    sess_opts = ort.SessionOptions()
    # BASIC = constant folding + dead-code elimination only.
    # ORT_ENABLE_ALL would also fuse Conv+ReLU into FusedConv (ORT-internal op
    # that X-CUBE-AI does not recognise), so we stop at BASIC.
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_opts.optimized_model_filepath = tmp_out
    sess_opts.log_severity_level = 3
    ort.InferenceSession(tmp_in, sess_opts, providers=["CPUExecutionProvider"])
    model = onnx.load(tmp_out)
    import os as _os
    _os.unlink(tmp_in)
    _os.unlink(tmp_out)

    # Step 5: Shape inference after ORT — ORT renames many tensors internally,
    # so the earlier bake/inference results no longer apply.  Running inference
    # here annotates every intermediate tensor with a concrete shape so X-CUBE-AI
    # never sees an "Unknown dimensions" error.
    print("Step 5: ONNX shape inference (post-ORT) ...")
    model = shape_inference.infer_shapes(model, strict_mode=False)
    sym_final = _count_symbolic_dims(model)
    print(f"  Symbolic dims remaining: {sym_final}")

    # Step 6: Strip foreign opset imports left over from ORT optimization.
    # ORT adds com.microsoft.nchwc, com.microsoft, org.pytorch.aten, etc. to
    # opset_import even when no node uses them.  Keep only domains actually
    # referenced by a node — in practice ('', 13).
    print("Step 6: Stripping foreign opset imports ...")
    used_domains = {n.domain or "" for n in model.graph.node}
    keep = [op for op in model.opset_import if (op.domain or "") in used_domains]
    removed = [op.domain for op in model.opset_import if op not in keep]
    del model.opset_import[:]
    model.opset_import.extend(keep)
    print(f"  Removed {len(removed)} unused opset imports: {removed}")

    # Step 7: onnxsim — flattens no-op Reshape((-1, K)) chains, fuses adjacent
    # Concat/Slice/etc.  Uses the version bundled with X-CUBE-AI itself.
    print("Step 7: onnxsim (graph simplification) ...")
    try:
        import onnxsim
        init_names = {i.name for i in model.graph.initializer}
        primary_inputs = [i for i in model.graph.input if i.name not in init_names]
        primary_input_name = primary_inputs[0].name if primary_inputs else "noisy_audio"
        model, ok = onnxsim.simplify(
            model,
            overwrite_input_shapes={primary_input_name: [1, input_dim]},
        )
        print(f"  onnxsim check: {'PASS' if ok else 'FAIL'}, nodes now: {len(model.graph.node)}")
        if not ok:
            raise RuntimeError("ONNX simplification failed equivalence validation")
    except ImportError:
        print("  onnxsim not installed — skipping (pip install onnxsim)")

    # Step 8: Inflate every Pad and ConvTranspose to rank-4 NCHW.  X-CUBE-AI
    # v2.2.0 cannot parse Pad on rank<4 (errors "Unknown dimensions: H/W")
    # or ConvTranspose1d (errors "tuple index out of range").  We wrap each
    # offending op as Unsqueeze -> 2D op -> Squeeze.
    print("Step 8: Inflating Pad / ConvTranspose to rank-4 NCHW ...")
    model = _inflate_pad_and_convtranspose(model)

    # Step 9: Eliminate BOOL tensors.  X-CUBE-AI v2.2.0 rejects bool with
    # "Unsupported operands type: BOOL" once the parser reaches the graph.
    # We replace every Greater/GreaterOrEqual + Cast(bool->float) with
    # Sub -> Sign -> Relu, and rewrite every Where(bool, A, B) as
    # bool_float * A + (1 - bool_float) * B.
    print("Step 9: Eliminating BOOL tensors ...")
    model = _eliminate_bool_tensors(model)

    # Re-run onnxsim and shape inference to fold the new ops where possible
    try:
        import onnxsim
        init_names2 = {i.name for i in model.graph.initializer}
        primary_inputs2 = [i for i in model.graph.input if i.name not in init_names2]
        primary_input_name2 = primary_inputs2[0].name if primary_inputs2 else "noisy_audio"
        model, ok = onnxsim.simplify(
            model,
            overwrite_input_shapes={primary_input_name2: [1, input_dim]},
        )
        print(f"  post-bool onnxsim: {'PASS' if ok else 'FAIL'}, nodes: {len(model.graph.node)}")
        if not ok:
            raise RuntimeError("Post-comparison ONNX simplification failed equivalence validation")
    except ImportError:
        pass
    model = shape_inference.infer_shapes(model, strict_mode=False)

    print("Checking model ...")
    onnx.checker.check_model(model)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "xb") as output:
        onnx.save(model, output)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nX-CUBE-AI ready  -> {output_path}  ({size_kb:.1f} KB)")
    print("-------------------------------------------\n")
    return output_path


def _eliminate_bool_tensors(model):
    """Replace every Greater/GreaterOrEqual + Cast(bool->float) and every
    Where(bool, A, B) with bool-free arithmetic equivalents.

    X-CUBE-AI v2.2.0 rejects bool tensors with "Unsupported operands type: BOOL"
    even though Cast(bool->float) and Where(bool, A, B) each parse fine in
    isolation.  This rewrites:
      Greater(x, c)        -> Relu(Sign(x - c))
      GreaterOrEqual(x, c) -> 1 - Relu(Sign(c - x))
      Cast(bool->float)  -> Identity         # bool input already replaced
      Where(bool, A, B)  -> bf*A + (1-bf)*B
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from collections import defaultdict, deque

    graph = model.graph

    ones_name = "_nobool_one_scalar"
    if ones_name not in {i.name for i in graph.initializer}:
        graph.initializer.append(numpy_helper.from_array(
            np.array(1.0, dtype=np.float32), ones_name))

    uid = [0]
    def fresh(base):
        uid[0] += 1
        return f"_nobool_{uid[0]}_{base}"

    bool_to_float = {}
    new_nodes = []
    n_cmp = n_cast = n_where = 0
    for n in graph.node:
        if n.op_type in ("Greater", "GreaterOrEqual"):
            x_name = n.input[0]; c_name = n.input[1]
            diff = fresh(n.output[0] + "_diff")
            sgn = fresh(n.output[0] + "_sign")
            rel = fresh(n.output[0] + "_relu")
            # The inclusive threshold must fire at equality; ReLU(Sign(x-c))
            # implements only strict Greater. Inputs must be finite.
            operands = [c_name, x_name] if n.op_type == "GreaterOrEqual" else [x_name, c_name]
            new_nodes.append(helper.make_node("Sub", operands, [diff]))
            new_nodes.append(helper.make_node("Sign", [diff], [sgn]))
            new_nodes.append(helper.make_node("Relu", [sgn], [rel]))
            if n.op_type == "GreaterOrEqual":
                inclusive = fresh(n.output[0] + "_inclusive")
                new_nodes.append(helper.make_node("Sub", [ones_name, rel], [inclusive]))
                rel = inclusive
            bool_to_float[n.output[0]] = rel
            n_cmp += 1
        else:
            new_nodes.append(n)

    final_nodes = []
    for n in new_nodes:
        if (n.op_type == "Cast" and len(n.input) == 1
                and n.input[0] in bool_to_float):
            final_nodes.append(helper.make_node(
                "Identity", [bool_to_float[n.input[0]]], [n.output[0]]))
            n_cast += 1
            continue
        if n.op_type == "Where" and n.input[0] in bool_to_float:
            bf = bool_to_float[n.input[0]]
            A = n.input[1]; B = n.input[2]
            inv = fresh(n.output[0] + "_inv")
            a_t = fresh(n.output[0] + "_a")
            b_t = fresh(n.output[0] + "_b")
            final_nodes.append(helper.make_node("Sub", [ones_name, bf], [inv]))
            final_nodes.append(helper.make_node("Mul", [bf, A], [a_t]))
            final_nodes.append(helper.make_node("Mul", [inv, B], [b_t]))
            final_nodes.append(helper.make_node("Add", [a_t, b_t], [n.output[0]]))
            n_where += 1
            continue
        final_nodes.append(n)

    print(f"  Eliminated bool: {n_cmp} cmp -> Sign/Relu, "
          f"{n_cast} Cast -> Identity, {n_where} Where -> arithmetic")
    del graph.node[:]
    graph.node.extend(final_nodes)
    del graph.value_info[:]

    # Topological re-sort because new nodes were appended
    n_total = len(graph.node)
    out_to_idx = {}
    for i, nd in enumerate(graph.node):
        for o in nd.output:
            if o:
                out_to_idx[o] = i
    indeg = [0] * n_total
    succ = defaultdict(list)
    for i, nd in enumerate(graph.node):
        for inp in nd.input:
            if not inp:
                continue
            if inp in out_to_idx:
                j = out_to_idx[inp]
                if j != i:
                    succ[j].append(i)
                    indeg[i] += 1
    q = deque(i for i in range(n_total) if indeg[i] == 0)
    order = []
    while q:
        i = q.popleft(); order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    if len(order) != n_total:
        raise RuntimeError(
            f"Topo sort failed after bool elimination: "
            f"{n_total - len(order)} unresolved")
    sorted_nodes = [graph.node[i] for i in order]
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return model


def _inflate_pad_and_convtranspose(model):
    """Wrap rank-2/3 Pad and ConvTranspose ops as 4D NCHW.

    X-CUBE-AI v2.2.0 rejects Pad on rank<4 and any rank-3 ConvTranspose.
    For each such node we insert Unsqueeze before it and Squeeze after it,
    inflating the spatial part to (H=1, W=spatial).  Op attributes
    (kernel_shape, strides, pads, dilations, output_padding) are extended
    with the corresponding extra-dim entries (1s for spatial, 0s for pads).
    The `pads` initializer of Pad and the weight tensor of ConvTranspose
    are also inflated to match.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    graph = model.graph
    vi_map = {vi.name: vi for vi in graph.value_info}
    vi_map.update({i.name: i for i in graph.input})
    vi_map.update({o.name: o for o in graph.output})
    init_map = {i.name: i for i in graph.initializer}

    def rank_of(name):
        if name in init_map:
            return len(init_map[name].dims)
        vi = vi_map.get(name)
        if (vi is None or not vi.type.HasField("tensor_type")
                or not vi.type.tensor_type.HasField("shape")):
            return None
        return len(vi.type.tensor_type.shape.dim)

    axes_cache = {}
    def get_axes_init(axes_tuple):
        key = tuple(axes_tuple)
        if key in axes_cache:
            return axes_cache[key]
        name = f"_inflate_axes_{'_'.join(map(str, key))}"
        graph.initializer.append(
            numpy_helper.from_array(np.array(list(key), dtype=np.int64), name))
        axes_cache[key] = name
        return name

    uid = [0]
    def fresh(base):
        uid[0] += 1
        return f"{base}_inflated_{uid[0]}"

    new_nodes = []
    n_pad = n_ct = 0
    for node in graph.node:
        if node.op_type not in ("Pad", "ConvTranspose"):
            new_nodes.append(node)
            continue
        in_rank = rank_of(node.input[0])
        # Fallback: ConvTranspose's weight rank tells us the op rank.
        # weight is (in_C, out_C, K) for 1D, (in_C, out_C, kH, kW) for 2D.
        # When the data input comes from DequantizeLinear, its shape may not
        # be populated in value_info; the weight initializer always has dims.
        if in_rank is None and node.op_type == "ConvTranspose" and len(node.input) >= 2:
            in_rank = rank_of(node.input[1])
        if in_rank is None or in_rank >= 4:
            new_nodes.append(node)
            continue
        extra = 4 - in_rank
        # Orientation: Pad uses time-as-W (insert H,C before spatial); ConvTranspose
        # uses time-as-H (append W after spatial).  X-CUBE-AI's ConvTranspose
        # parser specifically wants kernel = (K, 1), i.e. K on the H axis.
        # Pad works either way but time-as-W matches the bisect tests.
        if node.op_type == "ConvTranspose":
            # Append W=1 after the original spatial dim
            if in_rank == 3:
                unsq_axes = [3]            # (N,C,T) -> (N,C,T,1)
            elif in_rank == 2:
                unsq_axes = [1, 3]         # (N,T)   -> (N,1,T,1)
            else:
                unsq_axes = list(range(in_rank, in_rank + extra))
            # ConvTranspose: K goes to kH, kW=1
            spatial_extend_kernel = lambda old: list(old) + [1] * extra
            spatial_extend_pads = lambda old: (
                # old = [b1..bN, e1..eN] -> insert 0 after the H entries on each half
                list(old[:len(old)//2]) + [0] * extra + list(old[len(old)//2:]) + [0] * extra
            )
        else:
            # Pad uses time-as-W: insert H,C before original spatial dim
            if in_rank == 3:
                unsq_axes = [2]            # (N,C,T) -> (N,C,1,T)
            elif in_rank == 2:
                unsq_axes = [1, 2]         # (N,T)   -> (N,1,1,T)
            else:
                unsq_axes = list(range(1, 1 + extra))
            spatial_extend_kernel = lambda old: [1] * extra + list(old)
            spatial_extend_pads = lambda old: (
                [0] * extra + list(old[:len(old)//2]) + [0] * extra + list(old[len(old)//2:])
            )
        axes_name = get_axes_init(unsq_axes)

        pre_name = fresh(node.input[0] + "_pre")
        new_nodes.append(helper.make_node(
            "Unsqueeze", [node.input[0], axes_name], [pre_name]))

        inflated_out = fresh(node.output[0] + "_inflated")
        new_inputs = list(node.input)
        new_inputs[0] = pre_name

        new_attrs = []
        for a in node.attribute:
            if a.name in ("kernel_shape", "strides", "dilations"):
                new_attrs.append(helper.make_attribute(
                    a.name, spatial_extend_kernel(a.ints)))
            elif a.name == "pads":
                new_attrs.append(helper.make_attribute(
                    "pads", spatial_extend_pads(a.ints)))
            elif a.name == "output_padding":
                new_attrs.append(helper.make_attribute(
                    "output_padding", spatial_extend_kernel(a.ints)))
            else:
                new_attrs.append(a)

        if (node.op_type == "Pad" and len(node.input) >= 2
                and node.input[1] in init_map):
            old = numpy_helper.to_array(init_map[node.input[1]])
            half = len(old) // 2
            # Pad time-as-W: zeros for new H,C dims at the front of each half
            new_pads = np.concatenate([
                np.zeros(extra, old.dtype), old[:half],
                np.zeros(extra, old.dtype), old[half:],
            ])
            new_name = node.input[1] + "_inflated"
            if new_name not in init_map:
                init = numpy_helper.from_array(new_pads, new_name)
                graph.initializer.append(init)
                init_map[new_name] = init
            new_inputs[1] = new_name
            n_pad += 1

        if node.op_type == "ConvTranspose" and len(node.input) >= 2:
            # The weight input may be:
            #   (a) a direct initializer (FP32 case)
            #   (b) the output of a DequantizeLinear that consumes an INT8
            #       initializer (QDQ INT8 case)
            # In case (b), rewrite the original INT8 weight initializer; the
            # DQ output shape will follow once shape inference re-runs.
            w_input = node.input[1]
            w_init_name = None
            if w_input in init_map:
                w_init_name = w_input
            else:
                # Find producer of w_input
                for pn in graph.node:
                    if w_input in pn.output and pn.op_type == "DequantizeLinear":
                        if pn.input and pn.input[0] in init_map:
                            w_init_name = pn.input[0]
                        break
            if w_init_name is not None:
                arr = numpy_helper.to_array(init_map[w_init_name])
                if arr.ndim == 3:
                    # ConvTranspose time-as-H: (in_C, out_C, K) -> (in_C, out_C, K, 1)
                    new_arr = arr[:, :, :, np.newaxis]
                    # Replace the initializer in place so the DQ keeps working
                    new_init = numpy_helper.from_array(new_arr, w_init_name)
                    init_map[w_init_name].CopyFrom(new_init)
            n_ct += 1

        op = helper.make_node(node.op_type, new_inputs, [inflated_out],
                              name=node.name or None)
        op.attribute.extend(new_attrs)
        new_nodes.append(op)
        new_nodes.append(helper.make_node(
            "Squeeze", [inflated_out, axes_name], [node.output[0]]))

    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.value_info[:]  # ranks changed; let shape inference recompute
    print(f"  Inflated {n_pad} Pad and {n_ct} ConvTranspose nodes to NCHW")
    return model


def _count_symbolic_dims(model) -> int:
    import onnx
    count = 0
    for vi in (
        list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    ):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    count += 1
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export StreamSpikeNet to ONNX")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", required=True,
                        help="Path for the raw ONNX export")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--no_xcubeai", action="store_true",
                        help="Skip X-CUBE-AI post-processing pipeline")
    parser.add_argument("--xcubeai_out", default=None,
                        help="Path for the X-CUBE-AI output (default: <stem>_xcubeai.onnx)")
    args = parser.parse_args()

    xcubeai_path = args.xcubeai_out or xcubeai_output_path(args.output_path)
    require_new_outputs(*([args.output_path] if args.no_xcubeai else
                          [args.output_path, xcubeai_path]))

    print(f"Loading checkpoint: {args.ckpt_path}")
    model = load_from_checkpoint(args.ckpt_path)

    input_dim = model.hparams["input_dim"]
    context_dim = model.hparams["context_dim"]
    print(f"Model: input_dim={input_dim}, context_dim={context_dim}, "
          f"N={model.hparams['N']}, B={model.hparams['B']}, H={model.hparams['H']}, "
          f"scnn_only={model.hparams.get('scnn_only', False)}")

    output_size = model.time_steps * model.stride + model.L - model.stride
    print(f"Output samples: {output_size}")


    wrapper = ExportWrapper(model)
    dummy_input = torch.randn(1, input_dim)
    print(f"Dummy input shape: {dummy_input.shape}")

    print("Exporting raw ONNX ...")
    try:
        export_model(wrapper, dummy_input, args.output_path, args.opset)
        print("Raw export SUCCEEDED.")
    except Exception as exc:
        print(f"\nExport FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)

    if args.no_xcubeai:
        print("Skipping X-CUBE-AI post-processing (--no_xcubeai).")
        return

    try:
        postprocess_for_xcubeai(
            onnx_path=args.output_path,
            output_path=xcubeai_path,
            input_dim=input_dim,
            output_dim=output_size,
        )
    except Exception as exc:
        print(f"\nX-CUBE-AI post-processing FAILED: {type(exc).__name__}: {exc}")
        print("Raw ONNX is still usable — run the tool scripts manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
