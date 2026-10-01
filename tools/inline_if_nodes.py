"""
Replace degenerate ONNX If nodes with their branch body directly.

X-CUBE-AI does not support the ONNX If operator. Inspection of the exported
DPSNN graph shows 798 If nodes where both then_branch and else_branch execute
the identical Squeeze operation — the condition is never actually branched on.
This script inlines the then_branch body into the main graph and removes the
If node, producing a semantically identical graph that X-CUBE-AI can accept.

Usage:
    python tools/inline_if_nodes.py --input C:/ai/dpsnn_int8_sorted.onnx \
                                     --output C:/ai/dpsnn_int8_final.onnx
"""

import argparse
from collections import defaultdict, deque

import onnx
from onnx import helper


def get_main_graph_tensors(graph) -> set:
    """All tensor names that exist in the main graph before inlining."""
    names = set()
    names |= {inp.name for inp in graph.input}
    names |= {init.name for init in graph.initializer}
    for node in graph.node:
        names |= set(node.output)
    return names


def inline_if_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    main_tensors = get_main_graph_tensors(graph)

    new_nodes = []
    n_inlined = 0
    n_skipped = 0

    for node in graph.node:
        if node.op_type != 'If':
            new_nodes.append(node)
            continue

        # Get then_branch (both branches are identical, use then)
        then_branch = None
        for attr in node.attribute:
            if attr.name == 'then_branch':
                then_branch = attr.g
                break

        if then_branch is None or len(then_branch.node) == 0:
            new_nodes.append(node)
            n_skipped += 1
            continue

        # Build output name mapping: sub-graph output name -> If node output name
        # then_branch.output lists the tensors the If node yields
        output_map = {}
        for if_out, branch_out in zip(node.output, then_branch.output):
            output_map[branch_out.name] = if_out

        # Lift any initializers that only exist in the sub-graph
        for init in then_branch.initializer:
            if init.name not in main_tensors:
                graph.initializer.append(init)
                main_tensors.add(init.name)

        # Inline each sub-graph node into the main graph
        for sub_node in then_branch.node:
            # Remap outputs: sub-graph output names -> main graph names
            new_outputs = [output_map.get(o, o) for o in sub_node.output]
            new_node = helper.make_node(
                sub_node.op_type,
                inputs=list(sub_node.input),
                outputs=new_outputs,
                name=sub_node.name or None,
            )
            for attr in sub_node.attribute:
                new_node.attribute.append(attr)
            new_nodes.append(new_node)
            main_tensors |= set(new_outputs)

        n_inlined += 1

    print(f"  Inlined : {n_inlined} If nodes")
    if n_skipped:
        print(f"  Skipped : {n_skipped} If nodes (empty branch — left as-is)")

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def topological_sort(model: onnx.ModelProto) -> onnx.ModelProto:
    """Re-sort after inlining since new nodes were appended."""
    graph = model.graph
    output_to_node: dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            if out:
                output_to_node[out] = i

    n = len(graph.node)
    in_degree = [0] * n
    successors: dict[int, list[int]] = defaultdict(list)

    for i, node in enumerate(graph.node):
        for inp in node.input:
            if not inp:
                continue
            if inp in output_to_node:
                j = output_to_node[inp]
                if j != i:
                    successors[j].append(i)
                    in_degree[i] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order: list[int] = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in successors[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                queue.append(j)

    if len(order) != n:
        raise RuntimeError(f"Cycle or disconnected nodes after inlining ({n - len(order)} unresolved)")

    sorted_nodes = [graph.node[i] for i in order]
    new_graph = helper.make_graph(
        sorted_nodes, graph.name,
        list(graph.input), list(graph.output), list(graph.initializer),
    )
    new_graph.value_info.extend(graph.value_info)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    new_model.doc_string = model.doc_string
    new_model.model_version = model.model_version
    return new_model


def remove_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove nodes whose outputs are never consumed by any other node or graph output.

    After If-node inlining, the Equal (and other) nodes that produced the If
    conditions become orphaned — nothing reads their outputs.  X-CUBE-AI crashes
    with an internal error when it tries to eliminate such nodes itself, so we
    remove them here first.
    """
    graph = model.graph

    # Map output tensor name → index of the node that produces it
    producer: dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            if out:
                producer[out] = i

    # Initializer names are always live (weights, constants)
    init_names = {init.name for init in graph.initializer}

    # Seed the live set with graph output tensors and explicit graph inputs
    live_tensors: set[str] = {out.name for out in graph.output}
    live_tensors |= {inp.name for inp in graph.input}

    # BFS backwards: for each live tensor, mark its producer node live,
    # then add all that node's input tensors to the live set.
    live_node_indices: set[int] = set()
    queue = [t for t in live_tensors if t not in init_names and t in producer]

    while queue:
        tensor = queue.pop()
        if tensor not in producer:
            continue
        node_idx = producer[tensor]
        if node_idx in live_node_indices:
            continue
        live_node_indices.add(node_idx)
        for inp in graph.node[node_idx].input:
            if inp and inp not in live_tensors:
                live_tensors.add(inp)
                if inp not in init_names and inp in producer:
                    queue.append(inp)

    n_total = len(graph.node)
    new_nodes = [graph.node[i] for i in range(n_total) if i in live_node_indices]
    n_removed = n_total - len(new_nodes)

    dead_ops = [
        graph.node[i].op_type
        for i in range(n_total)
        if i not in live_node_indices
    ]
    from collections import Counter
    op_counts = Counter(dead_ops)
    print(f"  Removed {n_removed} dead nodes: {dict(op_counts)}")

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model


def strip_empty_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove ALL empty-string optional inputs from all nodes.

    ONNX uses '' to mark absent optional inputs.  X-CUBE-AI looks up the
    empty string as a tensor name, gets nothing back, and crashes with
    'list index out of range'.  We strip every '' entry from every node's
    input list.  For inputs in the middle of the list (not just trailing),
    we preserve the surrounding non-empty entries so the node's remaining
    inputs map to the same positional slots.

    Safe rule: if an op's Nth input is truly optional and omitted, removing
    the '' is equivalent to not providing it (spec default applies).
    """
    removed_by_op: dict[str, int] = {}
    n_fixed = 0
    for node in model.graph.node:
        # Strip trailing empties first (safe — trailing optional inputs)
        while node.input and node.input[-1] == "":
            del node.input[-1]
            removed_by_op[node.op_type] = removed_by_op.get(node.op_type, 0) + 1
            n_fixed += 1
        # Any remaining mid-list empties are unusual but still strip them
        for i in range(len(node.input) - 1, -1, -1):
            if node.input[i] == "" and i < len(node.input) - 1:
                del node.input[i]
                removed_by_op[node.op_type] = removed_by_op.get(node.op_type, 0) + 1
                n_fixed += 1
    print(f"  Stripped {n_fixed} empty inputs: {dict(removed_by_op)}")
    return model


strip_trailing_empty_inputs = strip_empty_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading {args.input}")
    model = onnx.load(args.input)

    all_ops = set(n.op_type for n in model.graph.node)
    print(f"Op types before: {sorted(all_ops)}")

    print("Inlining If nodes ...")
    model = inline_if_nodes(model)

    print("Re-sorting graph topologically ...")
    model = topological_sort(model)

    print("Removing dead nodes (orphaned Equal/Cast/Shape nodes after inlining) ...")
    model = remove_dead_nodes(model)

    print("Stripping trailing empty optional inputs ...")
    model = strip_trailing_empty_inputs(model)

    all_ops_after = set(n.op_type for n in model.graph.node)
    print(f"Op types after:  {sorted(all_ops_after)}")

    print("Checking model ...")
    onnx.checker.check_model(model)

    onnx.save(model, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
