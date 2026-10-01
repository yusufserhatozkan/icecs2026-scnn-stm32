"""
Topologically sort an ONNX graph's nodes.

X-CUBE-AI requires nodes to appear in topological order. Our QDQ insertion
script appends new nodes at the end without resorting, which breaks this.

Usage:
    python tools/sort_onnx_graph.py --input C:/ai/dpsnn_int8.onnx --output C:/ai/dpsnn_int8_sorted.onnx
"""

import argparse
from collections import defaultdict, deque

import onnx
from onnx import helper


def topological_sort(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph

    # Tensors that are always available (initializers + graph inputs)
    available = {init.name for init in graph.initializer}
    available |= {inp.name for inp in graph.input}

    # Map each output tensor name -> index of the node that produces it
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

    # Kahn's BFS
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
        raise RuntimeError(
            f"Graph has a cycle or disconnected nodes "
            f"({n - len(order)} nodes could not be sorted)"
        )

    sorted_nodes = [graph.node[i] for i in order]

    new_graph = helper.make_graph(
        sorted_nodes,
        graph.name,
        list(graph.input),
        list(graph.output),
        list(graph.initializer),
    )
    new_graph.value_info.extend(graph.value_info)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    new_model.doc_string = model.doc_string
    new_model.model_version = model.model_version
    return new_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading  {args.input}")
    model = onnx.load(args.input)
    n_before = len(model.graph.node)

    print(f"Sorting {n_before} nodes topologically ...")
    model = topological_sort(model)

    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"Saved   {args.output}  ({n_before} nodes, now sorted)")


if __name__ == "__main__":
    main()
