"""Boundary checks for arithmetic replacement of ONNX Boolean comparisons."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from export.export_to_onnx import _eliminate_bool_tensors


@pytest.mark.parametrize("operator,expected", (("Greater", [0.0, 0.0, 1.0]),
                                              ("GreaterOrEqual", [0.0, 1.0, 1.0])))
def test_comparison_preserves_below_equal_above_threshold(operator, expected):
    graph = helper.make_graph(
        [helper.make_node(operator, ["x", "threshold"], ["condition"]),
         helper.make_node("Cast", ["condition"], ["cast"], to=TensorProto.FLOAT),
         helper.make_node("Where", ["condition", "a", "b"], ["selected"])],
        "threshold_boundary",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("cast", TensorProto.FLOAT, [3]),
         helper.make_tensor_value_info("selected", TensorProto.FLOAT, [3])],
        initializer=[helper.make_tensor("threshold", TensorProto.FLOAT, [], [1.0]),
                     helper.make_tensor("a", TensorProto.FLOAT, [], [7.0]),
                     helper.make_tensor("b", TensorProto.FLOAT, [], [-2.0])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    values = np.array([np.nextafter(np.float32(1.0), np.float32(0.0)), 1.0,
                       np.nextafter(np.float32(1.0), np.float32(2.0))], dtype=np.float32)
    reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    original = reference.run(None, {"x": values})
    changed = _eliminate_bool_tensors(model)
    onnx.checker.check_model(changed)
    session = ort.InferenceSession(changed.SerializeToString(), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": values})
    np.testing.assert_array_equal(actual[0], expected)
    for before, after in zip(original, actual):
        np.testing.assert_array_equal(before, after)
    assert not any(node.op_type in ("Greater", "GreaterOrEqual", "Where") for node in changed.graph.node)
