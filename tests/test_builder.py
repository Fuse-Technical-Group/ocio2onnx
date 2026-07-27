"""The builder names tensors and settles the graph's interface; it holds no
color knowledge (§road:oracle-harness, §spec:emitted-graph).
"""

import numpy as np
import onnx
import pytest

from ocio2onnx.builder import (
    CHANNEL_SHAPE,
    INPUT,
    OUTPUT,
    SCALAR_SHAPE,
    GraphBuilder,
)
from ocio2onnx.oracle import run_graph

SAMPLES = np.array([[[[-2.0, 0.5]], [[0.25, 4.0]], [[1.0, -1.0]]]], dtype=np.float32)


def test_names_are_unique_per_hint():
    builder = GraphBuilder()
    assert builder.name("x") != builder.name("x")


def test_a_per_channel_constant_broadcasts_against_the_image():
    builder = GraphBuilder()
    name = builder.per_channel("gain", [1.0, 2.0, 3.0])
    model = builder.build(builder.mul(INPUT, name), "gain", {})
    (tensor,) = model.graph.initializer
    assert onnx.numpy_helper.to_array(tensor).shape == CHANNEL_SHAPE
    assert run_graph(model, SAMPLES) == pytest.approx(
        SAMPLES * np.array([1.0, 2.0, 3.0]).reshape(CHANNEL_SHAPE)
    )


def test_a_scalar_constant_is_rank_one():
    builder = GraphBuilder()
    name = builder.scalar("bias", 0.5)
    model = builder.build(builder.add(INPUT, name), "bias", {})
    (tensor,) = model.graph.initializer
    assert onnx.numpy_helper.to_array(tensor).shape == SCALAR_SHAPE
    assert run_graph(model, SAMPLES) == pytest.approx(SAMPLES + 0.5)


def test_the_arithmetic_helpers_chain():
    builder = GraphBuilder()
    two = builder.scalar("two", 2.0)
    chain = builder.div(builder.sub(builder.mul(INPUT, two), two), two)
    model = builder.build(chain, "chain", {})
    assert [node.op_type for node in model.graph.node] == [
        "Mul",
        "Sub",
        "Div",
        "Identity",
    ]
    assert run_graph(model, SAMPLES) == pytest.approx(SAMPLES - 1.0)


def test_pow_and_where_select_between_two_branches():
    """ONNX has no per-element branching, so a breakpoint evaluates both sides
    and selects (§spec:op-emission)."""
    builder = GraphBuilder()
    zero = builder.scalar("zero", 0.0)
    two = builder.scalar("two", 2.0)
    below = builder.op("Less", [INPUT, zero])
    model = builder.build(
        builder.where(below, zero, builder.pow(INPUT, two)), "branch", {}
    )
    assert run_graph(model, SAMPLES) == pytest.approx(
        np.where(SAMPLES < 0.0, 0.0, SAMPLES**2)
    )


def test_an_empty_chain_is_still_a_valid_graph():
    model = GraphBuilder().build(INPUT, "identity", {})
    assert [value.name for value in model.graph.output] == [OUTPUT]
    assert np.array_equal(run_graph(model, SAMPLES), SAMPLES)
