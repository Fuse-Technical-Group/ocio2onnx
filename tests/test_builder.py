"""The builder names tensors and settles the graph's interface; it holds no
color knowledge (§spec:emitted-graph).
"""

import numpy as np
import onnx
import pytest

from ocio2onnx.builder import (
    CHANNEL_SHAPE,
    CHANNELS,
    IMAGE_SHAPE,
    INPUT,
    OUTPUT,
    PARAMETER_SHAPE,
    SCALAR_SHAPE,
    GraphBuilder,
    parameters,
)
from ocio2onnx.oracle import run_graph

SAMPLES = np.array([[[[-2.0, 0.5]], [[0.25, 4.0]], [[1.0, -1.0]]]], dtype=np.float32)


def test_the_graph_interface_is_three_channels_with_spatial_dims_free():
    assert CHANNELS == 3
    assert IMAGE_SHAPE == ("N", CHANNELS, "H", "W")


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


def test_a_scalar_parameter_is_a_graph_input_defaulting_to_its_ocio_value():
    """A live parameter is an ONNX input backed by an initializer, which is how
    the format spells "input with a default" (§spec:dynamic-properties)."""
    builder = GraphBuilder()
    gain = builder.scalar_parameter("GAIN", 2.0)
    model = builder.build(builder.mul(INPUT, gain), "gain", {})

    assert [value.name for value in model.graph.input] == [INPUT, "GAIN"]
    assert parameters(model) == {"GAIN": pytest.approx([2.0])}
    assert run_graph(model, SAMPLES) == pytest.approx(SAMPLES * 2.0)


def test_a_scalar_parameter_is_varied_without_recompiling():
    """The whole point of a graph over a baked table: one artifact, a knob a
    consumer turns per frame (§spec:dynamic-properties)."""
    builder = GraphBuilder()
    gain = builder.scalar_parameter("GAIN", 2.0)
    model = builder.build(builder.mul(INPUT, gain), "gain", {})

    assert run_graph(model, SAMPLES, {"GAIN": [5.0]}) == pytest.approx(SAMPLES * 5.0)


def test_a_channel_parameter_is_declared_flat_and_broadcasts_across_channels():
    """A consumer binds three numbers, not a four-dimensional array; the
    reshape onto the image's channel axis is the graph's business."""
    builder = GraphBuilder()
    gain = builder.channel_parameter("GAIN", [1.0, 2.0, 3.0])
    model = builder.build(builder.mul(INPUT, gain), "gain", {})

    (declared,) = [value for value in model.graph.input if value.name == "GAIN"]
    assert [d.dim_value for d in declared.type.tensor_type.shape.dim] == list(
        PARAMETER_SHAPE
    )
    assert run_graph(model, SAMPLES, {"GAIN": [4.0, 5.0, 6.0]}) == pytest.approx(
        SAMPLES * np.array([4.0, 5.0, 6.0]).reshape(CHANNEL_SHAPE)
    )


def test_two_ops_claiming_one_parameter_name_get_a_knob_each():
    """Two graded ops in one transform carry their own defaults, so one shared
    input would have to drop one of them."""
    builder = GraphBuilder()
    first = builder.scalar_parameter("GAIN", 2.0)
    second = builder.scalar_parameter("GAIN", 3.0)
    model = builder.build(builder.mul(builder.mul(INPUT, first), second), "gain", {})

    assert list(parameters(model)) == ["GAIN", "GAIN_2"]
    assert run_graph(model, SAMPLES) == pytest.approx(SAMPLES * 6.0)
    assert run_graph(model, SAMPLES, {"GAIN_2": [1.0]}) == pytest.approx(SAMPLES * 2.0)


def test_a_graph_without_a_live_parameter_declares_only_the_image():
    model = GraphBuilder().build(INPUT, "identity", {})
    assert [value.name for value in model.graph.input] == [INPUT]
    assert parameters(model) == {}


def test_a_parameter_default_is_float32_at_its_declared_shape():
    builder = GraphBuilder()
    builder.scalar_parameter("GAIN", 2.0)
    builder.channel_parameter("TINT", [1.0, 2.0, 3.0])
    model = builder.build(INPUT, "shapes", {})

    defaults = parameters(model)
    assert defaults["GAIN"].shape == SCALAR_SHAPE
    assert defaults["TINT"].shape == PARAMETER_SHAPE
    assert all(value.dtype == np.float32 for value in defaults.values())
