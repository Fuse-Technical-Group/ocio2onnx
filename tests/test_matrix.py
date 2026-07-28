"""``Matrix`` emits a 1x1 convolution, and the oracle says whether the
parameters were read correctly (§spec:verification, §spec:op-emission).

``Matrix`` is the op whose correctness is legible by inspection, which is what
makes it the right one to exercise the harness on: a swapped row is visible in
the coefficients and must also be visible in the verdict.
"""

import numpy as np
import onnx
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import (
    METADATA_PREFIX,
    enumerate_transforms,
    resolve_colorspaces,
)
from ocio2onnx.builder import (
    IMAGE_SHAPE,
    INPUT,
    IR_VERSION,
    OPSET,
    OUTPUT,
    PRECISION,
    GraphBuilder,
)
from ocio2onnx.compiler import UnsupportedOpError, compile_processor, op_names
from ocio2onnx.oracle import compare, cpu_reference, lattice, run_graph, verify

PAIR = ("ACEScg", "ACES2065-1")

#: Matrix-only transforms in the pinned config, counted by the sweep below.
MATRIX_ONLY = 34


@pytest.fixture(scope="module")
def matrix_only(config):
    """Every transform in the pinned config built from ``Matrix`` alone."""
    return [
        (label, processor)
        for label, processor in enumerate_transforms(config)
        if op_names(processor) == ["Matrix"]
    ]


def matrix_transform(matrix, offset=(0.0, 0.0, 0.0, 0.0), *, inverse=False):
    """A bare ``MatrixTransform`` from a 4x4 and a 4-offset."""
    transform = OCIO.MatrixTransform()
    transform.setMatrix(list(np.asarray(matrix, dtype=np.float64).ravel()))
    transform.setOffset(list(offset))
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def emit(transform):
    """The model for a single transform, bypassing the compiler."""
    builder = GraphBuilder()
    output = emitters.emit_matrix(builder, transform, INPUT)
    return builder.build(output, "matrix", {})


def test_the_registry_carries_matrix_and_its_breakpoint_hook():
    entry = emitters.REGISTRY["MatrixTransform"]
    assert entry.emit is emitters.emit_matrix
    assert entry.breakpoints(OCIO.MatrixTransform()) == []


def test_an_unregistered_transform_yields_no_emitter():
    unregistered = OCIO.FixedFunctionTransform(
        OCIO.FIXED_FUNCTION_REC2100_SURROUND, [1.0]
    )
    assert emitters.emitter_for(unregistered) is None
    assert emitters.breakpoints(unregistered) == []


def test_matrix_emits_a_1x1_convolution():
    model = emit(matrix_transform(np.eye(4)))
    assert [node.op_type for node in model.graph.node] == ["Conv", "Identity"]
    weight, bias = model.graph.initializer
    assert onnx.numpy_helper.to_array(weight).shape == (3, 3, 1, 1)
    assert onnx.numpy_helper.to_array(bias).shape == (3,)


def test_the_emitted_graph_is_float32_with_spatial_dimensions_free():
    model = emit(matrix_transform(np.eye(4)))
    (graph_input,) = model.graph.input
    (graph_output,) = model.graph.output
    assert graph_input.name == INPUT
    assert graph_output.name == OUTPUT
    for value in (graph_input, graph_output):
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        dims = value.type.tensor_type.shape.dim
        assert [d.dim_param or d.dim_value for d in dims] == list(IMAGE_SHAPE)


def test_the_emitted_graph_pins_its_opset_and_ir_version():
    model = emit(matrix_transform(np.eye(4)))
    assert model.ir_version == IR_VERSION
    assert [(o.domain, o.version) for o in model.opset_import] == [("", OPSET)]


def test_the_emitted_graph_declares_its_precision(config, config_uri):
    resolved = resolve_colorspaces(config, *PAIR, uri=config_uri)
    model = compile_processor(resolved)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata[f"{METADATA_PREFIX}precision"] == PRECISION == "float32"


def test_the_emitted_graph_carries_the_resolved_provenance(config, config_uri):
    resolved = resolve_colorspaces(config, *PAIR, uri=config_uri)
    model = compile_processor(resolved)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert resolved.metadata.items() <= metadata.items()


def test_an_identity_matrix_leaves_the_lattice_alone():
    samples = lattice()
    got = run_graph(emit(matrix_transform(np.eye(4))), samples)
    assert np.array_equal(got, samples)


def test_an_offset_reaches_the_bias():
    offset = (0.25, -0.5, 0.125, 0.0)
    samples = np.array([[[[0.1]], [[0.2]], [[0.3]]]], dtype=np.float32)
    got = run_graph(emit(matrix_transform(np.eye(4), offset)), samples)
    assert got.ravel() == pytest.approx([0.35, -0.3, 0.425], abs=1e-6)


def test_an_inverse_matrix_is_inverted_before_emission(config):
    matrix = [
        [0.6, 0.1, 0.3, 0.0],
        [0.2, 0.7, 0.1, 0.0],
        [0.1, 0.1, 0.8, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    offset = (0.01, 0.02, 0.03, 0.0)
    transform = matrix_transform(matrix, offset, inverse=True)

    samples = lattice()
    want = cpu_reference(config.getProcessor(transform), samples)
    got = run_graph(emit(transform), samples)
    assert compare(want, got, samples).ok


def test_the_pair_under_test_is_matrix_only(config, config_uri):
    resolved = resolve_colorspaces(config, *PAIR, uri=config_uri)
    assert op_names(resolved.processor) == ["Matrix"]


def test_the_pair_under_test_verifies(config, config_uri):
    resolved = resolve_colorspaces(config, *PAIR, uri=config_uri)
    result = verify(resolved)
    assert result.ok, str(result)
    assert result.compared > 0
    assert result.failures == 0


def test_a_swapped_row_fails_verification_diagnosably(config, config_uri):
    """The harness exists to catch a misreading; prove it catches one."""
    resolved = resolve_colorspaces(config, *PAIR, uri=config_uri)
    transform = resolved.processor.createGroupTransform()[0]
    matrix = np.array(transform.getMatrix(), dtype=np.float64).reshape(4, 4)
    matrix[[0, 1]] = matrix[[1, 0]]

    samples = lattice(resolved.processor)
    want = cpu_reference(resolved.processor, samples)
    got = run_graph(emit(matrix_transform(matrix)), samples)
    result = compare(want, got, samples)

    assert not result.ok
    report = str(result)
    assert "channel" in report
    assert "want" in report and "got" in report


def test_an_unimplemented_op_is_refused_by_name(config, config_uri):
    resolved = resolve_colorspaces(
        config, "ACES2065-1", "Rec.2100-HLG - Display", uri=config_uri
    )
    with pytest.raises(UnsupportedOpError, match="FixedFunction"):
        compile_processor(resolved)


def test_the_pinned_config_holds_the_counted_matrix_only_transforms(matrix_only):
    assert len(matrix_only) == MATRIX_ONLY


def test_every_matrix_only_transform_verifies(matrix_only, check):
    failures = [
        f"{label}: {result}"
        for label, processor in matrix_only
        if not (result := check(processor, label)).ok
    ]
    assert not failures, "\n".join(failures)
