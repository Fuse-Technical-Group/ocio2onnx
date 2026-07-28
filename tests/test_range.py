"""``Range`` emits a clamp and an affine (§spec:op-emission).

Every ``Range`` in the pinned config sits in a transform that also carries a
``Lut1D``, so none of the 111 closed-form transforms exercises this emitter.
The parameters here are lifted off real config ops rather than invented, and
each is verified as the processor OCIO builds from that one transform.
"""

import math

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import enumerate_transforms, op_names
from ocio2onnx.oracle import cpu_reference, lattice, run_graph

#: Bound sets measured across the pinned config: an identity clamp, the two
#: halves of ACEScc's asymmetric pair, and a clamp with no upper bound.
IDENTITY_BOUNDS = (0.0, 1.0, 0.0, 1.0)
SCALING_BOUNDS = (-0.36, 1.5, 0.0, 1.0)
INVERSE_SCALING_BOUNDS = (0.0, 1.0, -0.36, 1.5)
HALF_OPEN_BOUNDS = (0.0, math.nan, 0.0, math.nan)

BOUND_SETS = (
    IDENTITY_BOUNDS,
    SCALING_BOUNDS,
    INVERSE_SCALING_BOUNDS,
    HALF_OPEN_BOUNDS,
    (0.0, 1024.0, 0.0, 1024.0),
    (0.0, 5020.76416015625, 0.0, 5020.76416015625),
)


def range_transform(bounds, *, inverse=False):
    """A bare ``RangeTransform`` from ``(min_in, max_in, min_out, max_out)``.

    OCIO reports an unset bound as NaN, which is how a half-open clamp arrives.
    """
    min_in, max_in, min_out, max_out = bounds
    transform = OCIO.RangeTransform()
    transform.setMinInValue(min_in)
    transform.setMaxInValue(max_in)
    transform.setMinOutValue(min_out)
    transform.setMaxOutValue(max_out)
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def row(*values):
    """A lattice-shaped sample carrying ``values`` in every channel."""
    return np.array([[[list(values)]] * 3], dtype=np.float32)


@pytest.fixture(scope="module")
def config_range_ops(config):
    """Every distinct ``Range`` op the pinned config carries."""
    seen = {}
    for _, processor in enumerate_transforms(config):
        for transform in processor.createGroupTransform():
            if type(transform).__name__ != "RangeTransform":
                continue
            key = (
                transform.getMinInValue(),
                transform.getMaxInValue(),
                transform.getMinOutValue(),
                transform.getMaxOutValue(),
            )
            seen.setdefault(repr(key), transform)
    return list(seen.values())


def test_the_registry_carries_range():
    assert "RangeTransform" in emitters.REGISTRY


def test_range_arrives_forward_and_clamping_throughout_the_config(config_range_ops):
    """OCIO folds an inverse ``Range``, and the pinned config uses one style."""
    assert config_range_ops
    for transform in config_range_ops:
        assert not str(transform.getDirection()).endswith("INVERSE")
        assert str(transform.getStyle()).endswith("RANGE_CLAMP")


def test_a_non_clamping_range_never_reaches_the_emitter(config):
    """OCIO rewrites ``RANGE_NO_CLAMP`` into a ``Matrix``, so the emitter has
    no non-clamping branch to get wrong."""
    transform = range_transform(SCALING_BOUNDS)
    transform.setStyle(OCIO.RANGE_NO_CLAMP)
    processor = config.getProcessor(transform)
    assert op_names(processor) == ["Matrix"]


def test_every_range_in_the_pinned_config_verifies(config_range_ops, check_transform):
    failures = [
        str(result)
        for transform in config_range_ops
        if not (result := check_transform(transform)).ok
    ]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("bounds", BOUND_SETS)
def test_a_forward_range_verifies(bounds, check_transform):
    result = check_transform(range_transform(bounds))
    assert result.ok, str(result)


@pytest.mark.parametrize("bounds", BOUND_SETS)
def test_an_inverse_range_verifies(bounds, check_transform):
    """``Range`` never arrives inverse in the pinned config; emit it anyway."""
    result = check_transform(range_transform(bounds, inverse=True))
    assert result.ok, str(result)


def test_an_inverse_range_swaps_the_bound_pairs():
    assert emitters.range_bounds(range_transform(SCALING_BOUNDS)) == SCALING_BOUNDS
    assert (
        emitters.range_bounds(range_transform(SCALING_BOUNDS, inverse=True))
        == INVERSE_SCALING_BOUNDS
    )


def test_the_declared_breakpoints_are_the_clamp_bounds():
    assert emitters.breakpoints(range_transform(SCALING_BOUNDS)) == [-0.36, 1.5]
    assert emitters.breakpoints(range_transform(SCALING_BOUNDS, inverse=True)) == [
        0.0,
        1.0,
    ]


def test_an_unset_bound_declares_no_breakpoint():
    assert emitters.breakpoints(range_transform(HALF_OPEN_BOUNDS)) == [0.0]


def test_the_declared_breakpoints_are_where_the_clamp_engages(config, compile_bare):
    """Either side of a bound lands on a different branch: inside the interval
    the graph is affine, outside it is constant."""
    transform = range_transform(SCALING_BOUNDS)
    low, high = emitters.breakpoints(transform)
    samples = row(low - 0.05, low + 0.05, high - 0.05, high + 0.05)
    got = run_graph(compile_bare(config.getProcessor(transform)), samples).ravel()[:4]

    assert got[0] == pytest.approx(0.0, abs=1e-6)
    assert got[1] > got[0]
    assert got[3] == pytest.approx(1.0, abs=1e-6)
    assert got[2] < got[3]


def test_an_identity_affine_is_not_emitted(config, compile_bare):
    """The clamp is the whole op where the two intervals coincide."""
    model = compile_bare(config.getProcessor(range_transform(IDENTITY_BOUNDS)))
    assert [node.op_type for node in model.graph.node] == ["Clip", "Identity"]


def test_a_scaling_range_emits_the_affine(config, compile_bare):
    model = compile_bare(config.getProcessor(range_transform(SCALING_BOUNDS)))
    assert [node.op_type for node in model.graph.node] == [
        "Clip",
        "Mul",
        "Add",
        "Identity",
    ]


def test_a_half_open_range_clamps_only_below(config, compile_bare):
    processor = config.getProcessor(range_transform(HALF_OPEN_BOUNDS))
    samples = row(-1.0, 0.5, 1e6)
    want = cpu_reference(processor, samples)
    got = run_graph(compile_bare(processor), samples)

    assert got == pytest.approx(want, abs=1e-6)
    assert got.ravel()[0] == pytest.approx(0.0)
    assert got.ravel()[2] == pytest.approx(1e6)


def test_a_range_widens_the_lattice_at_its_bounds(config):
    processor = config.getProcessor(range_transform(SCALING_BOUNDS))
    values = lattice(processor)[0, 0, 0, :]
    assert (values == np.float32(-0.36)).any()
    assert (values == np.float32(1.5)).any()
