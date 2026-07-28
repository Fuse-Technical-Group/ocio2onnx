"""``Lut1D`` emits a gather and a lerp over a uniform table (§spec:op-emission).

Forward and uniform-domain only. A half domain indexes by a float16 bit
pattern rather than by a coordinate, and an inverse op runs the table
backwards; both are refused here and lifted by their own workstreams, so a
transform carrying one is an answer rather than a crash.

The index is OCIO's: ``clip(x, 0, 1) * (length - 1)``, floored to a slot and
interpolated across it. OCIO's CPU processor interpolates rather than snapping
to the nearer slot, so a sub-slot input is what separates this emitter from a
nearest-neighbour lookup, and one test drives it there deliberately.

The upper index is ``i0 + (1 if frac > 0 else 0)`` rather than
``min(i0 + 1, length - 1)``. At ``x >= 1`` the index is the last slot and
``i0 + 1`` reads past the end; the ``frac > 0`` form keeps the top of the
table in range without a second clamp.
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest
from onnx import numpy_helper

from ocio2onnx import emitters
from ocio2onnx.builder import INPUT, GraphBuilder
from ocio2onnx.emitters import UnsupportedOpError
from ocio2onnx.oracle import cpu_reference, lattice, run_graph

LUT1D = "Lut1DTransform"
REFERENCE = "ACES2065-1"

#: The three color spaces whose forward direction carries a uniform table.
#: Their inverse direction carries the same table backwards and is refused.
UNIFORM_SPACES = (
    "ACEScc",
    "CanonLog2 CinemaGamut D55",
    "CanonLog3 CinemaGamut D55",
)

#: Measured across the pinned config (§spec:op-coverage).
HALF_DOMAIN_OPS = 34
UNIFORM_OPS = 6
FORWARD_OPS = 31
INVERSE_OPS = 9
HALF_DOMAIN_LENGTH = 65536
UNIFORM_LENGTH = 4096

#: Short enough to read in a failure, long enough that a slot is not the whole
#: domain.
SYNTHETIC_LENGTH = 32


def synthetic(length=SYNTHETIC_LENGTH, curves=None, **kwargs):
    """A bare ``Lut1DTransform`` carrying one curve per channel.

    Built without an interpolation, which is what OCIO reports for every 1D
    LUT it loads from a file.
    """
    curves = curves or (lambda x: x, lambda x: x, lambda x: x)
    lut = OCIO.Lut1DTransform(length=length, **kwargs)
    for i in range(length):
        x = i / (length - 1)
        lut.setValue(i, *(curve(x) for curve in curves))
    return lut


def per_channel(length=SYNTHETIC_LENGTH):
    """A LUT whose three channels disagree everywhere but at the ends."""
    return synthetic(
        length,
        curves=(lambda x: x**2, lambda x: x**0.5, lambda x: 0.25 + 0.5 * x),
    )


def emit(transform):
    """Run the emitter alone, for the refusals a processor would not reach."""
    return emitters.emit_lut1d(GraphBuilder(), transform, INPUT)


def table_of(model):
    """The one table initializer the graph carries.

    Exactly one: three coinciding channels share a table rather than emitting
    it three times.
    """
    (initializer,) = [
        entry
        for entry in model.graph.initializer
        if entry.name.startswith("lut1d_table")
    ]
    return numpy_helper.to_array(initializer)


def test_the_registry_carries_lut1d():
    assert LUT1D in emitters.REGISTRY


def test_the_pinned_configs_domains_are_what_the_specification_counted(config_ops):
    """An OCIO release that repacks these tables moves the split, and this
    fails rather than the coverage quietly changing."""
    domains = [
        (transform.getInputHalfDomain(), transform.getLength())
        for transform in config_ops(LUT1D)
    ]
    assert domains.count((True, HALF_DOMAIN_LENGTH)) == HALF_DOMAIN_OPS
    assert domains.count((False, UNIFORM_LENGTH)) == UNIFORM_OPS
    assert len(domains) == HALF_DOMAIN_OPS + UNIFORM_OPS


def test_the_pinned_configs_directions_are_what_the_specification_counted(config_ops):
    inverse = [
        str(transform.getDirection()).endswith("INVERSE")
        for transform in config_ops(LUT1D)
    ]
    assert inverse.count(True) == INVERSE_OPS
    assert inverse.count(False) == FORWARD_OPS


def test_every_lut1d_in_the_pinned_config_is_linear_and_unadjusted(config_ops):
    """The parameters the emitter refuses do not occur, so refusing them costs
    nothing here and keeps a config that does use one from being approximated."""
    for transform in config_ops(LUT1D):
        assert str(transform.getInterpolation()).endswith("INTERP_LINEAR")
        assert str(transform.getHueAdjust()).endswith("HUE_NONE")
        assert not transform.getOutputRawHalfs()


def test_every_lut1d_in_the_pinned_config_shares_one_curve_across_channels(config_ops):
    """Measured rather than assumed: OCIO reports a triple per entry, so a
    per-channel table is a shape another config can take."""
    for transform in config_ops(LUT1D):
        table = emitters.lut1d_table(transform)
        assert table.shape == (transform.getLength(), 3)
        assert (table == table[:, :1]).all()


@pytest.mark.parametrize("space", UNIFORM_SPACES)
def test_a_transform_carrying_a_uniform_forward_lut_verifies(space, config, check):
    processor = config.getProcessor(space, REFERENCE)
    result = check(processor, f"{space} -> {REFERENCE}")
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize("space", UNIFORM_SPACES)
def test_the_uniform_lut_op_alone_verifies(space, op_in, check_transform):
    result = check_transform(op_in(LUT1D, space, REFERENCE))
    assert result.ok, str(result)


def test_the_graph_interpolates_across_a_slot_rather_than_snapping_to_one(
    config, compile_bare, op_in, row
):
    """Nearest-neighbour would return a table entry. Half way between two
    entries the reference returns neither, and so must the graph."""
    transform = op_in(LUT1D, UNIFORM_SPACES[0], REFERENCE)
    table = emitters.lut1d_table(transform)
    slot = UNIFORM_LENGTH // 2
    low, high = float(table[slot, 0]), float(table[slot + 1, 0])
    assert low < high

    processor = config.getProcessor(transform)
    samples = row((slot + 0.5) / (UNIFORM_LENGTH - 1))
    want = float(cpu_reference(processor, samples).ravel()[0])
    got = float(run_graph(compile_bare(processor), samples).ravel()[0])

    assert low < got < high
    assert got == pytest.approx(want, rel=1e-5)
    assert got == pytest.approx(0.5 * (low + high), rel=1e-3)


def test_the_top_of_the_table_is_not_read_past(config, compile_bare, op_in, row):
    """At and above the domain's top the index is the last slot and the
    fraction is zero, so the upper index must stay on the last slot."""
    transform = op_in(LUT1D, UNIFORM_SPACES[0], REFERENCE)
    last = float(emitters.lut1d_table(transform)[-1, 0])
    processor = config.getProcessor(transform)
    samples = row(1.0, 2.0, 65504.0)
    got = run_graph(compile_bare(processor), samples).ravel()[:3]
    assert got == pytest.approx([last] * 3, rel=1e-6)


def test_a_per_channel_table_verifies(check_transform):
    """The only cover for the per-channel path: no op in the pinned config
    takes it."""
    result = check_transform(per_channel())
    assert result.ok, str(result)


def test_a_per_channel_table_is_read_per_channel(config, compile_bare, row):
    """Three different curves must give three different answers to one input,
    which a shared table cannot."""
    processor = config.getProcessor(per_channel())
    samples = row(0.36)
    got = run_graph(compile_bare(processor), samples).ravel()[:3]
    assert got == pytest.approx(cpu_reference(processor, samples).ravel()[:3], rel=1e-5)
    assert len(set(got.tolist())) == 3


def test_coinciding_channels_share_one_table(config, compile_bare):
    lut = synthetic(curves=(lambda x: x**2,) * 3)
    assert table_of(compile_bare(config.getProcessor(lut))).shape == (SYNTHETIC_LENGTH,)


def test_disagreeing_channels_flatten_into_one_channel_major_table(
    config, compile_bare
):
    """Flat rather than three tables: one ``Gather`` serves both paths, with a
    per-channel base offset added to the index."""
    lut = per_channel()
    flat = table_of(compile_bare(config.getProcessor(lut)))
    assert flat.shape == (3 * SYNTHETIC_LENGTH,)
    assert flat[:SYNTHETIC_LENGTH] == pytest.approx(
        emitters.lut1d_table(lut)[:, 0], rel=1e-6
    )


def test_a_default_interpolation_is_emitted_rather_than_refused(config):
    """Every 1D LUT OCIO loads from a file reports ``INTERP_DEFAULT``, which
    OCIO resolves to linear. Refusing it would refuse them all."""
    lut = per_channel()
    assert str(lut.getInterpolation()).endswith("INTERP_DEFAULT")
    assert emit(lut)


def test_a_half_domain_lut_is_refused(config_ops):
    """The 34 half-domain ops in the pinned config, and a bare one."""
    half = [
        transform for transform in config_ops(LUT1D) if transform.getInputHalfDomain()
    ]
    assert len(half) == HALF_DOMAIN_OPS
    for transform in (half[0], synthetic(HALF_DOMAIN_LENGTH, inputHalfDomain=True)):
        with pytest.raises(UnsupportedOpError, match="half domain"):
            emit(transform)


def test_an_inverse_lut_is_refused(op_in):
    """Both a bare inverse op and the one the config supplies."""
    bare = synthetic()
    bare.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    for transform in (bare, op_in(LUT1D, REFERENCE, UNIFORM_SPACES[0])):
        with pytest.raises(UnsupportedOpError, match="inverse"):
            emit(transform)


@pytest.mark.parametrize("interpolation", ("NEAREST", "BEST"))
def test_an_interpolation_this_compiler_does_not_emit_is_refused(interpolation):
    lut = synthetic(interpolation=getattr(OCIO, f"INTERP_{interpolation}"))
    with pytest.raises(UnsupportedOpError, match=f"interpolation {interpolation}"):
        emit(lut)


def test_a_hue_adjust_is_refused():
    lut = synthetic()
    lut.setHueAdjust(OCIO.HUE_DW3)
    with pytest.raises(UnsupportedOpError, match="hue adjust DW3"):
        emit(lut)


def test_raw_half_outputs_are_refused():
    lut = synthetic()
    lut.setOutputRawHalfs(True)
    with pytest.raises(UnsupportedOpError, match="raw half"):
        emit(lut)


def test_the_declared_breakpoints_are_the_domain_edges():
    assert emitters.breakpoints(synthetic()) == [0.0, 1.0]


def test_a_uniform_lut_widens_the_lattice_at_its_domain_edges(config, op_in):
    processor = config.getProcessor(op_in(LUT1D, UNIFORM_SPACES[0], REFERENCE))
    values = lattice(processor)[0, 0, 0, :]
    for edge in (0.0, 1.0):
        assert (values < np.float32(edge)).any()
        assert (values > np.float32(edge)).any()
        assert (values == np.float32(edge)).any()
