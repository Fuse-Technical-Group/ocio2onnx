"""``Lut1D`` emits a gather and a lerp over a table (§spec:op-emission).

Forward only. An inverse op runs the table backwards, is refused here, and is
lifted by its own workstream, so a transform carrying one is an answer rather
than a crash.

Two domains reach the same gather, and only the index separates them. A
uniform table's index is OCIO's ``clip(x, 0, 1) * (length - 1)``. A
half-domain table's is the bit pattern of the input rounded to float16, which
standard ONNX cannot reinterpret and so reconstructs in float arithmetic the
way OCIO's own shaders reconstruct it: an exponent from ``floor(log2)``, a
mantissa fraction, a linear case below 2**-14 where a half is denormal, and a
sign offset applied to the slot rather than to the position inside it.

OCIO's CPU processor interpolates across slots rather than snapping to the
nearer one, so a sub-slot input is what separates this emitter from a
nearest-neighbour lookup, and one test per domain drives it there deliberately.

The upper index is ``i0 + (1 if frac > 0 else 0)`` rather than
``min(i0 + 1, length - 1)``. At ``x >= 1`` the uniform index is the last slot
and ``i0 + 1`` reads past the end; the ``frac > 0`` form keeps the top of the
table in range without a second clamp.
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest
from onnx import numpy_helper

from ocio2onnx import emitters
from ocio2onnx.builder import INPUT, GraphBuilder
from ocio2onnx.emitters import UnsupportedOpError
from ocio2onnx.oracle import compare, cpu_reference, lattice, run_graph

LUT1D = "Lut1DTransform"
REFERENCE = "ACES2065-1"

#: The three color spaces whose forward direction carries a uniform table.
#: Their inverse direction carries the same table backwards and is refused.
UNIFORM_SPACES = (
    "ACEScc",
    "CanonLog2 CinemaGamut D55",
    "CanonLog3 CinemaGamut D55",
)

#: Five pairs whose forward direction carries a half-domain table: two display
#: encodings, one camera log, and the two ADX densities.
HALF_DOMAIN_PAIRS = (
    (REFERENCE, "Rec.2100-PQ - Display"),
    (REFERENCE, "ST2084-P3-D65 - Display"),
    ("Apple Log", REFERENCE),
    ("ADX10", REFERENCE),
    ("ADX16", REFERENCE),
)

#: Measured across the pinned config (§spec:op-coverage).
HALF_DOMAIN_OPS = 34
HALF_DOMAIN_FORWARD_OPS = 28
HALF_DOMAIN_INVERSE_OPS = 6
UNIFORM_OPS = 6
FORWARD_OPS = 31
INVERSE_OPS = 9
HALF_DOMAIN_LENGTH = 65536
UNIFORM_LENGTH = 4096

#: Every half value in bit-pattern order, which is the order a half-domain
#: table is indexed in.
HALF_SLOTS = np.arange(HALF_DOMAIN_LENGTH, dtype=np.uint16).view(np.float16)

#: The last slot a finite input reaches, at 65504. Above it the bit patterns
#: are infinity and NaN.
LAST_FINITE_SLOT = 31743

#: The first normal slot, at 2**-14. Below it a half is denormal and its slot
#: is linear in the magnitude rather than a binade plus a mantissa.
FIRST_NORMAL_SLOT = 1024

#: What the sign bit adds to a slot index.
SIGN_SLOTS = 32768

#: Enough slots to walk every binade without sampling all 65536 of them, and
#: coprime with 1024 so the walk does not land on the same mantissa each time.
SLOT_STRIDE = 331

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


def between_slots(slots, fractions=(0.25, 0.5, 0.75)):
    """Inputs strictly inside the slot each index opens.

    A half-domain table is indexed by a rounded float16, so every input that
    is not itself a half falls between two entries. That is the ordinary case
    rather than the exotic one, and it is where a nearest-neighbour lookup
    parts company with the reference.
    """
    slots = np.asarray(slots)
    low = HALF_SLOTS[slots].astype(np.float32)
    high = HALF_SLOTS[slots + 1].astype(np.float32)
    return np.concatenate([low + np.float32(f) * (high - low) for f in fractions])


def walk_slots(first, last, offset=0):
    """Slot indices striding from ``first`` to ``last``, in one half of the
    table."""
    return np.arange(first, last, SLOT_STRIDE) + offset


@pytest.fixture(scope="session")
def half_domain_lut(op_in):
    """The half-domain table ``Rec.2100-PQ - Display`` is implemented as.

    Odd-symmetric about zero, so an input that reads the wrong half of the
    table comes back with the wrong sign rather than merely the wrong value.
    """
    return op_in(LUT1D, *HALF_DOMAIN_PAIRS[0])


@pytest.fixture
def check_at(config, compile_bare, row):
    """Hold one transform's graph against the CPU processor at chosen inputs.

    The oracle's lattice is a sweep. These tests need particular inputs — a
    position inside a half slot, a denormal — that no sweep lands on.
    """

    def check_at(transform, values):
        processor = config.getProcessor(transform)
        samples = row(*np.asarray(values, dtype=np.float32).tolist())
        return compare(
            cpu_reference(processor, samples),
            run_graph(compile_bare(processor), samples),
            samples,
        )

    return check_at


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


def test_the_half_domain_ops_split_by_direction_as_the_coverage_claims(config_ops):
    """The half-domain workstream reaches the forward ops alone, so the split
    is what it moves rather than the 34."""
    inverse = [
        str(transform.getDirection()).endswith("INVERSE")
        for transform in config_ops(LUT1D)
        if transform.getInputHalfDomain()
    ]
    assert inverse.count(False) == HALF_DOMAIN_FORWARD_OPS
    assert inverse.count(True) == HALF_DOMAIN_INVERSE_OPS


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


@pytest.mark.parametrize(("src", "dst"), HALF_DOMAIN_PAIRS)
def test_a_transform_carrying_a_half_domain_forward_lut_verifies(
    src, dst, config, check
):
    result = check(config.getProcessor(src, dst), f"{src} -> {dst}")
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(("src", "dst"), HALF_DOMAIN_PAIRS)
def test_the_half_domain_lut_op_alone_verifies(src, dst, op_in, check_transform):
    result = check_transform(op_in(LUT1D, src, dst))
    assert result.ok, str(result)


def test_an_input_between_two_half_slots_verifies_against_the_cpu_processor(
    half_domain_lut, check_at
):
    """The acceptance criterion for this domain, and the reason it is its own
    test: a nearest-neighbour lookup agrees with the reference at every half
    and disagrees everywhere else, so a sweep that happened to land on halves
    would pass one."""
    result = check_at(
        half_domain_lut,
        between_slots(walk_slots(FIRST_NORMAL_SLOT, LAST_FINITE_SLOT)),
    )
    assert result.ok, str(result)
    assert result.compared > 0


def test_the_graph_interpolates_across_a_half_slot_rather_than_snapping_to_one(
    half_domain_lut, config, compile_bare, row
):
    """The one assertion that pins the answer between two entries rather than
    at one of them. Half 14336 is 0.5, and its neighbour 0.500488281."""
    table = emitters.lut1d_table(half_domain_lut)
    low, high = float(table[14336, 0]), float(table[14337, 0])
    assert low < high

    processor = config.getProcessor(half_domain_lut)
    samples = row(0.5 * (float(HALF_SLOTS[14336]) + float(HALF_SLOTS[14337])))
    want = float(cpu_reference(processor, samples).ravel()[0])
    got = float(run_graph(compile_bare(processor), samples).ravel()[0])

    assert low < got < high
    assert got == pytest.approx(want, rel=1e-5)
    assert got == pytest.approx(0.5 * (low + high), rel=1e-6)


def test_the_denormal_branch_verifies(half_domain_lut, check_at):
    """Below 2**-14 a half's exponent field is zero and its slot is linear in
    the magnitude. The oracle's near-zero samples reach the first few slots;
    these reach the rest of the branch and both sides of its join."""
    result = check_at(
        half_domain_lut,
        np.concatenate(
            [
                between_slots(walk_slots(0, FIRST_NORMAL_SLOT)),
                np.array(
                    [1e-9, 1.49e-8, 2.98e-8, 2.0**-15, 2.0**-14, 2.0**-13],
                    dtype=np.float32,
                ),
            ]
        ),
    )
    assert result.ok, str(result)


def test_a_negative_input_verifies(half_domain_lut, check_at):
    """The sign moves the slot, not the position inside it, so a sign read
    wrong lands in the other half of the table."""
    result = check_at(
        half_domain_lut,
        between_slots(walk_slots(0, LAST_FINITE_SLOT, offset=SIGN_SLOTS)),
    )
    assert result.ok, str(result)


def test_negative_zero_reads_the_negative_half_of_the_table(
    half_domain_lut, config, compile_bare, row
):
    """OCIO reads the sign bit rather than comparing against zero, and the two
    zeros index different entries. ``-0.0 < 0.0`` is false, so a comparison
    returns the positive entry — the right magnitude with the wrong sign, and
    small enough here that the oracle's absolute tolerance would admit it."""
    table = emitters.lut1d_table(half_domain_lut)
    assert float(table[SIGN_SLOTS, 0]) == -float(table[0, 0]) != 0.0

    processor = config.getProcessor(half_domain_lut)
    samples = row(-0.0)
    assert float(cpu_reference(processor, samples).ravel()[0]) < 0.0
    got = float(run_graph(compile_bare(processor), samples).ravel()[0])
    assert got == pytest.approx(float(table[SIGN_SLOTS, 0]), rel=1e-6)


def test_a_table_that_overflows_float32_lerps_to_the_finite_limit_not_to_nan(
    config, compile_bare, op_in, row
):
    """``Apple Log -> ref`` decodes to more than float32 holds, and its table
    reports infinity from slot 18898 up. OCIO's CPU processor renders those
    entries as the finite limit, so the graph must too: a lerp between two
    infinite entries is ``inf - inf``, and the whole top of the domain would
    come back NaN."""
    lut = op_in(LUT1D, "Apple Log", REFERENCE)
    assert np.isinf(lut.getValue(LAST_FINITE_SLOT)[0])

    processor = config.getProcessor(lut)
    samples = row(11.64, 12.0, 100.0, 1000.0, 65504.0)
    got = run_graph(compile_bare(processor), samples).ravel()[:5]
    assert np.isfinite(got).all()
    assert got == pytest.approx(cpu_reference(processor, samples).ravel()[:5], rel=1e-6)


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


def test_a_forward_half_domain_lut_is_emitted_rather_than_refused(config_ops):
    """The refusal this workstream lifted. Every forward half-domain op in the
    pinned config emits, and so does a bare one."""
    half = [
        transform
        for transform in config_ops(LUT1D)
        if transform.getInputHalfDomain()
        and not str(transform.getDirection()).endswith("INVERSE")
    ]
    assert len(half) == HALF_DOMAIN_FORWARD_OPS
    assert emit(half[0])
    assert emit(synthetic(HALF_DOMAIN_LENGTH, inputHalfDomain=True))


def test_an_inverse_lut_is_refused(op_in):
    """Both domains, in both the bare and the configured form. Direction is
    the one shape left, and its own workstream."""
    bare = synthetic()
    bare.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    transforms = (
        bare,
        op_in(LUT1D, REFERENCE, UNIFORM_SPACES[0]),
        op_in(LUT1D, REFERENCE, "Apple Log"),
    )
    for transform in transforms:
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


def test_a_uniform_lut_declares_its_domain_edges():
    assert emitters.breakpoints(synthetic()) == [0.0, 1.0]


def test_a_half_domain_lut_declares_the_branches_its_index_has():
    """Not the unit interval: a half domain does not clamp at 0 or 1, it spans
    the float16 line. Zero is where the sign offset switches and 2**-14 is
    where the denormal case joins the normal one."""
    lut = synthetic(HALF_DOMAIN_LENGTH, inputHalfDomain=True)
    assert emitters.breakpoints(lut) == [0.0, 2.0**-14, -(2.0**-14)]


def test_a_uniform_lut_widens_the_lattice_at_its_domain_edges(config, op_in):
    processor = config.getProcessor(op_in(LUT1D, UNIFORM_SPACES[0], REFERENCE))
    values = lattice(processor)[0, 0, 0, :]
    for edge in (0.0, 1.0):
        assert (values < np.float32(edge)).any()
        assert (values > np.float32(edge)).any()
        assert (values == np.float32(edge)).any()
