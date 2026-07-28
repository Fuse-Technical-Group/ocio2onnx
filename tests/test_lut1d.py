"""``Lut1D`` emits a gather and a lerp over a table (§spec:op-emission).

An inverse op runs the table backwards, which the graph never does: the table
is inverted once, at compile time, onto the half domain, and the graph reads
the result as an ordinary forward half-domain table. So there is one gather
for both directions, and the inversion's own decisions — the grid, and what a
flat stretch of the forward table inverts to — are what these tests pin.

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
from ocio2onnx.oracle import TOLERANCE, compare, cpu_reference, lattice, run_graph

LUT1D = "Lut1D"
REFERENCE = "ACES2065-1"

#: The three color spaces whose forward direction carries a uniform table.
#: Their inverse direction carries the same table backwards.
UNIFORM_SPACES = (
    "ACEScc",
    "CanonLog2 CinemaGamut D55",
    "CanonLog3 CinemaGamut D55",
)

#: The eight pairs whose ``Lut1D`` arrives inverse — the transforms this
#: workstream unblocks. The ninth inverse op in the config sits in
#: ``Rec.2100-HLG - Display -> ref``, which refuses on ``FixedFunction``
#: whatever this emitter does.
INVERSE_PAIRS = (
    ("Rec.2100-PQ - Display", REFERENCE),
    ("ST2084-P3-D65 - Display", REFERENCE),
    (REFERENCE, "ACEScc"),
    (REFERENCE, "ADX10"),
    (REFERENCE, "ADX16"),
    (REFERENCE, "Apple Log"),
    (REFERENCE, "CanonLog2 CinemaGamut D55"),
    (REFERENCE, "CanonLog3 CinemaGamut D55"),
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

#: Where a flat stretch of a forward table inverts to, read off OCIO's own
#: inverse rather than chosen. ``ref -> Apple Log`` decodes every input from
#: -65504 up to 0 as -0.0564109, and OCIO answers that value with 0 — the top
#: of the flat stretch, not its bottom. ``ref -> ADX10`` encodes everything
#: from 3.7597656 up as 4.816268, and OCIO answers that value with 3.7597656 —
#: the bottom of that one. Both are the endpoints of the interval the table is
#: invertible over, which is what the inversion has to clamp to.
FLAT_TAILS = (
    ((REFERENCE, "Apple Log"), -65504.0, 0.0),
    ((REFERENCE, "ADX10"), 65504.0, 3.7597656),
)


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


def is_inverse(transform):
    """Whether OCIO hands this op over to be run backwards.

    Spelled out here rather than imported: a test asserting a direction split
    with the implementation's own direction predicate would assert nothing.
    """
    return str(transform.getDirection()).endswith("INVERSE")


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


@pytest.fixture(scope="session")
def inverse_lut(op_in):
    """The inverse table ``Rec.2100-PQ - Display -> ref`` carries.

    The same odd-symmetric curve as ``half_domain_lut``, run backwards, so the
    two directions are testable against each other.
    """
    return op_in(LUT1D, *INVERSE_PAIRS[0])


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
    inverse = [is_inverse(transform) for transform in config_ops(LUT1D)]
    assert inverse.count(True) == INVERSE_OPS
    assert inverse.count(False) == FORWARD_OPS


def test_the_half_domain_ops_split_by_direction_as_the_coverage_claims(config_ops):
    """The half-domain workstream reaches the forward ops alone, so the split
    is what it moves rather than the 34."""
    inverse = [
        is_inverse(transform)
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
        if transform.getInputHalfDomain() and not is_inverse(transform)
    ]
    assert len(half) == HALF_DOMAIN_FORWARD_OPS
    assert emit(half[0])
    assert emit(synthetic(HALF_DOMAIN_LENGTH, inputHalfDomain=True))


def test_an_inverse_lut_is_emitted_rather_than_refused(config_ops):
    """The refusal this workstream lifted. Every inverse op in the pinned
    config emits, including the one inside a transform that refuses anyway, so
    the monotonicity guard is measured against all nine rather than eight."""
    inverse = [transform for transform in config_ops(LUT1D) if is_inverse(transform)]
    assert len(inverse) == INVERSE_OPS
    for transform in inverse:
        assert emit(transform)


@pytest.mark.parametrize(("src", "dst"), INVERSE_PAIRS)
def test_a_transform_carrying_an_inverse_lut_verifies(src, dst, config, check):
    """The eight transforms this workstream unblocks, end to end."""
    result = check(config.getProcessor(src, dst), f"{src} -> {dst}")
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(("src", "dst"), INVERSE_PAIRS)
def test_the_inverse_lut_op_alone_verifies(src, dst, op_in, check_transform):
    result = check_transform(op_in(LUT1D, src, dst))
    assert result.ok, str(result)


def test_an_input_between_two_half_slots_verifies_the_inverse_direction(
    inverse_lut, check_at
):
    """The section's acceptance criterion, in the direction this workstream
    added. An inverse is emitted as a half-domain table, so it meets the same
    sub-slot inputs the forward direction does and must interpolate across
    them the same way."""
    result = check_at(
        inverse_lut,
        between_slots(walk_slots(FIRST_NORMAL_SLOT, LAST_FINITE_SLOT)),
    )
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(
    ("src", "dst"), ((REFERENCE, "ACEScc"), (REFERENCE, "Rec.2100-PQ - Display"))
)
def test_the_inverse_undoes_the_forward(src, dst, config, compile_bare, row):
    """Compile both directions and compose them. Neither graph knows about the
    other, so a table inverted onto the wrong grid, or read off by a slot,
    comes back as a value that is not the input.

    Both round trips start scene-linear and stay under diffuse white, because
    that is where both encodings are invertible: ``ACEScc`` clamps its code
    value at 1 and the PQ display clamps its own, and a clamp does not undo.
    """
    values = [1e-3, 0.0078125, 0.18, 1.0]
    samples = row(*values)
    forward = run_graph(compile_bare(config.getProcessor(src, dst)), samples)
    back = run_graph(compile_bare(config.getProcessor(dst, src)), forward)
    assert back.ravel()[: len(values)] == pytest.approx(values, rel=1e-4, abs=2e-5)


@pytest.mark.parametrize(("pair", "value", "want"), FLAT_TAILS)
def test_a_flat_stretch_inverts_to_where_the_forward_table_leaves_it(
    pair, value, want, config, compile_bare, op_in, row
):
    """A flat run makes the inverse ambiguous over an interval, and the two
    ends of that interval are half a domain apart. OCIO answers with the end
    the curve leaves the run at — the invertible interval's edge — and the
    graph has to agree with it, not merely be inside the run."""
    transform = op_in(LUT1D, *pair)
    processor = config.getProcessor(transform)
    samples = row(value)
    assert float(cpu_reference(processor, samples).ravel()[0]) == pytest.approx(want)
    got = float(run_graph(compile_bare(processor), samples).ravel()[0])
    assert got == pytest.approx(want, rel=1e-4, abs=2e-5)


def test_an_inverse_is_sampled_onto_the_half_domain(config, compile_bare, op_in):
    """The grid, pinned. A uniform forward table inverts to 65536 entries, not
    back to its own 4096: the inversion grid is the half domain whichever
    domain the forward table had, because a half's spacing is geometric and so
    carries the same relative resolution in the toe as at the top
    (§spec:op-emission)."""
    for src, dst in (INVERSE_PAIRS[0], (REFERENCE, UNIFORM_SPACES[0])):
        transform = op_in(LUT1D, src, dst)
        assert emitters.lut1d_inverse_table(transform).shape == (
            HALF_DOMAIN_LENGTH,
            3,
        )
        model = compile_bare(config.getProcessor(transform))
        assert table_of(model).shape == (HALF_DOMAIN_LENGTH,)


def test_a_uniform_inversion_grid_would_not_hold_tolerance(op_in):
    """Why the grid is the half domain and not the obvious alternative.

    ``ACEScc``'s forward table spans [-5.7e-07, 96617.7]. A uniform grid over
    that range spends its samples above middle grey and leaves the whole toe
    to one interval, so resampling the inverse onto it loses the part of the
    picture a grade reaches for. Adding samples does not fix it: 65536 uniform
    samples are barely better than 4096, because the problem is the dynamic
    range rather than the count.

    Kept cheap and kept here so the rejected alternative stays rejected on
    evidence. §spec:op-emission records the whole measurement: over the
    lattices of all eight transforms carrying an inverse, a uniform grid
    misses 2267 of 10134 samples where the half domain misses none.
    """
    transform = op_in(LUT1D, REFERENCE, UNIFORM_SPACES[0])
    values = emitters.lut1d_table(transform)[:, 0].astype(np.float64)
    domain = np.linspace(0.0, 1.0, values.size)
    probes = np.array([1e-3, 0.0078125, 0.18, 1.0, 10.0])
    want = np.interp(probes, values, domain)

    for count in (UNIFORM_LENGTH, HALF_DOMAIN_LENGTH):
        grid = np.linspace(values[0], values[-1], count)
        got = np.interp(probes, grid, np.interp(grid, values, domain))
        assert (np.abs(got - want) > TOLERANCE.bound(want)).all()
        assert np.abs(got - want).max() > 0.3

    grid = HALF_SLOTS[:LAST_FINITE_SLOT].astype(np.float64)
    got = np.interp(probes, grid, np.interp(grid, values, domain))
    assert (np.abs(got - want) <= TOLERANCE.bound(want)).all()


def test_a_table_that_does_not_rise_is_refused():
    """A non-monotonic table has no inverse to read off, so it is refused
    rather than resolved to whichever branch a sort happened to keep."""
    lut = synthetic(curves=((lambda x: (x - 0.5) ** 2),) * 3)
    lut.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    with pytest.raises(UnsupportedOpError, match="does not rise"):
        emit(lut)


def test_the_monotonicity_guard_reads_the_pinned_configs_tables_as_rising(op_in):
    """``+0.0`` and ``-0.0`` compare equal, so a half-domain table holds two
    entries at domain zero and a naive difference reads the step between them
    as a fall. Both PQ tables would refuse on that, and both are fine."""
    for src, dst in INVERSE_PAIRS[:2]:
        assert emit(op_in(LUT1D, src, dst))


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


def test_an_inverse_uniform_lut_declares_the_branches_the_half_index_has():
    """It reaches the graph as a half-domain table whatever domain it was
    written over, so the lattice has to straddle the half index's branches
    rather than the unit interval the forward table used."""
    lut = synthetic()
    lut.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    assert emitters.breakpoints(lut) == [0.0, 2.0**-14, -(2.0**-14)]


def test_a_uniform_lut_widens_the_lattice_at_its_domain_edges(config, op_in):
    processor = config.getProcessor(op_in(LUT1D, UNIFORM_SPACES[0], REFERENCE))
    values = lattice(processor)[0, 0, 0, :]
    for edge in (0.0, 1.0):
        assert (values < np.float32(edge)).any()
        assert (values > np.float32(edge)).any()
        assert (values == np.float32(edge)).any()


def test_a_nan_input_reads_the_first_slot_rather_than_an_index_no_bound_covers(
    config, compile_bare, op_in, row
):
    """The index is cast to int64 and handed to ``Gather``, and
    ``Cast(NaN, INT64)`` is implementation-defined — ``INT64_MIN`` on x86,
    which no table bound covers. Left alone it aborts the whole inference on
    ONNX Runtime rather than returning a pixel, and an out-of-range ``Gather``
    is undefined on an executor that does not check.

    NaN is not exotic in the pixels this graph runs on, and it arises inside
    the graph too: an upstream ``Matrix`` overflowing to ``±inf`` yields
    ``inf - inf``. So this is asserted here rather than left to the lattice,
    which is finite throughout and cannot reach it.

    The first slot is where OCIO reads a NaN over a uniform table. Over a half
    domain OCIO reads the NaN's own float16 pattern, slot 32256, which holds
    the same value as slot 0 in every half-domain table in the pinned config.
    """
    for src, dst in (
        (UNIFORM_SPACES[0], REFERENCE),
        HALF_DOMAIN_PAIRS[0],
        INVERSE_PAIRS[0],
    ):
        lut = op_in(LUT1D, src, dst)
        processor = config.getProcessor(lut)
        samples = row(np.nan, np.inf, -np.inf)
        got = run_graph(compile_bare(processor), samples)
        assert np.isfinite(got).all(), f"{src} -> {dst} left the table"
        assert got.ravel()[0] == pytest.approx(
            emitters.lut1d_table(lut)[0, 0]
            if not is_inverse(lut)
            else emitters.lut1d_inverse_table(lut)[0, 0],
            rel=1e-6,
        )


def test_an_inverse_lut_with_no_finite_entry_is_refused(config):
    """A table of NaNs is reachable from a config file and has no curve under
    it to read backwards. Refused by name rather than raising an ``IndexError``
    out of the inversion's own arithmetic."""
    lut = synthetic(curves=(lambda x: np.nan,) * 3)
    lut.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    with pytest.raises(UnsupportedOpError, match="no finite entry"):
        emit(lut)
