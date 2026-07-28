"""ASC CDL emits slope, offset, power, and saturation as graph inputs
(§spec:dynamic-properties).

OCIO declares no dynamic property for a CDL — its run time bakes the ten
numbers into the op. A CDL is nonetheless *the* grade knob, and the emitted
artifact is a graph rather than a table precisely so a grade can move without
a recompile, so the four parameters reach the graph live and the sweep below is
what says they are wired to the arithmetic OCIO runs.

Both styles are exercised in both directions. ``ASC`` clamps to the unit
interval three times over an inverse; ``NO_CLAMP`` clamps nowhere and instead
passes the negative half of the domain through the power untouched. The
inverse pre-inverts its four parameters, which for a live one is a reciprocal
in the graph rather than a constant folded at compile time.
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import OPTIMIZATION_FLAGS
from ocio2onnx.builder import parameters
from ocio2onnx.compiler import UnsupportedOpError
from ocio2onnx.oracle import run_graph

CDL = "CDL"

STYLES = (OCIO.CDL_ASC, OCIO.CDL_NO_CLAMP)

#: The grade the graph's four inputs default to. Every channel differs from
#: every other and none is the identity, so a transposed or dropped channel
#: cannot agree with the reference.
SLOPE = [1.1, 0.95, 0.8]
OFFSET = [0.02, 0.0, -0.03]
POWER = [1.2, 1.0, 0.85]
SATURATION = 1.25

#: Per-channel sweeps. A slope of zero collapses a channel, which is a grade a
#: colourist reaches for rather than an invalid request.
SLOPE_SWEEP = ([0.0, 0.5, 2.0], [1.0, 1.0, 1.0], [3.0, 0.25, 1.4])
OFFSET_SWEEP = ([-0.4, 0.0, 0.4], [0.0, 0.0, 0.0], [0.15, -0.15, 0.05])

#: Powers stay positive: the ASC specification bounds them there and OCIO
#: refuses the rest before a processor is built. The sweep passes through unity
#: per channel but never on all three at once, which is where OCIO's optimizer
#: stops running the op
#: (`test_a_unit_power_is_where_ocios_optimizer_changes_the_transform`).
POWER_SWEEP = ([0.4, 1.0, 2.5], [1.0, 0.9, 1.1], [2.2, 0.45, 1.6])

#: The power at which OCIO rewrites the op rather than running it.
UNIT_POWER = [1.0, 1.0, 1.0]

#: Saturation from fully desaturated through untouched to well past it.
SATURATION_SWEEP = (0.0, 0.5, 1.0, 2.4)

#: Graph input, OCIO setter, and the range each is swept over.
SWEEPS = (
    ("CDL_SLOPE", "setSlope", SLOPE_SWEEP),
    ("CDL_OFFSET", "setOffset", OFFSET_SWEEP),
    ("CDL_POWER", "setPower", POWER_SWEEP),
    ("CDL_SATURATION", "setSat", SATURATION_SWEEP),
)

INPUTS = [name for name, _, _ in SWEEPS]


def cdl(style, *, inverse=False, **overrides):
    """A bare ``CDLTransform`` at the grade above, minus overrides."""
    transform = OCIO.CDLTransform()
    transform.setStyle(style)
    values = {
        "Slope": SLOPE,
        "Offset": OFFSET,
        "Power": POWER,
        "Sat": SATURATION,
        **overrides,
    }
    for name, value in values.items():
        getattr(transform, f"set{name}")(value)
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def test_the_registry_carries_cdl():
    assert CDL in emitters.REGISTRY


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_the_graph_declares_one_input_per_cdl_parameter(
    style, inverse, config, compile_bare
):
    """The defaults are the forward grade OCIO reports, whichever direction the
    op runs: the knob is the CDL's slope, not the reciprocal the inverse
    happens to multiply by."""
    defaults = parameters(
        compile_bare(config.getProcessor(cdl(style, inverse=inverse)))
    )
    assert list(defaults) == INPUTS
    assert defaults["CDL_SLOPE"] == pytest.approx(SLOPE)
    assert defaults["CDL_OFFSET"] == pytest.approx(OFFSET)
    assert defaults["CDL_POWER"] == pytest.approx(POWER)
    assert defaults["CDL_SATURATION"] == pytest.approx([SATURATION])


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_a_bare_cdl_verifies_at_its_defaults(style, inverse, check_transform):
    result = check_transform(cdl(style, inverse=inverse))
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(("name", "setter", "sweep"), SWEEPS)
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_one_graph_tracks_ocio_across_a_swept_parameter(
    name, setter, sweep, style, inverse, config, check, compile_bare
):
    """The acceptance test for a live parameter (§spec:dynamic-properties).

    One graph, compiled at the defaults, held against the processor OCIO builds
    at each swept value — the grade moves without a recompile.
    """
    model = compile_bare(config.getProcessor(cdl(style, inverse=inverse)))

    for value in sweep:
        reference = cdl(style, inverse=inverse)
        getattr(reference, setter)(value)
        result = check(
            config.getProcessor(reference),
            model=model,
            parameters={name: np.atleast_1d(value)},
        )
        assert result.ok, f"{name}={value}: {result}"
        assert result.compared > 0


@pytest.mark.parametrize("style", STYLES)
def test_a_swept_parameter_moves_the_picture(style, config, compile_bare, row):
    """A knob wired to nothing agrees with OCIO at the default and is ignored
    everywhere else, which every agreement test above would pass."""
    model = compile_bare(config.getProcessor(cdl(style)))
    samples = row(0.18, 0.5)
    dim = run_graph(model, samples, {"CDL_SLOPE": [0.25, 0.25, 0.25]})
    lifted = run_graph(model, samples, {"CDL_SLOPE": [0.9, 0.9, 0.9]})
    assert (dim < lifted).all()


def test_a_collapsed_channel_stays_invertible(config, check, compile_bare):
    """OCIO floors a slope, a power, and a saturation at 0.01 before
    reciprocating them, so a channel a colourist crushes to nothing inverts to
    100 rather than to an infinity. A static emitter folds that floor at
    compile time; a live parameter carries it into the graph."""
    model = compile_bare(config.getProcessor(cdl(OCIO.CDL_NO_CLAMP, inverse=True)))
    result = check(
        config.getProcessor(cdl(OCIO.CDL_NO_CLAMP, inverse=True, Sat=0.0)),
        model=model,
        parameters={"CDL_SATURATION": [0.0]},
    )
    assert result.ok, str(result)
    assert result.compared > 0


def test_a_unit_power_is_where_ocios_optimizer_changes_the_transform(
    config, compile_bare
):
    """Where the power sweep stops short of the identity, and why.

    At a unit power OCIO's optimizer replaces the clamping CDL with ``Range``
    and ``Matrix`` ops that drop its output clamp: the op answers 1 for an
    input of 1 on a lifted channel and the rewrite answers 1.2875. The graph
    emits the op, so it agrees with the first. Sweeping through the identity
    would be measuring the rewrite instead.
    """
    processor = config.getProcessor(cdl(OCIO.CDL_ASC, inverse=True, Power=UNIT_POWER))

    def reference(flags):
        pixels = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        processor.getOptimizedCPUProcessor(flags).applyRGB(pixels)
        return pixels.ravel()

    op = reference(OCIO.OPTIMIZATION_NONE)
    rewrite = reference(OPTIMIZATION_FLAGS)
    assert op[2] == pytest.approx(1.0, abs=1e-6)
    assert rewrite[2] == pytest.approx(1.2875, abs=1e-6)

    model = compile_bare(config.getProcessor(cdl(OCIO.CDL_ASC, inverse=True)))
    graph = run_graph(
        model,
        np.ones((1, 3, 1, 1), dtype=np.float32),
        {"CDL_POWER": UNIT_POWER},
    )
    assert graph.ravel() == pytest.approx(op, abs=1e-6)


def test_saturation_reads_the_luma_weights_off_the_transform(config, compile_bare, row):
    """A fully desaturated pixel is its own luminance in all three channels,
    weighted as OCIO reports rather than as this module assumes."""
    transform = cdl(OCIO.CDL_NO_CLAMP, Slope=[1.0] * 3, Offset=[0.0] * 3)
    transform.setPower([1.0] * 3)
    weights = np.asarray(transform.getSatLumaCoefs())
    model = compile_bare(config.getProcessor(transform))

    channels = [0.2, 0.5, 0.9]
    pixel = np.array([[[[value]] for value in channels]], dtype=np.float32)
    got = run_graph(model, pixel, {"CDL_SATURATION": [0.0]})
    assert got.ravel() == pytest.approx([float(weights @ channels)] * 3, rel=1e-6)


def test_two_ops_carry_a_knob_each(config, compile_bare):
    """Two graded ops in one transform hold different grades, so a shared input
    would have to discard one of them."""
    group = OCIO.GroupTransform()
    for saturation in (0.5, 1.5):
        group.appendTransform(cdl(OCIO.CDL_NO_CLAMP, Sat=saturation))
    defaults = parameters(compile_bare(config.getProcessor(group)))

    assert defaults["CDL_SATURATION"] == pytest.approx([0.5])
    assert defaults["CDL_SATURATION_2"] == pytest.approx([1.5])


#: Where the graded value crosses 0 and 1 for the grade above, stated rather
#: than re-derived: a sign error in ``cdl_breakpoints`` would be reproduced by
#: an expectation that runs the same formula.
SIGN_BRANCH = [-0.0181818, 0.0, 0.0375]
UNIT_CLAMP = [0.8909091, 1.0526316, 1.2875]


def test_the_clamping_style_declares_the_unit_interval_as_its_breakpoints():
    """``ASC`` clamps to [0, 1], and forward that clamp sits after the slope and
    the offset, so its breakpoints are those bounds read back through them."""
    assert emitters.breakpoints(cdl(OCIO.CDL_ASC)) == pytest.approx(
        sorted(SIGN_BRANCH + UNIT_CLAMP), abs=1e-6
    )


def test_the_clamping_inverse_declares_the_unit_interval_itself():
    """Inverse, ``ASC``'s first clamp is on the input."""
    assert emitters.breakpoints(cdl(OCIO.CDL_ASC, inverse=True)) == [0.0, 1.0]


def test_the_unclamped_style_declares_where_its_sign_branch_sits():
    """``NO_CLAMP`` passes a negative value through the power untouched, so the
    branch is where the graded value crosses zero — and only there."""
    assert emitters.breakpoints(cdl(OCIO.CDL_NO_CLAMP)) == pytest.approx(
        SIGN_BRANCH, abs=1e-6
    )


def test_the_unclamped_inverse_states_its_branch_on_the_diagonal():
    """Inverse, the sign branch sits after the saturation, which crosses
    channels — so no input value places it exactly. On a neutral pixel the
    saturation is the identity and the branch is at zero, which is the same
    reading ``rec2100_surround_breakpoints`` takes."""
    assert emitters.breakpoints(cdl(OCIO.CDL_NO_CLAMP, inverse=True)) == [0.0]


def test_a_breakpoint_a_collapsed_channel_cannot_place_is_dropped():
    """A slope of zero puts the branch at infinity. The lattice takes finite
    values, so there is nothing to straddle and that channel offers nothing —
    while the two beside it still do."""
    transform = cdl(OCIO.CDL_NO_CLAMP, Slope=[0.0, 1.0, 2.0], Offset=[0.5, 0.25, -0.5])
    assert emitters.breakpoints(transform) == pytest.approx([-0.25, 0.25])


def test_a_style_this_compiler_has_no_path_for_is_refused_by_name(
    config, compile_bare, monkeypatch
):
    """Following ``_negative_style``: an unrecognised style is refused rather
    than approximated by the nearest one implemented."""
    monkeypatch.setattr(emitters, "CDL_STYLES", ("ASC",))
    with pytest.raises(UnsupportedOpError, match="NO_CLAMP"):
        compile_bare(config.getProcessor(cdl(OCIO.CDL_NO_CLAMP)))
