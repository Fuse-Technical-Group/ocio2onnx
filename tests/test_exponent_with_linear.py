"""``ExponentWithLinear`` emits a monitor curve joined to a linear segment
(§spec:op-emission).

The join is C1-continuous, and its two constants are the whole risk in this
op: a plausible closed form for the linear slope is wrong by a factor of
around 50, and nothing but the oracle reports it. So the constants are tested
for the property that fixes them — continuity of value and of slope at the
break — rather than against numbers copied out of the implementation.
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import CHANNELS, enumerate_transforms, resolve_colorspaces
from ocio2onnx.compiler import UnsupportedOpError
from ocio2onnx.oracle import lattice, run_graph, verify

#: One closed-form transform per (direction, negative style) the config
#: carries. Forward is encoded to linear; inverse is linear to encoded.
MIRROR_FORWARD = ("sRGB - Display", "ACES2065-1")
MIRROR_INVERSE = ("ACES2065-1", "sRGB - Display")
LINEAR_FORWARD = ("sRGB Encoded Rec.709 (sRGB)", "ACES2065-1")
LINEAR_INVERSE = ("ACES2065-1", "sRGB Encoded Rec.709 (sRGB)")

PAIRS = (MIRROR_FORWARD, MIRROR_INVERSE, LINEAR_FORWARD, LINEAR_INVERSE)

STYLES = (OCIO.NEGATIVE_MIRROR, OCIO.NEGATIVE_LINEAR)

#: The two parameter sets the pinned config carries: sRGB, and the Rec.709
#: camera encoding.
SRGB = (2.4, 0.055)
REC709 = (2.2222222222222223, 0.099)


def moncurve_transform(gamma, offset, *, style=OCIO.NEGATIVE_MIRROR, inverse=False):
    """A bare ``ExponentWithLinearTransform``."""
    transform = OCIO.ExponentWithLinearTransform()
    transform.setGamma([gamma] * 3 + [1.0])
    transform.setOffset([offset] * 3 + [0.0])
    transform.setNegativeStyle(style)
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def row(*values):
    """A lattice-shaped sample carrying ``values`` in every channel."""
    return np.array([[[list(values)]] * 3], dtype=np.float32)


def moncurve_op(config, src, dst):
    """The ``ExponentWithLinear`` op inside one config transform."""
    ops = [
        t
        for t in config.getProcessor(src, dst).createGroupTransform()
        if type(t).__name__ == "ExponentWithLinearTransform"
    ]
    assert len(ops) == 1
    return ops[0]


def test_the_registry_carries_exponent_with_linear():
    assert "ExponentWithLinearTransform" in emitters.REGISTRY


@pytest.mark.parametrize("pair", PAIRS)
def test_a_config_transform_carrying_the_op_verifies(pair, config, config_uri):
    result = verify(resolve_colorspaces(config, *pair, uri=config_uri))
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(
    ("pair", "style"),
    (
        (MIRROR_FORWARD, "NEGATIVE_MIRROR"),
        (MIRROR_INVERSE, "NEGATIVE_MIRROR"),
        (LINEAR_FORWARD, "NEGATIVE_LINEAR"),
        (LINEAR_INVERSE, "NEGATIVE_LINEAR"),
    ),
)
def test_the_pairs_under_test_carry_the_styles_they_claim(pair, style, config):
    assert str(moncurve_op(config, *pair).getNegativeStyle()).endswith(style)


@pytest.mark.parametrize("parameters", (SRGB, REC709))
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_a_bare_monitor_curve_verifies(parameters, style, inverse, check_transform):
    transform = moncurve_transform(*parameters, style=style, inverse=inverse)
    result = check_transform(transform)
    assert result.ok, str(result)


def test_the_break_is_continuous_in_value():
    """The linear segment meets the curve at the break rather than stepping."""
    for gamma, offset in (SRGB, REC709):
        curve = emitters.moncurve(gamma, offset)
        power = ((curve.x_break + offset) / (1.0 + offset)) ** gamma
        assert curve.y_break == pytest.approx(power, rel=1e-12)


def test_the_break_is_continuous_in_slope():
    """C1, not merely C0: the linear slope is the curve's derivative at the
    break. Getting this wrong by a large factor is the failure this op is
    prone to, and it is invisible in the value alone."""
    for gamma, offset in (SRGB, REC709):
        curve = emitters.moncurve(gamma, offset)
        step = curve.x_break * 1e-7

        def power(x, gamma=gamma, offset=offset):
            return ((x + offset) / (1.0 + offset)) ** gamma

        derivative = (power(curve.x_break + step) - power(curve.x_break - step)) / (
            2.0 * step
        )
        assert curve.slope == pytest.approx(derivative, rel=1e-6)


def test_the_srgb_linear_segment_is_the_reciprocal_of_the_standard_slope():
    """sRGB's published encoding divides by 12.92 below the break; a slope off
    by a factor is what this pins."""
    assert emitters.moncurve(*SRGB).slope == pytest.approx(1.0 / 12.92, rel=1e-3)


@pytest.mark.parametrize("style", STYLES)
def test_the_forward_breakpoints_are_zero_and_the_input_break(style):
    transform = moncurve_transform(*SRGB, style=style)
    curve = emitters.moncurve(*SRGB)
    assert emitters.breakpoints(transform) == [0.0] + [curve.x_break] * CHANNELS


@pytest.mark.parametrize("style", STYLES)
def test_the_inverse_breakpoints_are_zero_and_the_output_break(style):
    """An inverse op's break lies in the encoded domain, not the linear one."""
    transform = moncurve_transform(*SRGB, style=style, inverse=True)
    curve = emitters.moncurve(*SRGB)
    assert emitters.breakpoints(transform) == [0.0] + [curve.y_break] * CHANNELS
    assert curve.y_break != pytest.approx(curve.x_break)


def test_the_forward_break_is_where_the_branch_switches(config, compile_bare):
    """Either side of the break lands on a different expression: linear below,
    power above. Read onto the wrong side, each is visibly wrong."""
    curve = emitters.moncurve(*SRGB)
    delta = 0.02
    below, above = curve.x_break - delta, curve.x_break + delta
    transform = moncurve_transform(*SRGB)
    got = run_graph(
        compile_bare(config.getProcessor(transform)), row(below, above)
    ).ravel()[:2]

    def power(x):
        return ((x + SRGB[1]) / (1.0 + SRGB[1])) ** SRGB[0]

    assert got[0] == pytest.approx(curve.slope * below, abs=1e-6)
    assert got[0] != pytest.approx(power(below), abs=1e-6)
    assert got[1] == pytest.approx(power(above), abs=1e-6)
    assert got[1] != pytest.approx(curve.slope * above, abs=1e-6)


def test_the_inverse_break_is_where_the_branch_switches(config, compile_bare):
    """The inverse break is two orders of magnitude below the forward one, so
    reading the wrong one onto the comparison misplaces the join entirely."""
    curve = emitters.moncurve(*SRGB)
    delta = 0.002
    below, above = curve.y_break - delta, curve.y_break + delta
    transform = moncurve_transform(*SRGB, style=OCIO.NEGATIVE_LINEAR, inverse=True)
    got = run_graph(
        compile_bare(config.getProcessor(transform)), row(below, above)
    ).ravel()[:2]

    def inverse_power(y):
        return y ** (1.0 / SRGB[0]) * (1.0 + SRGB[1]) - SRGB[1]

    assert got[0] == pytest.approx(below / curve.slope, abs=1e-6)
    assert got[1] == pytest.approx(inverse_power(above), abs=1e-6)
    assert got[1] != pytest.approx(above / curve.slope, abs=1e-6)


def test_mirror_reflects_the_whole_curve_through_the_origin(config, compile_bare):
    transform = moncurve_transform(*SRGB, style=OCIO.NEGATIVE_MIRROR)
    got = run_graph(
        compile_bare(config.getProcessor(transform)), row(-0.5, 0.5)
    ).ravel()[:2]
    assert got[0] == pytest.approx(-got[1], abs=1e-6)


def test_linear_extends_the_segment_below_zero(config, compile_bare):
    """``NEGATIVE_LINEAR`` runs the linear segment on through the origin, so
    the negative half is a straight line rather than a reflected curve."""
    curve = emitters.moncurve(*SRGB)
    transform = moncurve_transform(*SRGB, style=OCIO.NEGATIVE_LINEAR)
    got = run_graph(
        compile_bare(config.getProcessor(transform)), row(-0.5, -0.25)
    ).ravel()[:2]
    assert got[0] == pytest.approx(curve.slope * -0.5, abs=1e-6)
    assert got[1] == pytest.approx(curve.slope * -0.25, abs=1e-6)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_the_graph_is_finite_across_the_lattice(style, inverse, config, compile_bare):
    """The clip before each ``Pow`` is not decoration: without it the branch
    ``Where`` discards still evaluates, and NaN is what it evaluates to."""
    transform = moncurve_transform(*SRGB, style=style, inverse=inverse)
    processor = config.getProcessor(transform)
    got = run_graph(compile_bare(processor), lattice(processor))
    assert np.isfinite(got).all()


def test_no_op_in_the_pinned_config_has_a_zero_offset(config):
    """A zero offset degenerates: the break divides by ``gamma - 1`` and the
    linear segment vanishes. It does not occur here, which is why the emitter
    refuses it rather than guessing at it."""
    offsets = [
        value
        for _, processor in enumerate_transforms(config)
        for transform in processor.createGroupTransform()
        if type(transform).__name__ == "ExponentWithLinearTransform"
        for value in list(transform.getOffset())[:3]
    ]
    assert offsets
    assert all(value != 0.0 for value in offsets)


def test_a_zero_offset_is_refused_rather_than_guessed_at(config, compile_bare):
    transform = moncurve_transform(2.4, 0.0)
    with pytest.raises(UnsupportedOpError, match="offset"):
        compile_bare(config.getProcessor(transform))
