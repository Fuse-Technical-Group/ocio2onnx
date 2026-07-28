"""``Exponent`` emits a power with the sign put back (§spec:op-emission).

ONNX ``Pow`` on a negative base is NaN, and ``Where`` evaluates both branches,
so the negative half of the domain is the whole difficulty here. Both negative
styles the pinned config carries are exercised, in both directions — 18 of the
26 ``Exponent`` ops in it arrive inverse (§spec:op-coverage).
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import resolve_colorspaces
from ocio2onnx.compiler import UnsupportedOpError
from ocio2onnx.oracle import lattice, run_graph, verify

#: One closed-form transform per (direction, negative style) the config
#: carries. Forward reaches the reference; inverse leaves it.
MIRROR_FORWARD = ("Gamma 2.2 Rec.709 - Display", "ACES2065-1")
MIRROR_INVERSE = ("ACES2065-1", "Gamma 2.2 Rec.709 - Display")
PASS_THRU_FORWARD = ("Gamma 1.8 Encoded Rec.709", "ACES2065-1")
PASS_THRU_INVERSE = ("ACES2065-1", "Gamma 1.8 Encoded Rec.709")

PAIRS = (MIRROR_FORWARD, MIRROR_INVERSE, PASS_THRU_FORWARD, PASS_THRU_INVERSE)

STYLES = (OCIO.NEGATIVE_MIRROR, OCIO.NEGATIVE_PASS_THRU)

EXPONENT = "Exponent"


def exponent_transform(value, *, style=OCIO.NEGATIVE_MIRROR, inverse=False):
    """A bare ``ExponentTransform`` at one gamma."""
    transform = OCIO.ExponentTransform()
    transform.setValue([value] * 3 + [1.0])
    transform.setNegativeStyle(style)
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def test_the_registry_carries_exponent():
    assert EXPONENT in emitters.REGISTRY


@pytest.mark.parametrize("pair", PAIRS)
def test_a_config_transform_carrying_exponent_verifies(pair, config, config_uri):
    result = verify(resolve_colorspaces(config, *pair, uri=config_uri))
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(
    ("pair", "style"),
    (
        (MIRROR_FORWARD, "NEGATIVE_MIRROR"),
        (MIRROR_INVERSE, "NEGATIVE_MIRROR"),
        (PASS_THRU_FORWARD, "NEGATIVE_PASS_THRU"),
        (PASS_THRU_INVERSE, "NEGATIVE_PASS_THRU"),
    ),
)
def test_the_pairs_under_test_carry_the_styles_they_claim(pair, style, op_in):
    op = op_in(EXPONENT, *pair)
    assert str(op.getNegativeStyle()).endswith(style)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_a_bare_exponent_verifies(style, inverse, check_transform):
    result = check_transform(exponent_transform(2.4, style=style, inverse=inverse))
    assert result.ok, str(result)


def test_an_inverse_exponent_reciprocates_its_gamma(config, compile_bare):
    """The inverse of ``x**g`` is ``x**(1/g)``, not ``-g`` and not ``1-g``."""
    transform = exponent_transform(2.0, inverse=True)
    samples = np.array([[[[4.0, 9.0, 0.25]]] * 3], dtype=np.float32)
    got = run_graph(compile_bare(config.getProcessor(transform)), samples)
    assert got.ravel()[:3] == pytest.approx([2.0, 3.0, 0.5], rel=1e-6)


def test_mirror_reflects_the_curve_through_the_origin(config, compile_bare):
    transform = exponent_transform(2.0, style=OCIO.NEGATIVE_MIRROR)
    samples = np.array([[[[-0.5, 0.0, 0.5]]] * 3], dtype=np.float32)
    got = run_graph(compile_bare(config.getProcessor(transform)), samples)
    assert got.ravel()[:3] == pytest.approx([-0.25, 0.0, 0.25], abs=1e-6)


def test_pass_thru_leaves_the_negative_half_alone(config, compile_bare):
    transform = exponent_transform(2.0, style=OCIO.NEGATIVE_PASS_THRU)
    samples = np.array([[[[-0.5, 0.0, 0.5]]] * 3], dtype=np.float32)
    got = run_graph(compile_bare(config.getProcessor(transform)), samples)
    assert got.ravel()[:3] == pytest.approx([-0.5, 0.0, 0.25], abs=1e-6)


def test_the_negative_half_is_free_of_the_nan_pow_would_produce(config, compile_bare):
    """``Pow`` on a negative base is NaN in ONNX. A graph that reached it would
    poison the output whichever branch ``Where`` selects."""
    for style in STYLES:
        transform = exponent_transform(2.4, style=style)
        samples = lattice(config.getProcessor(transform))
        got = run_graph(compile_bare(config.getProcessor(transform)), samples)
        assert np.isfinite(got).all()


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_zero_is_the_declared_breakpoint(style, inverse):
    transform = exponent_transform(2.4, style=style, inverse=inverse)
    assert emitters.breakpoints(transform) == [0.0]


def test_zero_is_where_the_sign_handling_switches(config, compile_bare):
    """Either side of the breakpoint lands on a different branch: the curve is
    a power above it and sign-dependent below."""
    (breakpoint_,) = emitters.breakpoints(exponent_transform(2.0))
    samples = np.array(
        [[[[breakpoint_ - 0.5, breakpoint_ + 0.5]]] * 3], dtype=np.float32
    )

    mirror = config.getProcessor(exponent_transform(2.0, style=OCIO.NEGATIVE_MIRROR))
    pass_thru = config.getProcessor(
        exponent_transform(2.0, style=OCIO.NEGATIVE_PASS_THRU)
    )
    above = run_graph(compile_bare(mirror), samples).ravel()[1]
    below_mirror = run_graph(compile_bare(mirror), samples).ravel()[0]
    below_pass_thru = run_graph(compile_bare(pass_thru), samples).ravel()[0]

    assert above == pytest.approx(0.25, abs=1e-6)
    assert below_mirror == pytest.approx(-0.25, abs=1e-6)
    assert below_pass_thru == pytest.approx(-0.5, abs=1e-6)
    assert below_mirror != pytest.approx(below_pass_thru)


def test_an_unhandled_negative_style_is_refused_by_name(config, compile_bare):
    """A style the compiler has no path for is refused rather than treated as
    the nearest one it does: the two differ exactly where the style governs."""
    transform = exponent_transform(2.4, style=OCIO.NEGATIVE_CLAMP)
    with pytest.raises(UnsupportedOpError, match="CLAMP"):
        compile_bare(config.getProcessor(transform))
