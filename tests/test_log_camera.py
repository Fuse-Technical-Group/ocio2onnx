"""``LogCamera`` emits a parametric camera log curve (§spec:op-emission).

One op covers every vendor curve in the config — ACEScct, ARRI LogC3 and
LogC4, BMDFilm, DaVinci Intermediate, D-Log, V-Log, RED Log3G10, and the
S-Log3 family — with different coefficients, which OCIO hands over as plain
floats (§spec:op-coverage).

The linear slope is the parameter to get wrong. OCIO reports it as NaN rather
than as an absent value where the config leaves it unset, which is 14 of the
24 ops here; taking the NaN at face value yields a graph that is NaN below the
break and finite above it.
"""

import math

import numpy as np
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import CHANNELS, enumerate_transforms, resolve_colorspaces
from ocio2onnx.oracle import lattice, run_graph, verify

#: A curve whose linear slope the config leaves unset, and one it declares.
DERIVED = "Log3G10 REDWideGamutRGB"
DECLARED = "S-Log3 S-Gamut3"
REFERENCE = "ACES2065-1"

PAIRS = (
    (DERIVED, REFERENCE),
    (REFERENCE, DERIVED),
    (DECLARED, REFERENCE),
    (REFERENCE, DECLARED),
)


def log_camera_op(config, src, dst):
    """The ``LogCamera`` op inside one config transform."""
    ops = [
        t
        for t in config.getProcessor(src, dst).createGroupTransform()
        if type(t).__name__ == "LogCameraTransform"
    ]
    assert len(ops) == 1
    return ops[0]


def row(*values):
    """A lattice-shaped sample carrying ``values`` in every channel."""
    return np.array([[[list(values)]] * 3], dtype=np.float32)


def declares_linear_slope(transform):
    """Whether the config set this op's linear slope, NaN meaning it did not."""
    values = transform.getLinearSlopeValue()
    return bool(values) and not any(math.isnan(float(v)) for v in values)


def test_the_registry_carries_log_camera():
    assert "LogCameraTransform" in emitters.REGISTRY


def test_the_pinned_config_carries_both_declared_and_unset_linear_slopes(config):
    """Both paths are exercised by real data, so neither is untested code."""
    states = [
        declares_linear_slope(transform)
        for _, processor in enumerate_transforms(config)
        for transform in processor.createGroupTransform()
        if type(transform).__name__ == "LogCameraTransform"
    ]
    assert states.count(False) == 14
    assert states.count(True) == 10


@pytest.mark.parametrize("pair", PAIRS)
def test_a_config_transform_carrying_log_camera_verifies(pair, config, config_uri):
    result = verify(resolve_colorspaces(config, *pair, uri=config_uri))
    assert result.ok, str(result)
    assert result.compared > 0


def test_every_log_camera_in_the_pinned_config_verifies(config, check_transform):
    failures = []
    for label, processor in enumerate_transforms(config):
        for transform in processor.createGroupTransform():
            if type(transform).__name__ != "LogCameraTransform":
                continue
            result = check_transform(transform)
            if not result.ok:
                failures.append(f"{label}: {result}")
    assert not failures, "\n".join(failures)


def test_the_pairs_under_test_are_the_two_linear_slope_cases(config):
    assert not declares_linear_slope(log_camera_op(config, DERIVED, REFERENCE))
    assert declares_linear_slope(log_camera_op(config, DECLARED, REFERENCE))


def test_a_declared_linear_slope_is_used_as_given(config):
    transform = log_camera_op(config, DECLARED, REFERENCE)
    declared = [float(v) for v in transform.getLinearSlopeValue()][:CHANNELS]
    curve = emitters.camera_curve(transform)
    assert curve.linear_slope == pytest.approx(declared, rel=1e-12)


def test_an_unset_linear_slope_is_derived_rather_than_taken_as_nan(config):
    transform = log_camera_op(config, DERIVED, REFERENCE)
    curve = emitters.camera_curve(transform)
    assert all(math.isfinite(value) for value in curve.linear_slope)
    assert all(value != 0.0 for value in curve.linear_slope)


def test_the_derived_linear_slope_makes_the_break_continuous_in_slope(config):
    """C1 is what the derivation is for: the linear segment leaves the break at
    the rate the curve arrives at it."""
    transform = log_camera_op(config, DERIVED, REFERENCE)
    curve = emitters.camera_curve(transform)
    ln_base = math.log(curve.base)

    for i in range(CHANNELS):
        step = abs(curve.lin_break[i]) * 1e-7
        low, high = curve.lin_break[i] - step, curve.lin_break[i] + step

        def curved(x, i=i, ln_base=ln_base):
            argument = curve.lin_slope[i] * x + curve.lin_offset[i]
            return (
                curve.log_slope[i] * math.log(argument) / ln_base + curve.log_offset[i]
            )

        derivative = (curved(high) - curved(low)) / (2.0 * step)
        assert curve.linear_slope[i] == pytest.approx(derivative, rel=1e-6)


@pytest.mark.parametrize("name", (DERIVED, DECLARED))
def test_the_break_is_continuous_in_value(config, name):
    """The linear segment meets the curve at the break rather than stepping."""
    curve = emitters.camera_curve(log_camera_op(config, name, REFERENCE))
    for i in range(CHANNELS):
        linear = curve.linear_slope[i] * curve.lin_break[i] + curve.linear_offset[i]
        assert linear == pytest.approx(curve.log_break[i], rel=1e-9)


def test_the_forward_breakpoints_are_the_linear_side_break(config):
    transform = log_camera_op(config, REFERENCE, DERIVED)
    curve = emitters.camera_curve(transform)
    assert emitters.breakpoints(transform) == curve.lin_break


def test_the_inverse_breakpoints_are_the_log_side_break(config):
    """An inverse op's break lies in the encoded domain, not the linear one."""
    transform = log_camera_op(config, DERIVED, REFERENCE)
    curve = emitters.camera_curve(transform)
    assert emitters.breakpoints(transform) == curve.log_break
    assert curve.log_break[0] != pytest.approx(curve.lin_break[0])


def test_the_forward_break_is_where_the_branch_switches(config, compile_bare):
    transform = log_camera_op(config, REFERENCE, DERIVED)
    curve = emitters.camera_curve(transform)
    delta = 0.05
    below, above = curve.lin_break[0] - delta, curve.lin_break[0] + delta
    got = run_graph(
        compile_bare(config.getProcessor(transform)), row(below, above)
    ).ravel()[:2]

    def linear(x):
        return curve.linear_slope[0] * x + curve.linear_offset[0]

    def curved(x):
        argument = max(curve.lin_slope[0] * x + curve.lin_offset[0], emitters.LOG_FLOOR)
        return (
            curve.log_slope[0] * math.log(argument) / math.log(curve.base)
            + curve.log_offset[0]
        )

    assert got[0] == pytest.approx(linear(below), abs=1e-5)
    assert got[0] != pytest.approx(curved(below), abs=1e-5)
    assert got[1] == pytest.approx(curved(above), abs=1e-5)
    assert got[1] != pytest.approx(linear(above), abs=1e-5)


def test_the_inverse_break_is_where_the_branch_switches(config, compile_bare):
    transform = log_camera_op(config, DERIVED, REFERENCE)
    curve = emitters.camera_curve(transform)
    delta = 0.05
    below, above = curve.log_break[0] - delta, curve.log_break[0] + delta
    got = run_graph(
        compile_bare(config.getProcessor(transform)), row(below, above)
    ).ravel()[:2]

    def linear(y):
        return (y - curve.linear_offset[0]) / curve.linear_slope[0]

    def curved(y):
        power = curve.base ** ((y - curve.log_offset[0]) / curve.log_slope[0])
        return (power - curve.lin_offset[0]) / curve.lin_slope[0]

    assert got[0] == pytest.approx(linear(below), abs=1e-5)
    assert got[0] != pytest.approx(curved(below), abs=1e-5)
    assert got[1] == pytest.approx(curved(above), abs=1e-5)
    assert got[1] != pytest.approx(linear(above), abs=1e-5)


@pytest.mark.parametrize("pair", PAIRS)
def test_the_graph_is_finite_below_the_break(pair, config, compile_bare):
    """A NaN linear slope taken at face value yields a graph that is NaN below
    the break and finite above it — a failure the oracle reports as a finite
    mismatch and a bare tolerance check would miss entirely."""
    processor = config.getProcessor(*pair)
    samples = lattice(processor)
    got = run_graph(compile_bare(processor), samples)
    assert not np.isnan(got).any()
