"""The lattice, the tolerance, and the reference (§spec:verification,
§spec:verification).

The harness is the failing test every emitter is written against, so its own
failure modes are worth testing directly: a lattice that misses the
interesting inputs, or a tolerance that accepts a deviation it should not,
would pass a misread parameter through in silence.
"""

import dataclasses

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import resolve_colorspaces
from ocio2onnx.builder import CHANNELS
from ocio2onnx.oracle import (
    OPTIMIZATION_FLAGS,
    TOLERANCE,
    compare,
    cpu_reference,
    lattice,
)

EXTREMES = [-65504.0, -100.0, 100.0, 1000.0, 65504.0]
NEAR_ZERO = [-1e-3, -1e-6, 0.0, 1e-9, 1e-6, 1e-3]
ANCHORS = [0.0078125, 0.18, 1.0, 10.0]


def values(samples: np.ndarray, channel: int = 0) -> np.ndarray:
    """The lattice's base vector, as it reaches one channel."""
    return samples[0, channel, 0, :]


def contains(samples: np.ndarray, value: float) -> bool:
    """Whether a channel carries a value exactly, at float32."""
    return bool((values(samples) == np.float32(value)).any())


def test_lattice_has_the_graph_shape_and_precision():
    samples = lattice()
    assert samples.dtype == np.float32
    assert samples.ndim == 4
    assert samples.shape[0] == 1
    assert samples.shape[1] == CHANNELS
    assert samples.shape[2] == 1


def test_lattice_sweeps_across_and_beyond_the_unit_interval():
    samples = lattice()
    v = values(samples)
    assert v.min() < -1.0
    assert v.max() > 2.0
    inside = v[(v >= -1.0) & (v <= 2.0)]
    assert inside.size >= 401


@pytest.mark.parametrize("value", EXTREMES)
def test_lattice_carries_the_out_of_range_extremes(value):
    assert contains(lattice(), value)


@pytest.mark.parametrize("value", NEAR_ZERO)
def test_lattice_carries_near_zero_on_both_sides(value):
    assert contains(lattice(), value)


@pytest.mark.parametrize("value", ANCHORS)
def test_lattice_carries_the_grey_and_white_anchors(value):
    assert contains(lattice(), value)


def test_lattice_channels_are_not_identical():
    samples = lattice()
    a, b, c = (values(samples, i) for i in range(CHANNELS))
    assert not np.array_equal(a, b)
    assert not np.array_equal(b, c)
    assert not np.array_equal(a, c)
    assert set(a.tolist()) == set(b.tolist()) == set(c.tolist())


def test_matrix_declares_no_breakpoints(config):
    processor = config.getProcessor("ACEScg", "ACES2065-1")
    transform = processor.createGroupTransform()[0]
    assert emitters.breakpoints(transform) == []
    assert values(lattice(processor)).size == values(lattice()).size


def register_breakpoints(monkeypatch, *points):
    """Give ``Matrix`` breakpoints it does not have, to exercise the hook."""
    entry = emitters.REGISTRY["Matrix"]
    monkeypatch.setitem(
        emitters.REGISTRY,
        "Matrix",
        dataclasses.replace(entry, breakpoints=lambda transform: list(points)),
    )


def test_declared_breakpoints_widen_the_lattice(config, monkeypatch):
    processor = config.getProcessor("ACEScg", "ACES2065-1")
    assert not contains(lattice(), 0.123456)

    register_breakpoints(monkeypatch, 0.123456)
    widened = lattice(processor)

    assert contains(widened, 0.123456)
    assert values(widened).size > values(lattice()).size


def test_a_breakpoint_is_straddled_on_both_sides(config, monkeypatch):
    """A breakpoint on the wrong side of a comparison is exactly the
    misreading the harness exists to catch (§spec:verification)."""
    register_breakpoints(monkeypatch, 0.123456)
    v = values(lattice(config.getProcessor("ACEScg", "ACES2065-1")))
    assert v[(v > np.float32(0.1234)) & (v < np.float32(0.123456))].size
    assert v[(v > np.float32(0.123456)) & (v < np.float32(0.1235))].size


def test_a_near_zero_breakpoint_gets_an_absolute_nudge(config, monkeypatch):
    """Scaling a breakpoint near zero moves it nowhere, so the straddle needs
    an absolute floor or both sides collapse onto the breakpoint itself."""
    register_breakpoints(monkeypatch, 1e-9)
    v = values(lattice(config.getProcessor("ACEScg", "ACES2065-1")))
    assert v[(v > np.float32(1e-6)) & (v < np.float32(1e-5))].size


def deviate(want, delta):
    """A reference array and a graph output differing by ``delta``."""
    w = np.array([[[[want]]]], dtype=np.float64)
    return w, w + delta


def test_a_large_value_is_governed_by_the_relative_bound():
    assert compare(*deviate(1e8, 9e3)).ok
    assert not compare(*deviate(1e8, 1.1e4)).ok


def test_a_near_zero_value_is_governed_by_the_absolute_bound():
    assert compare(*deviate(0.0, 1.5e-5)).ok
    assert not compare(*deviate(0.0, 3e-5)).ok


def test_the_tolerance_is_the_looser_bound_not_their_sum():
    """At ``want`` = 0.2 both bounds are 2e-5, so a sum-form tolerance would
    admit twice what the specification allows (§spec:verification)."""
    want = 0.2
    assert TOLERANCE.bound(want) == pytest.approx(2e-5)
    deviation = 3e-5
    assert deviation < TOLERANCE.absolute + TOLERANCE.relative * want
    assert not compare(*deviate(want, deviation)).ok


def test_a_failure_names_the_channel_the_input_and_both_values():
    samples = lattice()
    want = np.zeros_like(samples, dtype=np.float64)
    got = want.copy()
    got[0, 1, 0, 3] = 0.5
    result = compare(want, got, samples)

    assert not result.ok
    assert result.failures == 1
    assert result.max_abs == pytest.approx(0.5)
    assert result.worst is not None
    assert result.worst.channel == 1
    assert result.worst.value == pytest.approx(float(samples[0, 1, 0, 3]))
    report = str(result)
    assert "channel 1" in report
    assert "0.5" in report


def test_a_matching_pair_reports_no_failures():
    samples = lattice()
    want = np.asarray(samples, dtype=np.float64)
    result = compare(want, want.copy(), samples)
    assert result.ok
    assert result.failures == 0
    assert result.worst is None
    assert result.compared == want.size


def test_a_non_finite_reference_is_matched_by_class_not_dropped():
    """The three non-finite answers are distinct, and a graph returning the one
    the reference returned agrees at that sample (§spec:evidence-floor)."""
    want = np.array([[[[1.0, np.inf, -np.inf, np.nan]]]])
    result = compare(want, want.copy())
    assert result.ok
    assert result.compared == 1
    assert result.matched == 3
    assert result.disagreed == 0
    assert result.compared + result.matched + result.disagreed == want.size


def test_a_silently_non_finite_graph_does_not_pass():
    want = np.array([[[[1.0, 2.0, 3.0]]]])
    got = np.array([[[[1.0, np.nan, 3.0]]]])
    result = compare(want, got)
    assert not result.ok
    assert result.disagreed == 1
    assert "class" in str(result)


def test_a_finite_graph_where_the_reference_overflows_does_not_pass():
    want = np.array([[[[1.0, np.inf, 3.0]]]])
    got = np.array([[[[1.0, 4.0, 3.0]]]])
    result = compare(want, got)
    assert not result.ok
    assert result.disagreed == 1


def test_an_inverted_infinity_does_not_pass():
    """A direction inverted at overflow: the reference falls to ``-inf`` where
    the graph climbs to ``+inf``, with every finite sample agreeing. Dropping
    the non-finite samples reads this as agreement (§spec:evidence-floor)."""
    want = np.array([[[[1.0, np.inf, 3.0]]]])
    got = np.array([[[[1.0, -np.inf, 3.0]]]])
    result = compare(want, got)
    assert not result.ok
    assert result.compared == 2
    assert result.disagreed == 1


def test_a_nan_reference_against_an_infinite_graph_does_not_pass():
    want = np.array([[[[1.0, np.nan, 3.0]]]])
    got = np.array([[[[1.0, np.inf, 3.0]]]])
    result = compare(want, got)
    assert not result.ok
    assert result.disagreed == 1


def test_a_reference_that_is_nan_everywhere_is_not_evidence():
    """Nothing is held against the tolerance, so there is nothing to certify
    (§spec:evidence-floor)."""
    result = compare(np.full(12, np.nan), np.full(12, np.nan))
    assert not result.ok
    assert result.compared == 0
    assert result.matched == 12


def test_opposite_infinities_everywhere_are_not_evidence():
    result = compare(np.full(12, np.inf), np.full(12, -np.inf))
    assert not result.ok
    assert result.compared == 0
    assert result.disagreed == 12


def test_cpu_reference_preserves_shape_and_precision(config, config_uri):
    resolved = resolve_colorspaces(config, "ACEScg", "ACES2065-1", uri=config_uri)
    samples = lattice(resolved.processor)
    want = cpu_reference(resolved.processor, samples)
    assert want.shape == samples.shape
    assert want.dtype == np.float32


def test_cpu_reference_does_not_mutate_its_input(config):
    processor = config.getProcessor("ACEScg", "ACES2065-1")
    samples = lattice(processor)
    before = samples.copy()
    cpu_reference(processor, samples)
    assert np.array_equal(samples, before)


def test_cpu_reference_does_not_mutate_a_single_column_of_samples(config, row):
    """One column is the case the transpose does not already copy, and every
    caller runs the reference before the graph over the same array: an
    in-place ``applyRGB`` would feed the graph the reference's own output."""
    processor = config.getProcessor("ACEScg", "ACES2065-1")
    samples = row(0.5)
    cpu_reference(processor, samples)
    assert samples == pytest.approx(row(0.5))


def test_cpu_reference_clears_ocios_fast_math_path(config):
    """The oracle must be the accurate reference, not OCIO's approximate
    ``pow``/``log``/``exp``, whose error is two orders of magnitude above the
    verification tolerance (§spec:verification)."""
    processor = config.getProcessor("Log3G10 REDWideGamutRGB", "ACES2065-1")
    samples = lattice(processor)
    exact = cpu_reference(processor, samples)

    fast = np.ascontiguousarray(samples[0, :, 0, :].T)
    processor.getDefaultCPUProcessor().applyRGB(fast)
    fast = fast.T.reshape(exact.shape)

    finite = np.isfinite(exact) & np.isfinite(fast)
    assert not np.array_equal(exact[finite], fast[finite])


def test_fast_log_exp_pow_is_the_only_optimization_cleared():
    assert (
        OCIO.OPTIMIZATION_DEFAULT.value ^ OPTIMIZATION_FLAGS.value
        == OCIO.OPTIMIZATION_FAST_LOG_EXP_POW.value
    )
    assert OPTIMIZATION_FLAGS.value & OCIO.OPTIMIZATION_LUT_INV_FAST.value


#: A pixel whose channels disagree, carrying 65504 in the first. OCIO's exact
#: inverse ``Lut1D`` renderer answers it correctly when all three channels hold
#: the same value and incorrectly when they do not, and the lattice rolls each
#: channel so they never do (``CHANNEL_ROLLS``).
INVERSE_LUT_PIXEL = [65504.0, 0.18, 1.0]


def test_clearing_the_fast_inverse_lut_would_move_the_oracle_off_the_answer(config):
    """``OPTIMIZATION_LUT_INV_FAST`` stays on, and not because it is inert.

    Cleared, OCIO's inverse ``Lut1D`` renderer parts company with its own fast
    path at the top of the domain — 5 samples across the pinned config, every
    one at an input of 65504. The fast path is the one that is right there:
    ACEScc encodes a linear value as ``(log2(x) + 9.72) / 17.52``, which puts
    65504 at 1.468, and the exact path answers 1.4987.

    Which flags the oracle runs under is a correctness decision, and this is
    the direction it points here: clearing this one would fail a graph for
    agreeing with the encoding (§spec:verification).
    """
    processor = config.getProcessor("ACES2065-1", "ACEScc")
    want = (np.log2(65504.0) + 9.72) / 17.52

    def encoded(flags):
        pixel = np.array([INVERSE_LUT_PIXEL], dtype=np.float32, order="C")
        processor.getOptimizedCPUProcessor(flags).applyRGB(pixel)
        return float(pixel[0, 0])

    assert encoded(OPTIMIZATION_FLAGS) == pytest.approx(want, rel=1e-4)
    cleared = OCIO.OptimizationFlags(
        OPTIMIZATION_FLAGS.value & ~OCIO.OPTIMIZATION_LUT_INV_FAST.value
    )
    assert abs(encoded(cleared) - want) > TOLERANCE.bound(want)
