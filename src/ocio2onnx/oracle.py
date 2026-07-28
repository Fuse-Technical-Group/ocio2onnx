"""OCIO's CPU processor is the oracle (§spec:verification, §road:oracle-harness).

The failure mode this catches is not a wrong constant but a subtly wrong
reading of the config — a direction inverted, a breakpoint on the wrong side
of a comparison. Hand-checked values do not catch that; a lattice against the
reference does. So the harness comes before the emitters: it is the failing
test each one is written against.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx
import onnxruntime as ort
import PyOpenColorIO as OCIO

from ocio2onnx import emitters
from ocio2onnx.addressing import CHANNELS, Resolved
from ocio2onnx.builder import INPUT
from ocio2onnx.compiler import compile_processor

#: OCIO's default CPU optimization includes ``OPTIMIZATION_FAST_LOG_EXP_POW``,
#: an approximate ``pow``/``log``/``exp``. Its error exceeds the verification
#: tolerance, so leaving it on measures a math library rather than the
#: compiler's reading of the config: across the pinned config 110 of the 111
#: closed-form transforms verify against OCIO's fast path, and 111 of 111
#: verify against its accurate one. The oracle must be the accurate reference,
#: so the flag is cleared. This is a correctness decision, not a tuning knob;
#: ``tests/test_oracle.py`` pins it.
OPTIMIZATION_FLAGS = OCIO.OptimizationFlags(
    OCIO.OPTIMIZATION_DEFAULT.value & ~OCIO.OPTIMIZATION_FAST_LOG_EXP_POW.value
)

#: A dense sweep across and beyond the unit interval.
SWEEP = (-1.0, 2.0, 401)

#: Values outside any transform's design range. float16's finite limit bounds
#: both ends: a consumer feeding a half-float image cannot exceed it.
EXTREMES = (-65504.0, -100.0, 100.0, 1000.0, 65504.0)

#: Either side of zero, where a log or a division is most fragile.
NEAR_ZERO = (-1e-3, -1e-6, 0.0, 1e-9, 1e-6, 1e-3)

#: 18% mid grey, diffuse white, and the two decades around them.
ANCHORS = (0.0078125, 0.18, 1.0, 10.0)

#: How far either side of a breakpoint the lattice samples. The absolute term
#: is a floor: scaling a breakpoint near zero moves it nowhere.
BREAKPOINT_REL = 1e-4
BREAKPOINT_ABS = 1e-6

#: Each channel carries the same values in a different order, so a graph that
#: swaps or drops a channel cannot agree with the reference.
CHANNEL_ROLLS = (0, 7, 13)


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """The looser of an absolute and a relative bound (§spec:verification).

    Stated both ways because either alone is wrong, and combined with ``max``
    rather than a sum: a sum would admit up to twice the deviation wherever
    the two bounds meet.
    """

    absolute: float
    relative: float

    def bound(self, want):
        """The largest deviation from ``want`` that still agrees."""
        return np.maximum(self.absolute, self.relative * np.abs(want))


TOLERANCE = Tolerance(absolute=2e-5, relative=1e-4)


@dataclasses.dataclass(frozen=True)
class Worst:
    """The sample that missed the tolerance by the widest margin."""

    channel: int
    index: int
    value: float
    want: float
    got: float
    absolute: float
    relative: float


@dataclasses.dataclass(frozen=True)
class Comparison:
    """What the oracle found, in enough detail to act on without a debugger."""

    ok: bool
    compared: int
    failures: int
    nonfinite: int
    finite_mismatch: int
    max_abs: float
    max_rel: float
    worst: Worst | None

    def __str__(self) -> str:
        parts = [
            f"{self.failures} of {self.compared} samples outside tolerance "
            f"(max abs {self.max_abs:.3g}, max rel {self.max_rel:.3g})"
        ]
        if self.finite_mismatch:
            parts.append(
                f"{self.finite_mismatch} samples where the graph is finite and "
                "the reference is not, or the reverse"
            )
        if self.nonfinite:
            parts.append(f"{self.nonfinite} non-finite reference samples ignored")
        if self.worst is not None:
            w = self.worst
            parts.append(
                f"worst at channel {w.channel} index {w.index}, "
                f"input {w.value:.9g}: want {w.want:.9g}, got {w.got:.9g} "
                f"(abs {w.absolute:.3g}, rel {w.relative:.3g})"
            )
        return "; ".join(parts)


def straddle(value: float) -> tuple[float, float, float]:
    """A breakpoint and the two values either side of it."""
    delta = max(abs(value) * BREAKPOINT_REL, BREAKPOINT_ABS)
    return value, value - delta, value + delta


def lattice(processor: OCIO.Processor | None = None) -> np.ndarray:
    """Input samples shaped ``(1, 3, 1, N)`` at float32.

    Covers the unit interval and beyond, the out-of-range extremes, both sides
    of zero, and the grey and white anchors. Given a processor, each op is
    asked where it switches branches and the lattice straddles every answer
    (``emitters.breakpoints``), so an op added later widens the lattice by
    declaring its breakpoints rather than by editing this function.
    """
    points = [
        np.linspace(*SWEEP),
        np.array(EXTREMES + NEAR_ZERO + ANCHORS, dtype=np.float64),
    ]
    if processor is not None:
        for transform in processor.createGroupTransform():
            for value in emitters.breakpoints(transform):
                points.append(np.array(straddle(value), dtype=np.float64))

    values = np.unique(np.concatenate(points)).astype(np.float32)
    rolled = np.stack([np.roll(values, roll) for roll in CHANNEL_ROLLS])
    return np.ascontiguousarray(rolled.reshape(1, CHANNELS, 1, -1))


def cpu_reference(processor: OCIO.Processor, samples: np.ndarray) -> np.ndarray:
    """Run ``samples`` through OCIO's CPU processor.

    ``applyRGB`` wants a contiguous ``(N, 3)`` float32 array and mutates it in
    place, so the channels-first lattice is transposed in and back out.
    """
    shape = np.shape(samples)
    planes = np.reshape(samples, (shape[0], CHANNELS, -1))
    pixels = np.ascontiguousarray(
        planes.transpose(0, 2, 1).reshape(-1, CHANNELS), dtype=np.float32
    )
    processor.getOptimizedCPUProcessor(OPTIMIZATION_FLAGS).applyRGB(pixels)
    return np.reshape(
        pixels.reshape(planes.shape[0], planes.shape[2], CHANNELS).transpose(0, 2, 1),
        shape,
    )


def run_graph(model: onnx.ModelProto, samples: np.ndarray) -> np.ndarray:
    """Execute an emitted graph over ``samples``."""
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inputs = {INPUT: np.ascontiguousarray(samples, dtype=np.float32)}
    return session.run(None, inputs)[0]


def compare(
    want: np.ndarray, got: np.ndarray, samples: np.ndarray | None = None
) -> Comparison:
    """Hold the graph's output against the reference (§spec:verification).

    Non-finite reference values are ignored rather than compared — some
    transforms overflow float32 at 65504 — but the graph must be non-finite in
    exactly the same places, or a silently NaN graph would pass.
    """
    want = np.asarray(want, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if want.shape != got.shape:
        raise ValueError(f"reference is {want.shape}, graph output is {got.shape}")

    reference_finite = np.isfinite(want)
    mismatch = reference_finite != np.isfinite(got)
    usable = np.flatnonzero(reference_finite & ~mismatch)

    w = want.ravel()[usable]
    g = got.ravel()[usable]
    deviation = np.abs(g - w)
    relative = np.divide(deviation, np.abs(w), out=np.zeros_like(w), where=w != 0)
    bound = TOLERANCE.bound(w)
    over = deviation > bound

    worst = None
    if over.any():
        k = int(np.argmax(deviation / bound))
        coords = np.unravel_index(usable[k], want.shape)
        worst = Worst(
            channel=int(coords[1]),
            index=int(usable[k]),
            value=float(samples.ravel()[usable[k]]) if samples is not None else np.nan,
            want=float(w[k]),
            got=float(g[k]),
            absolute=float(deviation[k]),
            relative=float(relative[k]),
        )

    failures = int(over.sum())
    finite_mismatch = int(mismatch.sum())
    return Comparison(
        ok=failures == 0 and finite_mismatch == 0,
        compared=usable.size,
        failures=failures,
        nonfinite=int((~reference_finite).sum()),
        finite_mismatch=finite_mismatch,
        max_abs=float(deviation.max()) if deviation.size else 0.0,
        max_rel=float(relative.max()) if relative.size else 0.0,
        worst=worst,
    )


def verify(resolved: Resolved, model: onnx.ModelProto | None = None) -> Comparison:
    """Hold a graph against the oracle, compiling one if none is given.

    A caller holding the model it is about to write hands it over, so what is
    verified is the artifact rather than a second compilation of the request.
    """
    samples = lattice(resolved.processor)
    return compare(
        cpu_reference(resolved.processor, samples),
        run_graph(compile_processor(resolved) if model is None else model, samples),
        samples,
    )
