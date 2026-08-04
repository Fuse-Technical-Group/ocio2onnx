"""OCIO's CPU processor is the oracle (§spec:verification).

The failure mode this catches is not a wrong constant but a subtly wrong
reading of the config — a direction inverted, a breakpoint on the wrong side
of a comparison. Hand-checked values do not catch that; a lattice against the
reference does. So the harness comes before the emitters: it is the failing
test each one is written against.

Every sample it draws is evidence: one the tolerance decides, or one where the
two sides carry the same class of non-finite value. None is dropped, and a
graph verifies only once the finite kind has occurred (§spec:evidence-floor).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import PyOpenColorIO as OCIO

from ocio2onnx import emitters
from ocio2onnx.addressing import OPTIMIZATION_FLAGS, Resolved
from ocio2onnx.builder import CHANNELS, INPUT
from ocio2onnx.compiler import compile_processor

#: Re-exported: the reference is OCIO's CPU processor at the same optimization
#: level the compiler read its op list at, and reading them from one place is
#: what makes that so (`addressing.OPTIMIZATION_FLAGS`).
__all__ = [
    "OPTIMIZATION_FLAGS",
    "Comparison",
    "ProviderError",
    "compare",
    "lattice",
    "verify",
]

#: Where an emitted graph runs unless a caller says otherwise. The reference is
#: OCIO's CPU processor, so executing the graph on the same kind of arithmetic
#: keeps a disagreement attributable to the compiler rather than to a runtime's
#: liberties — fused multiplies and lower-precision transcendentals both move a
#: result by more than this module's tolerance allows (§spec:verification).
DEFAULT_PROVIDER = "CPUExecutionProvider"

#: Short names for the runtimes a caller is likely to ask for. Anything absent
#: is passed through unchanged, so a provider this table has not heard of is
#: still nameable in full.
PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

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
class Inverted:
    """A sample where the graph answered a different kind of value.

    There is no deviation to report — the point is which of the four answers
    each side gave, and at which input.
    """

    channel: int
    index: int
    value: float
    want: float
    got: float


@dataclasses.dataclass(frozen=True)
class Comparison:
    """What the oracle found, in enough detail to act on without a debugger.

    Every sample lands in exactly one of three counts (§spec:evidence-floor):
    ``compared`` held against the tolerance, ``matched`` where both sides are
    the same class of non-finite value, and ``disagreed`` where the classes
    differ. Their sum is the lattice, so a report states what was established
    rather than what survived a filter.

    Each of the two ways to fail names a sample: ``worst`` the widest miss of
    the tolerance, ``inverted`` the first class disagreement.

    ``margin`` is the one figure that compares across runs. ``max_abs`` and
    ``max_rel`` cannot: each is dominated by whichever samples the *other*
    bound governed, so a graph agreeing to nine digits everywhere still reports
    a large ``max_abs`` if it also handles a value of 1e31. Stating each
    deviation as a fraction of the bound that decided it removes the scale —
    1.0 sits exactly on the tolerance, whichever bound got it there.
    """

    compared: int
    failures: int
    matched: int
    disagreed: int
    max_abs: float
    max_rel: float
    worst: Worst | None
    inverted: Inverted | None = None
    margin: float = 0.0

    @property
    def ok(self) -> bool:
        """Every finite comparison inside the tolerance, every other sample on
        the reference's class, and evidence that the finite kind occurred at
        all (§spec:evidence-floor)."""
        return self.failures == 0 and self.disagreed == 0 and self.compared > 0

    def __str__(self) -> str:
        parts = [
            f"{self.failures} of {self.compared} samples outside tolerance "
            f"(max abs {self.max_abs:.3g}, max rel {self.max_rel:.3g})"
            if self.compared
            else "no finite sample was compared, so nothing was established"
        ]
        if self.matched:
            parts.append(f"{self.matched} non-finite samples agreed on class")
        if self.disagreed:
            parts.append(
                f"{self.disagreed} samples disagreed on class "
                "(finite, +inf, -inf, and NaN are four distinct answers)"
            )
        if self.inverted is not None:
            i = self.inverted
            parts.append(
                f"first at channel {i.channel} index {i.index}, "
                f"input {i.value:.9g}: want {i.want:.9g}, got {i.got:.9g}"
            )
        if self.worst is not None:
            w = self.worst
            parts.append(
                f"worst at channel {w.channel} index {w.index}, "
                f"input {w.value:.9g}: want {w.want:.9g}, got {w.got:.9g} "
                f"(abs {w.absolute:.3g}, rel {w.relative:.3g})"
            )
        return "; ".join(parts)


#: The four answers a sample can carry. A non-finite value has no magnitude to
#: measure a tolerance against, so agreement there is agreement on the class
#: (§spec:evidence-floor).
FINITE, POSITIVE_INFINITY, NEGATIVE_INFINITY, NOT_A_NUMBER = range(4)


def channel_of(index: int, shape: tuple[int, ...]) -> int:
    """Which channel a flat index falls in.

    Zero where the arrays carry no channel axis, which is a caller comparing
    bare vectors rather than a lattice.
    """
    return int(np.unravel_index(index, shape)[1]) if len(shape) > 1 else 0


def classify(values: np.ndarray) -> np.ndarray:
    """Which of the four classes each value falls in.

    The labels carry no order or magnitude; they exist to be compared for
    equality.
    """
    return np.select(
        [np.isfinite(values), np.isnan(values), values == np.inf],
        [FINITE, NOT_A_NUMBER, POSITIVE_INFINITY],
        default=NEGATIVE_INFINITY,
    )


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

    The copy is unconditional. Transposing a lattice of more than one column
    yields a non-contiguous array, so ``ascontiguousarray`` alone happens to
    copy — but at one column it returns the caller's own buffer, and OCIO then
    overwrites the samples a caller is about to hand to the graph.
    """
    shape = np.shape(samples)
    planes = np.reshape(samples, (shape[0], CHANNELS, -1))
    pixels = np.array(
        planes.transpose(0, 2, 1).reshape(-1, CHANNELS), dtype=np.float32, order="C"
    )
    processor.getOptimizedCPUProcessor(OPTIMIZATION_FLAGS).applyRGB(pixels)
    return np.reshape(
        pixels.reshape(planes.shape[0], planes.shape[2], CHANNELS).transpose(0, 2, 1),
        shape,
    )


class ProviderError(RuntimeError):
    """A runtime was named that this onnxruntime build will not run on."""


def resolve_provider(name: str | None) -> str:
    """The execution provider ``name`` asks for, in onnxruntime's spelling.

    Refusing an absent provider is the point. onnxruntime reads its provider
    list as a preference rather than a requirement and quietly falls back to
    the CPU, so a caller who asked for the GPU would otherwise be told its
    graph verified there when it never left the CPU — a report about a run that
    did not happen, which is the class of answer this module exists to prevent
    (§spec:verification).
    """
    if name is None:
        return DEFAULT_PROVIDER
    provider = PROVIDER_ALIASES.get(name.lower(), name)
    available = ort.get_available_providers()
    if provider not in available:
        raise ProviderError(
            f"{name!r} is not in this onnxruntime build, which offers "
            f"{', '.join(available)}"
        )
    return provider


def run_graph(
    model: onnx.ModelProto,
    samples: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    provider: str | None = None,
    strict: bool = False,
) -> np.ndarray:
    """Execute an emitted graph over ``samples``.

    ``parameters`` binds live inputs (§spec:dynamic-properties). One left
    unbound reads the default the graph carries, which is what makes a sweep a
    statement about one artifact rather than about several.

    ``provider`` names the runtime, defaulting to the CPU for the reason
    `DEFAULT_PROVIDER` gives. A named one that loaded but did not take the
    session is refused rather than reported, on the same grounds as one that is
    absent: both would otherwise return a CPU result under a GPU's name.

    That check answers for the session, not for every node in it. onnxruntime
    appends the CPU behind whatever it was asked for and places on it whatever
    the named provider would not take — sometimes by design, as with the shape
    ops it keeps off an accelerator deliberately. ``strict`` refuses that too,
    so the provider named is the provider that ran all of it. It is not the
    default because it is a stronger claim than most callers want: on the
    pinned config's heaviest transform, TensorRT passes it and the CUDA
    provider does not.

    It cannot bind the image. A caller that named it there would run the graph
    on one array while `compare` held the answer against a reference computed
    from another, and at a matching shape that reports agreement — a graph
    certified on evidence it never produced, which is the one failure this
    module exists to prevent (§spec:verification).
    """
    bound = dict(parameters or {})
    if INPUT in bound:
        raise ValueError(f"{INPUT!r} is the image, not a live parameter")

    wanted = resolve_provider(provider)
    options = ort.SessionOptions()
    if strict:
        if wanted == DEFAULT_PROVIDER:
            raise ProviderError(
                f"{wanted} cannot be held to running no node on the CPU"
            )
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    try:
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=[wanted]
        )
    except Exception as exc:
        # Under `strict` this is where a node placed on the CPU surfaces, and
        # for a provider whose libraries are missing it is where some builds
        # refuse. Either way it is the runtime declining, not the graph.
        if wanted == DEFAULT_PROVIDER:
            raise
        raise ProviderError(f"{wanted} would not take the graph: {exc}") from exc

    if wanted not in session.get_providers():
        # onnxruntime lists a provider it was built with whether or not the
        # driver and libraries behind it are installed, so availability is
        # settled here rather than in `resolve_provider`.
        raise ProviderError(
            f"{wanted} did not load; the session runs on "
            f"{', '.join(session.get_providers())}"
        )
    inputs = {
        name: np.asarray(value, dtype=np.float32) for name, value in bound.items()
    }
    inputs[INPUT] = np.ascontiguousarray(samples, dtype=np.float32)
    try:
        return session.run(None, inputs)[0]
    except Exception as exc:
        # Off the default only. The same graph runs on the CPU, so a failure
        # that appears once it moves is the accelerator's — an unimplemented
        # kernel, or a support library it found no more than a name for.
        # Widening this would rename the compiler's own bugs after a runtime.
        if wanted == DEFAULT_PROVIDER:
            raise
        raise ProviderError(f"{wanted} could not run the graph: {exc}") from exc


def compare(
    want: np.ndarray, got: np.ndarray, samples: np.ndarray | None = None
) -> Comparison:
    """Hold the graph's output against the reference (§spec:verification).

    No sample is dropped (§spec:evidence-floor). Where both sides are finite
    the tolerance decides. Where either is not — some transforms overflow
    float32 at 65504 — there is no magnitude to measure, so the graph shall
    return the class the reference returned: a reference falling to ``-inf``
    where the graph climbs to ``+inf`` is a direction inverted, not agreement.

    Agreement also needs one finite comparison. A reference that is non-finite
    across the whole lattice otherwise certifies a graph on no evidence.
    """
    want = np.asarray(want, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if want.shape != got.shape:
        raise ValueError(f"reference is {want.shape}, graph output is {got.shape}")
    if samples is not None and np.shape(samples) != want.shape:
        # Only the two report paths read it, so a mismatch would otherwise
        # surface as an IndexError on the way to explaining a failure.
        raise ValueError(f"reference is {want.shape}, samples are {np.shape(samples)}")

    want_class = classify(want)
    agree = want_class == classify(got)
    usable = np.flatnonzero(agree & (want_class == FINITE))

    w = want.ravel()[usable]
    g = got.ravel()[usable]
    deviation = np.abs(g - w)
    relative = np.divide(deviation, np.abs(w), out=np.zeros_like(w), where=w != 0)
    bound = TOLERANCE.bound(w)
    over = deviation > bound
    # Each deviation against the bound that governed it, which is what makes
    # one run comparable with another. Read by `worst` to rank the misses and
    # by `margin` to state how close the whole lattice came.
    ratio = deviation / bound

    def input_at(index):
        return float(samples.ravel()[index]) if samples is not None else np.nan

    worst = None
    if over.any():
        k = int(np.argmax(ratio))
        worst = Worst(
            channel=channel_of(usable[k], want.shape),
            index=int(usable[k]),
            value=input_at(usable[k]),
            want=float(w[k]),
            got=float(g[k]),
            absolute=float(deviation[k]),
            relative=float(relative[k]),
        )

    inverted = None
    disagreeing = np.flatnonzero(~agree)
    if disagreeing.size:
        first = int(disagreeing[0])
        inverted = Inverted(
            channel=channel_of(first, want.shape),
            index=first,
            value=input_at(first),
            want=float(want.ravel()[first]),
            got=float(got.ravel()[first]),
        )

    return Comparison(
        compared=usable.size,
        failures=int(over.sum()),
        matched=int((agree & (want_class != FINITE)).sum()),
        disagreed=disagreeing.size,
        max_abs=float(deviation.max()) if deviation.size else 0.0,
        max_rel=float(relative.max()) if relative.size else 0.0,
        worst=worst,
        inverted=inverted,
        margin=float(ratio.max()) if ratio.size else 0.0,
    )


def verify(
    resolved: Resolved,
    model: onnx.ModelProto | None = None,
    provider: str | None = None,
    strict: bool = False,
) -> Comparison:
    """Hold a graph against the oracle, compiling one if none is given.

    A caller holding the model it is about to write hands it over, so what is
    verified is the artifact rather than a second compilation of the request.

    ``provider`` moves only the graph's side of the comparison; the reference
    stays OCIO's CPU processor, and the tolerance stays what §spec:verification
    set. So a sweep that passes on the CPU and fails on an accelerator has
    measured that runtime's arithmetic, not the compiler's reading of the
    config — which is why the CPU remains the default and the claim.
    """
    samples = lattice(resolved.processor)
    return compare(
        cpu_reference(resolved.processor, samples),
        run_graph(
            compile_processor(resolved) if model is None else model,
            samples,
            provider=provider,
            strict=strict,
        ),
        samples,
    )
