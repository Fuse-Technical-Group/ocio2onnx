"""Emit ONNX arithmetic for one OCIO op (§spec:op-emission).

Every coefficient is read off OCIO's transform introspection; this module
holds no opinion the config does not already state (§spec:non-goals). The
registry keys on the OCIO transform class name, so adding an op is adding an
entry rather than editing the compiler.

Each entry also declares its **breakpoints**: the input values at which the
op switches branches. The oracle's lattice asks every op in a processor for
them and samples either side, because a breakpoint read onto the wrong side
of a comparison is the misreading verification exists to catch
(§spec:verification). ``Matrix`` has none; the branching ops added by
the branching ops widen the lattice by declaring theirs.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ocio2onnx.builder import CHANNELS, GraphBuilder

#: A ``Range`` bound OCIO leaves unset arrives as NaN, and ``Clip`` needs a
#: number. This stands in for "no bound": float16's finite limit is 65504, so
#: nothing a consumer can feed the graph reaches it.
UNBOUNDED = 1e30

#: The floor OCIO's own ``Log`` op holds its argument at, so ``log(0)`` never
#: reaches an output. Measured against the CPU processor: OCIO returns
#: -37.9298 for log10 of any non-positive input, which is log10 of the
#: smallest normal float32. A rounder floor — 1e-30, say — would sit nearly 8
#: units from the reference at the first non-positive input.
LOG_FLOOR = float(np.finfo(np.float32).tiny)

#: How OCIO's negative-value styles are named, with the enum prefix dropped.
MIRROR = "MIRROR"
PASS_THRU = "PASS_THRU"
LINEAR = "LINEAR"

#: A uniform ``Lut1D`` spans this and clamps outside it. OCIO exposes no
#: domain accessor for one because there is nothing to expose: the domain is
#: the unit interval, and the index is a position in it.
LUT1D_DOMAIN = (0.0, 1.0)

#: The smallest normal half, ``2**-14``. At and below it a half's exponent
#: field is zero and its slot is linear in the magnitude; above it the slot is
#: a binade plus a mantissa. Both readings give 1024 at the join, so the index
#: is continuous across it.
HALF_NORMAL_MIN = 2.0**-14

#: The largest finite half. An input above it indexes the top finite slot,
#: which is what OCIO's own shader clamps to.
HALF_MAX = 65504.0

#: Slots per binade: ``2**10``, for a half's ten mantissa bits.
HALF_BINADE = 1024.0

#: What ``floor(log2(x))`` is shifted by to reach a half's exponent field —
#: 15 for the bias, which already counts the denormal binade below it.
HALF_EXPONENT_BIAS = 15.0

#: What the sign bit adds to a slot index.
HALF_SIGN = 32768.0

#: What a denormal magnitude is scaled by to reach its slot: ``2**24``, which
#: is ``HALF_BINADE / HALF_NORMAL_MIN``.
HALF_DENORMAL_SCALE = HALF_BINADE / HALF_NORMAL_MIN

#: Slots in a half-domain table: one per float16 bit pattern.
HALF_SLOT_COUNT = 1 << 16

#: The input value each slot of a half-domain table stands for, which is that
#: table's domain. 2048 of the patterns are infinities and NaNs.
HALF_SLOT_VALUES = (
    np.arange(HALF_SLOT_COUNT, dtype=np.uint16).view(np.float16).astype(np.float64)
)

#: The grid an inverse ``Lut1D`` is sampled onto (§spec:op-emission): the half
#: domain, with the non-finite patterns pulled to the finite limit of their own
#: sign. ``_half_domain_index`` clips a magnitude to ``HALF_MAX``, so nothing
#: reaches those slots, and an entry no input can read is better finite than
#: NaN.
INVERSE_GRID = np.where(
    np.isfinite(HALF_SLOT_VALUES),
    HALF_SLOT_VALUES,
    np.copysign(HALF_MAX, HALF_SLOT_VALUES),
)

#: The finite limit a ``Lut1D`` entry is held at. OCIO reports an entry beyond
#: float32 as an infinity but its CPU processor renders the finite limit —
#: measured on ``Apple Log``, whose decode overflows from slot 18898 up, where
#: the reference interpolates towards this value rather than towards infinity.
#: The graph has to match: a lerp between two infinite entries is ``inf - inf``
#: and would return NaN across the whole top of the domain.
LUT1D_ENTRY_MAX = float(np.finfo(np.float32).max)

#: The interpolations OCIO renders a 1D LUT linearly. ``DEFAULT`` is not a
#: convenience synonym: every 1D LUT OCIO loads from a file reports it, and
#: OCIO resolves it to linear, so refusing it would refuse them all.
LUT1D_INTERPOLATIONS = ("LINEAR", "DEFAULT")

#: The only hue adjustment this compiler emits, which is none.
HUE_NONE = "NONE"


class UnsupportedOpError(NotImplementedError):
    """An op, or a parameter of one, the compiler does not emit.

    One type for both: ``compiler.unsupported_ops`` refuses an op type before
    emission, an emitter refuses a parameter it has no path for during it, and
    a caller catching a refusal should not have to care which raised it.
    """


#: An emitter appends nodes for one transform and returns its output tensor.
Emit = Callable[[GraphBuilder, Any, str], str]

#: A breakpoint hook reports where an op switches branches.
Breakpoints = Callable[[Any], list[float]]


def no_breakpoints(transform: Any) -> list[float]:
    """An op that evaluates one expression everywhere."""
    return []


@dataclasses.dataclass(frozen=True)
class Emitter:
    """What the compiler and the oracle each need from one op."""

    emit: Emit
    breakpoints: Breakpoints = no_breakpoints


#: OCIO transform class name -> emitter.
REGISTRY: dict[str, Emitter] = {}


def register(
    class_name: str, *, breakpoints: Breakpoints = no_breakpoints
) -> Callable[[Emit], Emit]:
    """Register an emitter for an OCIO transform class."""

    def decorate(emit: Emit) -> Emit:
        REGISTRY[class_name] = Emitter(emit=emit, breakpoints=breakpoints)
        return emit

    return decorate


def emitter_for(transform: Any) -> Emitter | None:
    """The emitter for a transform, or ``None`` if the compiler has none."""
    return REGISTRY.get(type(transform).__name__)


def supported_ops() -> frozenset[str]:
    """The op types this compiler emits, as bare names.

    Read off the registry, so it is one source of truth: an op OCIO adds in a
    future release is absent from it and refused, rather than needing a second
    list to be updated before anyone notices (§spec:op-coverage).
    """
    return frozenset(name.removesuffix("Transform") for name in REGISTRY)


def op_name(transform: Any) -> str:
    """One op's bare type name, as the registry and the census spell it."""
    return type(transform).__name__.removesuffix("Transform")


def _enum_member(value: Any, prefix: str) -> str:
    """An OCIO enum member's bare name. ``str`` gives ``Class.MEMBER``, and the
    member repeats its class in the prefix callers drop here."""
    return str(value).rsplit(".", 1)[-1].removeprefix(prefix)


#: Op types whose behaviour is set by an attribute rather than by the type
#: alone, and the reading that names it. ``FixedFunction`` is two unrelated
#: transforms sharing a class — ``REC2100_SURROUND`` and
#: ``ACES_OUTPUT_TRANSFORM_20`` differ by an order of magnitude in cost and in
#: risk, and are separate workstreams — so a refusal naming only the type
#: cannot say which one blocked the caller (§spec:op-coverage).
DISTINGUISHING: dict[str, Callable[[Any], str]] = {
    "FixedFunctionTransform": lambda transform: _enum_member(
        transform.getStyle(), "FIXED_FUNCTION_"
    ),
}


def op_label(transform: Any) -> str:
    """How a refusal names one op.

    An op type with a distinguishing attribute names it; one without names the
    type alone.
    """
    op = op_name(transform)
    attribute = DISTINGUISHING.get(type(transform).__name__)
    return f"{op}[{attribute(transform)}]" if attribute is not None else op


def breakpoints(transform: Any) -> list[float]:
    """Where this transform switches branches. Empty for an unregistered op:
    the lattice widens where it can and the compiler refuses where it cannot.
    """
    entry = emitter_for(transform)
    return entry.breakpoints(transform) if entry is not None else []


def _is_inverse(transform: Any) -> bool:
    """Whether OCIO hands this op over to be run backwards.

    Most of the closed-form ops arrive inverse, so this is the common case
    rather than the exotic one (§spec:op-coverage).
    """
    return str(transform.getDirection()).endswith("INVERSE")


def _channels(values: Any) -> list[float]:
    """The first three of an OCIO per-channel parameter.

    OCIO reports four; alpha bypasses the graph (§spec:emitted-graph), so the
    fourth is dropped here rather than carried through the arithmetic.
    """
    return [float(value) for value in values][:CHANNELS]


def _negative_style(transform: Any, supported: tuple[str, ...]) -> str:
    """How this op treats inputs below zero, refused if unimplemented.

    OCIO's four styles differ only below zero, which is precisely where they
    are hardest to notice being wrong. Treating an unrecognised one as the
    nearest implemented style would emit a graph that disagrees with the
    config over exactly the inputs the style exists to govern, so it is
    refused instead (§spec:op-coverage).
    """
    style = _enum_member(transform.getNegativeStyle(), "NEGATIVE_")
    if style not in supported:
        op = op_name(transform)
        raise UnsupportedOpError(
            f"{op} negative style {style} is not emitted by this compiler; "
            f"it emits {', '.join(supported)}"
        )
    return style


@register("MatrixTransform")
def emit_matrix(builder: GraphBuilder, transform: Any, x: str) -> str:
    """A 1x1 convolution over the top-left 3x3 (§spec:op-emission).

    OCIO hands over a row-major 4x4 and a 4-offset. The alpha row is identity
    in every transform in the pinned config, and alpha bypasses the graph
    anyway (§spec:emitted-graph), so only the 3x3 and the first three offsets
    reach the weight and the bias.

    An inverse transform inverts here, in float64, and casts to float32 only
    at the initializer: inverting a near-singular matrix at float32 would
    lose digits the oracle would then report as the compiler's error.
    """
    matrix = np.array(transform.getMatrix(), dtype=np.float64).reshape(4, 4)
    offset = np.array(transform.getOffset(), dtype=np.float64)

    if _is_inverse(transform):
        matrix = np.linalg.inv(matrix)
        offset = -(matrix @ offset)

    weight = builder.constant(
        "matrix_weight", matrix[:CHANNELS, :CHANNELS].reshape(CHANNELS, CHANNELS, 1, 1)
    )
    bias = builder.constant("matrix_bias", offset[:CHANNELS])
    return builder.op("Conv", [x, weight, bias], kernel_shape=[1, 1])


def range_bounds(transform: Any) -> tuple[float, float, float, float]:
    """``(min_in, max_in, min_out, max_out)``, with the direction applied.

    An inverse ``Range`` maps its output interval back onto its input one, so
    the two pairs swap. Any of the four may be NaN, which is how OCIO reports
    a bound the config left unset; that is a half-open clamp, not an error.
    """
    bounds = (
        float(transform.getMinInValue()),
        float(transform.getMaxInValue()),
        float(transform.getMinOutValue()),
        float(transform.getMaxOutValue()),
    )
    if _is_inverse(transform):
        return bounds[2], bounds[3], bounds[0], bounds[1]
    return bounds


def range_breakpoints(transform: Any) -> list[float]:
    """The clamp bounds, in the input domain."""
    min_in, max_in, _, _ = range_bounds(transform)
    return [value for value in (min_in, max_in) if not math.isnan(value)]


@register("RangeTransform", breakpoints=range_breakpoints)
def emit_range(builder: GraphBuilder, transform: Any, x: str) -> str:
    """A clamp, then an affine (§spec:op-emission).

    The affine maps the clamped input interval onto the output one, and is
    emitted only when all four bounds are set: with a bound missing there is
    no interval to map, and OCIO's op is the clamp alone. It is also skipped
    when it is the identity, which is every ``Range`` in the pinned config bar
    one pair.

    In that config ``Range`` always arrives forward and always clamps — OCIO
    folds the inverse — but both directions emit. There is no non-clamping
    branch because there is nothing to emit one for: OCIO rewrites a
    ``RANGE_NO_CLAMP`` op into a ``Matrix`` before a processor reports it.
    """
    min_in, max_in, min_out, max_out = range_bounds(transform)

    low = -UNBOUNDED if math.isnan(min_in) else min_in
    high = UNBOUNDED if math.isnan(max_in) else max_in
    y = builder.op(
        "Clip",
        [x, builder.scalar("range_low", low), builder.scalar("range_high", high)],
    )

    if any(math.isnan(bound) for bound in (min_in, max_in, min_out, max_out)):
        return y

    scale = (max_out - min_out) / (max_in - min_in)
    offset = min_out - min_in * scale
    if scale == 1.0 and offset == 0.0:
        return y
    return builder.add(
        builder.mul(y, builder.scalar("range_scale", scale)),
        builder.scalar("range_offset", offset),
    )


def exponent_breakpoints(transform: Any) -> list[float]:
    """Zero, where the negative-value handling switches."""
    return [0.0]


@register("ExponentTransform", breakpoints=exponent_breakpoints)
def emit_exponent(builder: GraphBuilder, transform: Any, x: str) -> str:
    """A power, with the sign put back the way the style asks.

    The inverse of ``x**g`` is ``x**(1/g)``, so an inverse op reciprocates the
    gamma and takes the same path.
    """
    gamma = _channels(transform.getValue())
    if _is_inverse(transform):
        gamma = [1.0 / value for value in gamma]
    style = _negative_style(transform, (MIRROR, PASS_THRU))
    return _signed_power(
        builder, x, builder.per_channel("exponent_gamma", gamma), style
    )


def _signed_power(builder: GraphBuilder, x: str, exponent: str, style: str) -> str:
    """``|x| ** g`` with the negative half restored per ``style``.

    ONNX ``Pow`` yields NaN for a negative base, and ``Where`` evaluates both
    branches, so a raw ``Pow`` would poison the result whichever branch is
    selected. Taking ``Abs`` first keeps the base non-negative; the sign is
    put back afterwards.
    """
    magnitude = builder.pow(builder.op("Abs", [x]), exponent)
    if style == MIRROR:
        return builder.mul(builder.op("Sign", [x]), magnitude)
    below = builder.op("Less", [x, builder.scalar("zero", 0.0)])
    return builder.where(below, x, magnitude)


@dataclasses.dataclass(frozen=True)
class MonCurve:
    """Where a monitor curve meets its linear segment, and that segment's slope.

    Above the break ``y = ((x + off)/(1 + off))**g``; below it ``y = k*x``.
    """

    x_break: float
    y_break: float
    slope: float


def moncurve(gamma: float, offset: float) -> MonCurve:
    """The C1-continuous join between the curve and its linear segment.

    ``slope`` is the curve's derivative at the break, which is what makes the
    join C1 rather than merely continuous. Do not re-derive it: a plausible
    closed form for it is wrong by a factor of around 50, and nothing but the
    oracle reports that — the values either side of the break stay close
    enough to look right.

    A zero offset degenerates. It never occurs in the pinned config, measured
    across every ``ExponentWithLinear`` op in it, so it is refused rather than
    approximated by a plain ``Exponent``.
    """
    if offset == 0.0:
        raise UnsupportedOpError(
            "ExponentWithLinear with a zero offset has no linear segment and "
            "no break; this compiler does not emit one"
        )
    x_break = offset / (gamma - 1.0)
    slope = (gamma / (1.0 + offset)) * ((x_break + offset) / (1.0 + offset)) ** (
        gamma - 1.0
    )
    return MonCurve(x_break=x_break, y_break=slope * x_break, slope=slope)


def _moncurves(transform: Any) -> list[MonCurve]:
    """One join per channel."""
    return [
        moncurve(gamma, offset)
        for gamma, offset in zip(
            _channels(transform.getGamma()),
            _channels(transform.getOffset()),
            strict=True,
        )
    ]


def exponent_with_linear_breakpoints(transform: Any) -> list[float]:
    """Zero, plus the break — which lies in the encoded domain when the op
    arrives inverse, not in the linear one."""
    inverse = _is_inverse(transform)
    return [0.0] + [
        curve.y_break if inverse else curve.x_break for curve in _moncurves(transform)
    ]


@register("ExponentWithLinearTransform", breakpoints=exponent_with_linear_breakpoints)
def emit_exponent_with_linear(builder: GraphBuilder, transform: Any, x: str) -> str:
    """A monitor curve above the break, a straight line below it.

    ``MIRROR`` runs the whole thing on ``|x|`` and puts the sign back;
    ``LINEAR`` extends the linear segment through the origin instead, so its
    negative half is a straight line rather than a reflected curve.

    Each ``Pow`` takes a clipped base. That is not defensive decoration:
    ``Where`` evaluates both branches, and the branch it discards would
    otherwise raise a negative number to a fractional power and hand back NaN.
    """
    curves = _moncurves(transform)
    gamma = _channels(transform.getGamma())
    offset = _channels(transform.getOffset())
    style = _negative_style(transform, (MIRROR, LINEAR))

    source = builder.op("Abs", [x]) if style == MIRROR else x
    zero = builder.scalar("zero", 0.0)
    slope = builder.per_channel("moncurve_slope", [curve.slope for curve in curves])
    offsets = builder.per_channel("moncurve_offset", offset)
    scaled = builder.per_channel(
        "moncurve_denominator", [1.0 + value for value in offset]
    )

    inverse = _is_inverse(transform)
    # The break lies in the encoded domain when the op arrives inverse and in
    # the linear one when it does not, as ``exponent_with_linear_breakpoints``
    # declares.
    breaks = builder.per_channel(
        "moncurve_break",
        [curve.y_break if inverse else curve.x_break for curve in curves],
    )

    if inverse:
        linear = builder.div(source, slope)
        curved = builder.sub(
            builder.mul(
                builder.pow(
                    builder.op("Clip", [source, zero]),
                    builder.per_channel(
                        "moncurve_reciprocal_gamma", [1.0 / value for value in gamma]
                    ),
                ),
                scaled,
            ),
            offsets,
        )
    else:
        linear = builder.mul(source, slope)
        curved = builder.pow(
            builder.op(
                "Clip", [builder.div(builder.add(source, offsets), scaled), zero]
            ),
            builder.per_channel("moncurve_gamma", gamma),
        )

    y = builder.where(builder.op("Less", [source, breaks]), linear, curved)
    return builder.mul(builder.op("Sign", [x]), y) if style == MIRROR else y


def log_breakpoints(transform: Any) -> list[float]:
    """Zero, where the forward clamp engages. The inverse evaluates one
    expression over the whole line and declares none."""
    return [] if _is_inverse(transform) else [0.0]


@register("LogTransform", breakpoints=log_breakpoints)
def emit_log(builder: GraphBuilder, transform: Any, x: str) -> str:
    """``log(x)/log(base)`` forward, ``base ** x`` inverse.

    The forward clamp is read off OCIO rather than chosen: OCIO's own op holds
    its argument at ``LOG_FLOOR``, so an input at or below zero leaves with a
    definite value rather than an infinity. A rounder floor would sit whole
    units away from the reference at the first non-positive input.
    """
    base = float(transform.getBase())
    if _is_inverse(transform):
        return builder.pow(builder.scalar("log_base", base), x)
    argument = builder.op("Clip", [x, builder.scalar("log_floor", LOG_FLOOR)])
    return builder.mul(
        builder.op("Log", [argument]),
        builder.scalar("log_scale", 1.0 / math.log(base)),
    )


@dataclasses.dataclass(frozen=True)
class CameraCurve:
    """One parametric camera log curve, per channel, with its break resolved.

    ``LogCamera`` is a single op; RED Log3G10, ARRI LogC3 and LogC4, ACEScct,
    BMDFilm, D-Log, V-Log, DaVinci Intermediate, and the S-Log3 family are all
    it with different coefficients (§spec:op-coverage). Supporting another
    camera is a config update, not a change here.
    """

    base: float
    log_slope: list[float]
    log_offset: list[float]
    lin_slope: list[float]
    lin_offset: list[float]
    lin_break: list[float]
    linear_slope: list[float]
    log_break: list[float]
    linear_offset: list[float]


def camera_curve(transform: Any) -> CameraCurve:
    """Read a ``LogCamera``'s parameters and resolve its break.

    ``linear_slope`` is the one parameter OCIO may not supply. It reports an
    unset slope as NaN rather than as an absent value — 14 of the 24 ops in
    the pinned config — so "unset" is tested per channel rather than on the
    container. Taken at face value the NaN yields a graph that is NaN below
    the break and finite above it.

    Where it is unset, it is derived for C1 continuity: the linear segment
    leaves the break at the rate the logarithm arrives at it. ``log_break``
    and ``linear_offset`` then place that segment so the join has no step.
    """
    base = float(transform.getBase())
    ln_base = math.log(base)
    log_slope = _channels(transform.getLogSideSlopeValue())
    log_offset = _channels(transform.getLogSideOffsetValue())
    lin_slope = _channels(transform.getLinSideSlopeValue())
    lin_offset = _channels(transform.getLinSideOffsetValue())
    lin_break = _channels(transform.getLinSideBreakValue())

    declared = transform.getLinearSlopeValue()
    declared = _channels(declared) if declared else [math.nan] * CHANNELS

    linear_slope = []
    log_break = []
    linear_offset = []
    for i in range(CHANNELS):
        argument = lin_slope[i] * lin_break[i] + lin_offset[i]
        slope = declared[i]
        if math.isnan(slope):
            slope = log_slope[i] * lin_slope[i] / (argument * ln_base)
        linear_slope.append(slope)
        log_break.append(log_slope[i] * math.log(argument) / ln_base + log_offset[i])
        linear_offset.append(log_break[i] - slope * lin_break[i])

    return CameraCurve(
        base=base,
        log_slope=log_slope,
        log_offset=log_offset,
        lin_slope=lin_slope,
        lin_offset=lin_offset,
        lin_break=lin_break,
        linear_slope=linear_slope,
        log_break=log_break,
        linear_offset=linear_offset,
    )


def log_camera_breakpoints(transform: Any) -> list[float]:
    """The break, which lies in the encoded domain when the op arrives
    inverse and in the linear one when it does not."""
    curve = camera_curve(transform)
    return curve.log_break if _is_inverse(transform) else curve.lin_break


@register("LogCameraTransform", breakpoints=log_camera_breakpoints)
def emit_log_camera(builder: GraphBuilder, transform: Any, x: str) -> str:
    """A logarithm above the break, a straight line below it."""
    curve = camera_curve(transform)
    log_offset = builder.per_channel("logcamera_log_offset", curve.log_offset)
    lin_offset = builder.per_channel("logcamera_lin_offset", curve.lin_offset)
    lin_slope = builder.per_channel("logcamera_lin_slope", curve.lin_slope)
    linear_slope = builder.per_channel("logcamera_linear_slope", curve.linear_slope)
    linear_offset = builder.per_channel("logcamera_linear_offset", curve.linear_offset)

    inverse = _is_inverse(transform)
    # The break lies in the encoded domain when the op arrives inverse and in
    # the linear one when it does not, as ``log_camera_breakpoints`` declares.
    breaks = builder.per_channel(
        "logcamera_break", curve.log_break if inverse else curve.lin_break
    )

    if inverse:
        exponent = builder.div(
            builder.sub(x, log_offset),
            builder.per_channel("logcamera_log_slope", curve.log_slope),
        )
        curved = builder.div(
            builder.sub(
                builder.pow(builder.scalar("logcamera_base", curve.base), exponent),
                lin_offset,
            ),
            lin_slope,
        )
        linear = builder.div(builder.sub(x, linear_offset), linear_slope)
    else:
        argument = builder.op(
            "Clip",
            [
                builder.add(builder.mul(x, lin_slope), lin_offset),
                builder.scalar("log_floor", LOG_FLOOR),
            ],
        )
        curved = builder.add(
            builder.mul(
                builder.op("Log", [argument]),
                builder.per_channel(
                    "logcamera_log_scale",
                    [value / math.log(curve.base) for value in curve.log_slope],
                ),
            ),
            log_offset,
        )
        linear = builder.add(builder.mul(x, linear_slope), linear_offset)

    return builder.where(builder.op("Less", [x, breaks]), linear, curved)


def lut1d_table(transform: Any) -> np.ndarray:
    """One ``Lut1D``'s entries, as ``(length, 3)`` float32.

    OCIO reports a triple per entry even where the three curves coincide,
    which they do for all 40 ops in the pinned config. The emitter reads them
    and compares rather than assuming it: a per-channel table is a shape
    another config can take (§spec:non-goals).

    An entry past float32's range is clamped to ``LUT1D_ENTRY_MAX`` rather
    than left at the infinity OCIO reports, because that is where OCIO's CPU
    processor holds it.
    """
    values = np.asarray(transform.getData()).reshape(transform.getLength(), CHANNELS)
    # ``clip`` allocates, so OCIO's own buffer is read rather than aliased.
    return np.clip(values, -LUT1D_ENTRY_MAX, LUT1D_ENTRY_MAX)


def _is_half_indexed(transform: Any) -> bool:
    """Whether the graph reads this op through the half index.

    A forward half-domain table does. So does an inverse one whatever domain
    it was written over, because ``lut1d_inverse_table`` puts it on the half
    domain. ``lut1d_breakpoints`` and ``emit_lut1d`` have to agree on this:
    were they to disagree, the lattice would straddle branches the graph does
    not have and miss the ones it does, and the oracle cannot report a branch
    it was never told to sample.
    """
    return bool(transform.getInputHalfDomain()) or _is_inverse(transform)


def _lut1d_domain(transform: Any) -> np.ndarray:
    """The input value each of a table's entries stands for.

    A uniform table's is a position in the unit interval, which is how
    ``_uniform_index`` reads one. A half domain's is the value of the slot's
    own bit pattern.
    """
    if transform.getInputHalfDomain():
        return HALF_SLOT_VALUES
    return np.linspace(*LUT1D_DOMAIN, transform.getLength())


def _refuse_falling_lut1d(domain: np.ndarray, values: np.ndarray) -> None:
    """Refuse a table with no inverse this reading can take.

    Read over strictly increasing domain steps only. ``+0.0`` and ``-0.0``
    compare equal, so a half-domain table holds two entries at domain zero and
    a plain difference reads the step between them as a fall — which is what
    both PQ tables would refuse on, though they rise across every step their
    domain actually takes.
    """
    if (np.diff(values)[np.diff(domain) > 0] >= 0.0).all():
        return
    raise UnsupportedOpError(
        "an inverse Lut1D whose table does not rise is not emitted by this "
        "compiler; it inverts a non-decreasing table"
    )


def _invert_channel(domain: np.ndarray, values: np.ndarray) -> np.ndarray:
    """One channel's forward curve, read backwards onto ``INVERSE_GRID``.

    Pairs with a non-finite side drop out, which is the 2048 half patterns
    that are not values. ``Apple Log``'s decode is not among them: it
    overflows float32 from slot 18898 up, and ``lut1d_table`` holds an
    overflowing entry at ``LUT1D_ENTRY_MAX``, so it arrives here as a flat run
    rather than as missing pairs.

    Where the table is flat the inverse is ambiguous across a whole interval,
    so each run of equal values collapses to the domain the curve *leaves* it
    at — the lowest run's largest domain, every other run's smallest. What is
    left is the interval the table is invertible over, and outside it
    ``np.interp`` clamps to the same two endpoints OCIO does. Measured rather
    than reasoned: OCIO answers -65504 with 0 for ``ref -> Apple Log``, whose
    decode is flat across [-65504, 0], and 65504 with 3.7597656 for
    ``ref -> ADX10``, whose is flat across [3.7597656, 65504]. Taking the same
    end of every run puts one of those two half a domain out.
    """
    finite = np.isfinite(domain) & np.isfinite(values)
    if not finite.any():
        # A table of NaNs is reachable from a config file, and there is no
        # curve under it to read backwards. Refused rather than left to raise
        # an IndexError out of the arithmetic below.
        raise UnsupportedOpError(
            "an inverse Lut1D whose table holds no finite entry is not "
            "emitted by this compiler; there is no curve to invert"
        )
    order = np.argsort(domain[finite], kind="stable")
    d, v = domain[finite][order], values[finite][order]
    _refuse_falling_lut1d(d, v)

    starts = np.flatnonzero(np.concatenate([[True], np.diff(v) > 0.0]))
    keep = starts.copy()
    keep[0] = starts[1] - 1 if starts.size > 1 else v.size - 1
    return np.interp(INVERSE_GRID, v[keep], d[keep])


def lut1d_inverse_table(transform: Any) -> np.ndarray:
    """A monotonic ``Lut1D``, inverted at compile time (§spec:op-emission).

    The result is an ordinary forward half-domain table, so the graph reads it
    with the gather and lerp it already has and searches nothing at run time.

    The grid is the half domain whichever domain the forward table used. Its
    spacing is geometric, so it holds the same relative resolution in the toe
    as at the top — which is the shape a log-encoded curve has. A uniform grid
    over the forward table's output range does not: those ranges are enormous
    (``ACEScc`` spans [-5.7e-07, 96617.7]), so a uniform grid spends its
    samples above middle grey and leaves the toe to a single interval
    (§spec:op-emission).
    """
    domain = _lut1d_domain(transform)
    table = lut1d_table(transform)
    return np.stack(
        [
            _invert_channel(domain, table[:, channel].astype(np.float64))
            for channel in range(CHANNELS)
        ],
        axis=1,
    ).astype(np.float32)


def _refuse_unemitted_lut1d(transform: Any) -> None:
    """Refuse the ``Lut1D`` shapes this compiler has no path for.

    Each would change the result rather than degrade it. All three redefine
    what a slot means, and none occurs in the pinned config — so, following
    ``_negative_style``, an unrecognised one is refused rather than
    approximated by the reading this emitter does implement.
    """
    interpolation = _enum_member(transform.getInterpolation(), "INTERP_")
    if interpolation not in LUT1D_INTERPOLATIONS:
        raise UnsupportedOpError(
            f"Lut1D interpolation {interpolation} is not emitted by this "
            f"compiler; it emits {', '.join(LUT1D_INTERPOLATIONS)}"
        )
    hue_adjust = _enum_member(transform.getHueAdjust(), "HUE_")
    if hue_adjust != HUE_NONE:
        raise UnsupportedOpError(
            f"Lut1D hue adjust {hue_adjust} is not emitted by this compiler; "
            "it emits an unadjusted table"
        )
    if transform.getOutputRawHalfs():
        raise UnsupportedOpError(
            "Lut1D with raw half-float entries is not emitted by this "
            "compiler; it emits the float32 entries OCIO reports"
        )


def lut1d_breakpoints(transform: Any) -> list[float]:
    """Where the index switches branches.

    A uniform table's are its domain edges, where the clamp engages. A half
    domain has no such edges — it spans the float16 line rather than the unit
    interval — and its branches are elsewhere: zero, where the sign offset
    switches, and ``±2**-14``, where the denormal reading joins the normal one.

    An inverse op reaches the graph as a half-domain table whatever domain it
    was written over (``lut1d_inverse_table``), so its branches are the half
    index's rather than the forward table's.

    Not the slot boundaries in between: there are tens of thousands, they would
    swamp the lattice, and the lerp is continuous across every one. A
    breakpoint is declared where a misreading would put a sample on the wrong
    branch, and a slot boundary has no branch.
    """
    if _is_half_indexed(transform):
        return [0.0, HALF_NORMAL_MIN, -HALF_NORMAL_MIN]
    return list(LUT1D_DOMAIN)


def _uniform_index(
    builder: GraphBuilder, x: str, length: int
) -> tuple[str, str | None]:
    """Where an input falls in a uniform table, as OCIO indexes one:
    ``clip(x, 0, 1) * (length - 1)``. No slot offset — the position is the
    whole index."""
    position = builder.mul(
        builder.op(
            "Clip",
            [
                x,
                builder.scalar("lut1d_low", LUT1D_DOMAIN[0]),
                builder.scalar("lut1d_high", LUT1D_DOMAIN[1]),
            ],
        ),
        builder.scalar("lut1d_scale", float(length - 1)),
    )
    return position, None


def _half_domain_index(builder: GraphBuilder, x: str) -> tuple[str, str]:
    """Where an input falls in a half-domain table, as a slot offset and a
    position within a binade (§spec:op-emission).

    A half-domain table is indexed by the bit pattern of the input rounded to
    float16. Standard ONNX has no bit-reinterpreting cast, so the index is
    reconstructed in float arithmetic — which is what OCIO's own shaders do,
    for the same reason: they target GLSL 1.2 and ES 1.0, which have no
    bitwise integer ops.

    The reconstruction is OCIO's ``computePos``. Above ``2**-14`` a half's
    slot is ``1024 * (exponent + 15) + 1024 * mantissa``; at and below it the
    half is denormal and its slot is ``magnitude * 2**24``. Both give 1024 at
    the join, so the two readings meet without a step.

    Two details are not decoration:

    The ``Log`` argument is clipped to ``[2**-14, 65504]``. ``Where`` selects
    rather than branches, so the discarded normal reading is still evaluated
    at zero, where ``log(0)`` is ``-inf``, ``2**-inf`` is 0, and ``0/0`` is
    NaN. Clipping makes it finite everywhere, as ``emit_exponent_with_linear``
    clips its ``Pow`` bases.

    The binade and the sign are returned as a slot offset rather than folded
    into the position. Both are exact integers below ``2**16``, so their sum
    is exact in float32 and the fraction that drives the lerp keeps its full
    precision. Added before the floor, a value near 64511 would resolve only
    to about 1/256, quantizing the fraction to a few thousandths of a slot.
    """
    zero = builder.scalar("zero", 0.0)
    magnitude = builder.op("Abs", [x])
    normal_min = builder.scalar("half_normal_min", HALF_NORMAL_MIN)
    normal = builder.op("Greater", [magnitude, normal_min])

    clipped = builder.op(
        "Clip", [magnitude, normal_min, builder.scalar("half_max", HALF_MAX)]
    )
    exponent = builder.op(
        "Floor",
        [
            builder.mul(
                builder.op("Log", [clipped]),
                builder.scalar("half_log2", 1.0 / math.log(2.0)),
            )
        ],
    )
    binade_low = builder.pow(builder.scalar("two", 2.0), exponent)
    mantissa = builder.div(builder.sub(clipped, binade_low), binade_low)
    slots = builder.scalar("half_binade", HALF_BINADE)

    position = builder.where(
        normal,
        builder.mul(mantissa, slots),
        builder.mul(
            magnitude, builder.scalar("half_denormal_scale", HALF_DENORMAL_SCALE)
        ),
    )
    binade = builder.where(
        normal,
        builder.mul(
            builder.add(
                exponent, builder.scalar("half_exponent_bias", HALF_EXPONENT_BIAS)
            ),
            slots,
        ),
        zero,
    )
    # OCIO reads the sign bit, so -0.0 indexes the negative half of the table
    # and ``Less(x, 0)`` would not put it there. A value and its reciprocal
    # share a sign bit and never both round to zero, so the smaller of the two
    # is negative exactly when the sign bit is set: at -0.0 the reciprocal is
    # -inf, and where the reciprocal is denormal enough for an executor to
    # flush it, the input itself still carries the sign.
    signed = builder.op("Min", [x, builder.op("Reciprocal", [x])])
    sign = builder.where(
        builder.op("Less", [signed, zero]),
        builder.scalar("half_sign", HALF_SIGN),
        zero,
    )
    return position, builder.to_int64(builder.add(binade, sign))


@register("Lut1DTransform", breakpoints=lut1d_breakpoints)
def emit_lut1d(builder: GraphBuilder, transform: Any, x: str) -> str:
    """A gather and a lerp over a table (§spec:op-emission).

    The two domains differ only in how the index is arrived at, so each
    computes one and the gather is shared. An index is a slot offset and a
    position: the position is floored to a slot and interpolated across it,
    because OCIO's CPU processor interpolates rather than snapping to the
    nearer slot, and the offset is added afterwards in integer space.

    The upper index is ``i0 + (1 if frac > 0 else 0)`` rather than
    ``min(i0 + 1, length - 1)``. At ``x >= 1`` the uniform index is the last
    slot and ``i0 + 1`` would read past the end.

    ``Gather`` over a rank-1 table with rank-4 indices returns the indices'
    shape, so the whole op runs at ``(N, 3, H, W)`` with no reshaping. Three
    coinciding channels share one table; three that disagree flatten into one
    channel-major table read through a per-channel base offset, so both paths
    are the same gather.

    A half-domain table is emitted flat, all 65536 entries of it. OCIO's GPU
    path folds the index into a 4096-by-17 texture because texture widths cap
    below 65536; an ONNX initializer has no such limit, so neither the packing
    nor its stride carries over.

    An inverse op inverts here, once, and emits as a forward half-domain table
    (``lut1d_inverse_table``). Direction costs no second code path in the
    graph and no search at run time.

    A NaN input reads the first slot. It has to be handled rather than left to
    the arithmetic: ``Cast(NaN, INT64)`` is implementation-defined, yields
    ``INT64_MIN`` on x86, and would reach ``Gather`` as an index no bound
    covers. NaN is not exotic in the pixels this graph runs on, and it arrives
    from inside the graph too — an upstream ``Matrix`` overflowing to ``±inf``
    yields ``inf - inf``.

    The first slot is where OCIO reads a NaN over a uniform table. Over a half
    domain OCIO reads the NaN's own float16 pattern, slot 32256 — a different
    rule that lands on the same value, because every half-domain table in the
    pinned config holds ``table[32256] == table[0]``. Measured across all 14
    distinct tables in that config, against four NaN payloads and both
    infinities, the two readings agree everywhere. A config whose half-domain
    table disagreed at that slot would need the pattern read rather than the
    slot assumed.
    """
    _refuse_unemitted_lut1d(transform)
    inverse = _is_inverse(transform)
    table = lut1d_inverse_table(transform) if inverse else lut1d_table(transform)
    length = table.shape[0]

    if _is_half_indexed(transform):
        position, offset = _half_domain_index(builder, x)
    else:
        position, offset = _uniform_index(builder, x, length)

    zero = builder.scalar("zero", 0.0)
    position = builder.where(builder.op("IsNaN", [position]), zero, position)
    floor = builder.op("Floor", [position])
    frac = builder.sub(position, floor)

    lower = builder.to_int64(floor)
    if offset is not None:
        lower = builder.add(lower, offset)
    upper = builder.add(lower, builder.to_int64(builder.op("Greater", [frac, zero])))

    if (table == table[:, :1]).all():
        data = builder.constant("lut1d_table", table[:, 0])
        entries = length
    else:
        # Channel-major, so channel ``c`` occupies ``[c * length, ...)`` and a
        # constant offset per channel turns a shared index into its own.
        data = builder.constant("lut1d_table", table.T.reshape(-1))
        entries = length * CHANNELS
        base = builder.per_channel(
            "lut1d_channel_base",
            [channel * length for channel in range(CHANNELS)],
            dtype=np.int64,
        )
        lower = builder.add(lower, base)
        upper = builder.add(upper, base)

    # Both indices are in range by construction once NaN is out of the way.
    # Clamped anyway, so that holds locally rather than resting on an argument
    # spanning the whole index. An out-of-range ``Gather`` is undefined, and
    # the executor is the consumer's, not this compiler's.
    lower, upper = (
        builder.op(
            "Clip",
            [
                index,
                builder.constant("lut1d_first", 0, dtype=np.int64),
                builder.constant("lut1d_last", entries - 1, dtype=np.int64),
            ],
        )
        for index in (lower, upper)
    )

    low = builder.op("Gather", [data, lower], axis=0)
    high = builder.op("Gather", [data, upper], axis=0)
    return builder.add(low, builder.mul(frac, builder.sub(high, low)))
