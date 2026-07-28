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
