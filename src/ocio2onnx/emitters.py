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
§road:closed-form-emitters widen the lattice by declaring theirs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from ocio2onnx.addressing import CHANNELS
from ocio2onnx.builder import GraphBuilder

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


def breakpoints(transform: Any) -> list[float]:
    """Where this transform switches branches. Empty for an unregistered op:
    the lattice widens where it can and the compiler refuses where it cannot.
    """
    entry = emitter_for(transform)
    return entry.breakpoints(transform) if entry is not None else []


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

    if str(transform.getDirection()).endswith("INVERSE"):
        matrix = np.linalg.inv(matrix)
        offset = -(matrix @ offset)

    weight = builder.constant(
        "matrix_weight", matrix[:CHANNELS, :CHANNELS].reshape(CHANNELS, CHANNELS, 1, 1)
    )
    bias = builder.constant("matrix_bias", offset[:CHANNELS])
    return builder.op("Conv", [x, weight, bias], kernel_shape=[1, 1])
