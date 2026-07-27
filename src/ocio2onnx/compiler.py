"""Walk a resolved processor and emit one ONNX graph (§road:compiler-core).

OCIO reports a processor's ops before anything executes, so an op the
compiler does not implement fails here rather than emitting a graph that is
quietly wrong (§spec:op-coverage).
"""

from __future__ import annotations

import onnx

from ocio2onnx import emitters
from ocio2onnx.addressing import Resolved
from ocio2onnx.builder import INPUT, GraphBuilder


class UnsupportedOpError(NotImplementedError):
    """An op the compiler does not emit.

    Raised at the op that stopped emission. §road:named-refusal replaces this
    with a pre-emission check that names the transform and its endpoints too.
    """


def compile_processor(resolved: Resolved) -> onnx.ModelProto:
    """Compile a resolved request to a graph, or refuse naming the op."""
    builder = GraphBuilder()
    tensor = INPUT

    for transform in resolved.processor.createGroupTransform():
        emitter = emitters.emitter_for(transform)
        if emitter is None:
            op = type(transform).__name__.removesuffix("Transform")
            raise UnsupportedOpError(
                f"{op} is not emitted by this compiler, so {resolved.endpoints} "
                "is refused rather than approximated"
            )
        tensor = emitter.emit(builder, transform, tensor)

    return builder.build(tensor, resolved.endpoints, resolved.metadata)
