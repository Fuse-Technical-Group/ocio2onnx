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
from ocio2onnx.emitters import UnsupportedOpError

#: Re-exported: an emitter refuses a parameter it has no path for, and the
#: refusal a caller catches is the same one whichever raised it.
__all__ = ["UnsupportedOpError", "compile_processor"]


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
