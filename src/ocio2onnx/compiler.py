"""Walk a resolved processor and emit one ONNX graph (§spec:op-coverage).

OCIO reports a processor's ops before anything executes, so the compiler reads
the whole op list first and refuses there. Refusing
during emission would name whichever unimplemented op came first and leave a
half-built graph behind; refusing before it names them all, and the answer
does not depend on op order (§spec:op-coverage).
"""

from __future__ import annotations

import onnx

from ocio2onnx import emitters
from ocio2onnx.addressing import Resolved
from ocio2onnx.builder import INPUT, GraphBuilder
from ocio2onnx.emitters import UnsupportedOpError

#: Re-exported: an emitter refuses a parameter it has no path for, and the
#: refusal a caller catches is the same one whichever raised it.
__all__ = ["UnsupportedOpError", "compile_processor", "op_names", "unsupported_ops"]


def op_names(processor) -> list[str]:
    """The ops a processor is built from, in order, as this compiler names them.

    ``op_label``'s naming rather than the bare class name, so an op type
    carrying two unrelated transforms is counted as the two it is. The census
    marks a name emitted or refused against the same set the compiler selects
    an emitter from, and a type half of which is emitted cannot be marked
    either way (§spec:op-coverage).
    """
    return [emitters.op_label(t) for t in processor.createGroupTransform()]


def unsupported_ops(processor) -> list[str]:
    """The ops this compiler will not emit, named, in the order they appear.

    Empty when the transform compiles, so a caller — the CLI, the census —
    asks whether something would compile without compiling it.

    The test is membership of ``emitters.REGISTRY``, which is where an
    implemented op declares itself. Written that way round, an op type a
    future OCIO release adds is refused because nothing registered an emitter
    for it, rather than dropped because no list named it (§spec:op-coverage).
    """
    refused: list[str] = []
    for transform in processor.createGroupTransform():
        if emitters.emitter_for(transform) is None:
            name = emitters.op_label(transform)
            if name not in refused:
                refused.append(name)
    return refused


def compile_processor(resolved: Resolved) -> onnx.ModelProto:
    """Compile a resolved request to a graph, or refuse naming the ops."""
    refused = unsupported_ops(resolved.processor)
    if refused:
        verb = "is" if len(refused) == 1 else "are"
        raise UnsupportedOpError(
            f"{', '.join(refused)} {verb} not emitted by this compiler, so "
            f"{resolved.endpoints!r} is refused rather than approximated"
        )

    builder = GraphBuilder()
    tensor = INPUT
    for transform in resolved.processor.createGroupTransform():
        # Every op has an emitter: the refusal above is what established it,
        # through this same key.
        tensor = emitters.REGISTRY[emitters.op_label(transform)].emit(
            builder, transform, tensor
        )

    return builder.build(tensor, resolved.endpoints, resolved.metadata)
