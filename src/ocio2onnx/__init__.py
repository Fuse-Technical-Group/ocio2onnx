"""Compile an OpenColorIO transform into an ONNX graph.

OCIO stays the authority for what a transform is; this package only changes
what can execute it (§spec:problem-statement).

The two entry points here are the whole public surface. Each loads a config,
resolves a request against it, and compiles the processor OCIO returns; the
CLI wraps them rather than duplicating them. They hold no logic of their own,
so what an emitted graph is worth is settled by the modules they compose and
by the oracle (§spec:verification), not here.
"""

from __future__ import annotations

# Assigned before the imports below because ``addressing`` reads it back off
# this module to stamp an emitted graph's provenance.
__version__ = "0.1.0"

from typing import TYPE_CHECKING, Any

from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    AddressError,
    load_config,
    resolve_colorspaces,
    resolve_display_view,
)
from ocio2onnx.compiler import UnsupportedOpError, compile_processor

if TYPE_CHECKING:
    import onnx

__all__ = [
    "DEFAULT_CONFIG",
    "AddressError",
    "UnsupportedOpError",
    "__version__",
    "compile_colorspaces",
    "compile_display_view",
    "verify",
]


def compile_colorspaces(
    src: str, dst: str, *, config: str = DEFAULT_CONFIG
) -> onnx.ModelProto:
    """Compile the transform from one color space to another.

    ``src`` and ``dst`` may each be a color space name, an alias, or a role.
    ``config`` is a built-in URI (``ocio://…``) or a filesystem path, and is
    recorded in the emitted graph beside the name it resolved to.

    Raises ``AddressError`` if the loaded config cannot resolve a name, and
    ``UnsupportedOpError`` if the transform carries an op this compiler does
    not emit — refused rather than approximated (§spec:op-coverage).
    """
    loaded = load_config(config)
    return compile_processor(resolve_colorspaces(loaded, src, dst, uri=config))


def compile_display_view(
    display: str, view: str, *, src: str | None = None, config: str = DEFAULT_CONFIG
) -> onnx.ModelProto:
    """Compile the transform from a color space to a display view.

    ``src`` defaults to the config's reference space. Refuses on the same
    terms as ``compile_colorspaces``.
    """
    loaded = load_config(config)
    return compile_processor(
        resolve_display_view(loaded, display, view, src=src, uri=config)
    )


def __getattr__(name: str) -> Any:
    """Serve ``verify`` lazily.

    The oracle executes an emitted graph, which needs an ONNX runtime.
    Compiling does not need one, and this package is not an executor
    (§spec:non-goals), so the runtime stays an optional extra and importing
    this module does not require it.
    """
    if name == "verify":
        from ocio2onnx.oracle import verify

        return verify
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
