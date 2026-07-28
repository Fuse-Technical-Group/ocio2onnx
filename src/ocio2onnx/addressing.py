"""Resolve a compile request to an OCIO processor (§spec:emitted-graph).

A request names a config plus either a source and target color space or a
display and a view. Every name is checked against the config that was
actually loaded, so an unresolvable request fails naming both the offender
and the database it was checked against (§spec:emitted-graph).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import PyOpenColorIO as OCIO

from ocio2onnx import __version__

#: The config the specification's numbers were measured against. Versioned
#: deliberately: ``ocio://default`` moves between releases, so an artifact
#: built against it cannot say what produced it (§spec:emitted-graph).
DEFAULT_CONFIG = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5"

#: The URI this module refuses, for the reason above.
FLOATING_CONFIG = "ocio://default"

#: The reference space to assume when a config declares no reference role.
#: The pinned config leaves ``ROLE_REFERENCE`` empty.
FALLBACK_REFERENCE = "ACES2065-1"

#: Namespace for the keys stamped onto an emitted model's ``metadata_props``.
METADATA_PREFIX = "ocio2onnx."

#: What OCIO raises when it refuses. ``ExceptionMissingFile`` does **not**
#: derive from ``OCIO.Exception`` — the bindings hang both off the builtin —
#: so catching the latter alone lets a config naming an absent LUT through.
OCIO_ERRORS = (OCIO.Exception, OCIO.ExceptionMissingFile)

#: The optimization level every processor this module hands out is resolved at.
#:
#: It lives here, rather than beside the oracle that measures against it,
#: because the compiler and the oracle have to read **one** op list. A
#: processor reports the ops as the config declares them; OCIO's CPU renderer
#: runs an optimized rewrite of that list, and the two are the same transform
#: but not the same arithmetic. A display view holds two adjacent matrices that
#: compose to an identity — OCIO folds them and the compiler used to emit both,
#: which round-trips a near-black channel to about 1e-10 instead of to zero. A
#: mirrored gamma downstream turns that into 6e-5, past tolerance, on a value
#: the reference reports as exactly black.
#:
#: ``OPTIMIZATION_FAST_LOG_EXP_POW`` is cleared. It is an approximate ``pow``,
#: ``log``, and ``exp`` whose error exceeds the verification tolerance, so
#: leaving it on would measure a math library rather than this compiler's
#: reading of the config: across the pinned config 110 of the 111 closed-form
#: transforms verify against OCIO's fast path and 111 of 111 against its
#: accurate one. Which flags are set is a correctness decision, not a tuning
#: knob (§spec:verification); ``tests/test_oracle.py`` pins it.
#:
#: ``OPTIMIZATION_LUT_INV_FAST`` stays set, which is the same decision read the
#: other way: cleared, OCIO's inverse ``Lut1D`` renderer parts company with its
#: own fast path at 5 samples, and the fast path is the one that agrees with
#: the encoding.
#:
#: ``OPTIMIZATION_SIMPLIFY_OPS`` is cleared because it decides which parameters
#: stay live (§spec:dynamic-properties). It rewrites a graded op whose values it
#: can express another way — a CDL at a unit power becomes a matrix, and the
#: grade arrives as coefficients no consumer can turn. That is the primary, the
#: commonest CDL there is. The rewrite is also not value-preserving, and OCIO
#: applies it at ``OPTIMIZATION_LOSSLESS``: an inverse clamping CDL answers 1.0
#: for an input of 1 on a lifted channel as an op, and 1.111 once rewritten,
#: because the rewrite drops the style's output clamp. So the op is the correct
#: reading of the two, and clearing the flag costs nothing: it folds only graded
#: ops, so across the pinned config, which carries none, no op count moves.
#: ``tests/test_cdl.py`` pins the clamp and ``tests/test_oracle.py`` the cost.
OPTIMIZATION_FLAGS = OCIO.OptimizationFlags(
    OCIO.OPTIMIZATION_DEFAULT.value
    & ~OCIO.OPTIMIZATION_FAST_LOG_EXP_POW.value
    & ~OCIO.OPTIMIZATION_SIMPLIFY_OPS.value
)


class AddressError(ValueError):
    """A compile request that the loaded config cannot resolve."""


@dataclasses.dataclass(frozen=True)
class Resolved:
    """A compile request bound to a processor, with its provenance."""

    processor: OCIO.Processor
    config_name: str
    config_uri: str
    endpoints: str

    @property
    def metadata(self) -> dict[str, str]:
        """The provenance an emitted model carries in ``metadata_props``.

        Records the resolved config name beside the URI the caller typed, so
        an artifact says which database produced it rather than which alias
        was used to reach it.
        """
        return {
            f"{METADATA_PREFIX}version": __version__,
            f"{METADATA_PREFIX}ocio.version": OCIO.GetVersion(),
            f"{METADATA_PREFIX}config.name": self.config_name,
            f"{METADATA_PREFIX}config.uri": self.config_uri,
            f"{METADATA_PREFIX}endpoints": self.endpoints,
        }


def load_config(uri: str = DEFAULT_CONFIG) -> OCIO.Config:
    """Load a config by built-in URI (``ocio://…``) or filesystem path.

    Refuses ``ocio://default``: it points at a different config from one OCIO
    release to the next, so a graph built through it cannot record what
    produced it.
    """
    if uri == FLOATING_CONFIG:
        raise AddressError(
            f"{FLOATING_CONFIG} moves between OCIO releases, so an artifact "
            "built against it cannot say which database produced it; name a "
            f"versioned config such as {DEFAULT_CONFIG}"
        )
    try:
        return OCIO.Config.CreateFromFile(uri)
    except OCIO_ERRORS as exc:
        raise AddressError(f"cannot load config {uri!r}: {exc}") from exc


def reference_space(config: OCIO.Config) -> str:
    """The config's reference space.

    ``ROLE_REFERENCE`` is empty in the ACES studio configs, which reference
    ACES2065-1 without declaring the role.
    """
    return config.getCanonicalName(OCIO.ROLE_REFERENCE) or FALLBACK_REFERENCE


def resolve_colorspaces(
    config: OCIO.Config, src: str, dst: str, *, uri: str
) -> Resolved:
    """Resolve a source and target color space to a processor.

    Both names may be color spaces, aliases, or roles.
    """
    _require_colorspace(config, src, uri=uri, role="source")
    _require_colorspace(config, dst, uri=uri, role="target")
    return Resolved(
        processor=_processor(config, src, dst, uri=uri),
        config_name=config.getName(),
        config_uri=uri,
        endpoints=f"{src} -> {dst}",
    )


def resolve_display_view(
    config: OCIO.Config,
    display: str,
    view: str,
    *,
    src: str | None = None,
    uri: str,
) -> Resolved:
    """Resolve a display and view to a processor.

    ``src`` defaults to the config's reference space.
    """
    src = src if src is not None else reference_space(config)
    _require_colorspace(config, src, uri=uri, role="source")

    if display not in set(config.getDisplays()):
        raise AddressError(f"display {display!r} is not in {_label(config, uri)}")
    views = list(config.getViews(display))
    if view not in views:
        raise AddressError(
            f"{view!r} is not a view of display {display!r} in "
            f"{_label(config, uri)}; views are {', '.join(repr(v) for v in views)}"
        )

    transform = OCIO.DisplayViewTransform()
    transform.setSrc(src)
    transform.setDisplay(display)
    transform.setView(view)
    return Resolved(
        processor=_processor(config, transform, uri=uri),
        config_name=config.getName(),
        config_uri=uri,
        endpoints=f"{src} -> {display} / {view}",
    )


def enumerate_transforms(
    config: OCIO.Config, reference: str | None = None, *, uri: str = ""
) -> Iterator[tuple[str, OCIO.Processor]]:
    """Every transform worth measuring: each color space both ways against the
    reference, then each display view."""
    reference = reference if reference is not None else reference_space(config)

    for space in config.getColorSpaces():
        name = space.getName()
        if name == reference:
            continue
        yield f"{name} -> ref", _processor(config, name, reference, uri=uri)
        yield f"ref -> {name}", _processor(config, reference, name, uri=uri)

    for display in config.getDisplays():
        for view in config.getViews(display):
            transform = OCIO.DisplayViewTransform()
            transform.setSrc(reference)
            transform.setDisplay(display)
            transform.setView(view)
            yield f"view {display}/{view}", _processor(config, transform, uri=uri)


def _processor(config: OCIO.Config, *request: Any, uri: str) -> OCIO.Processor:
    """Build a processor, reporting OCIO's own refusal as an ``AddressError``.

    Optimized at `OPTIMIZATION_FLAGS` before it leaves, so every caller — the
    compiler, the census, the oracle — walks the op list OCIO's own renderer
    runs rather than the one the config declares.

    A config parses long before its `FileTransform` references resolve, so a
    config naming a LUT that is missing or unreadable fails here rather than
    at load. That is still a request the loaded config cannot serve, so the
    caller sees the one error type it already handles.
    """
    try:
        return config.getProcessor(*request).getOptimizedProcessor(OPTIMIZATION_FLAGS)
    except OCIO_ERRORS as exc:
        raise AddressError(
            f"{_label(config, uri)} cannot build a processor: {exc}"
        ) from exc


def _require_colorspace(config: OCIO.Config, name: str, *, uri: str, role: str) -> None:
    """Reject a name the loaded config does not carry."""
    if config.getColorSpace(name) is None:
        raise AddressError(
            f"{role} color space {name!r} is not in {_label(config, uri)}"
        )


def _label(config: OCIO.Config, uri: str) -> str:
    """How an error message names the config a check was made against."""
    name = config.getName()
    if name and uri:
        return f"config {name!r} ({uri})"
    return f"config {name or uri}"
