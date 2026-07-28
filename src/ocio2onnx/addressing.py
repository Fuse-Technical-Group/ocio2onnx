"""Resolve a compile request to an OCIO processor (§spec:emitted-graph).

A request names a config plus either a source and target color space or a
display and a view. Every name is checked against the config that was
actually loaded, so an unresolvable request fails naming both the offender
and the database it was checked against (§spec:emitted-graph).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

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

#: The emitted graph carries three channels; alpha bypasses it rather than
#: flowing through identity arithmetic (§spec:emitted-graph).
CHANNELS = 3

#: The graph's interface: channels-first, batch and spatial dimensions free,
#: so one graph serves every resolution (§spec:emitted-graph).
IMAGE_SHAPE = ("N", CHANNELS, "H", "W")

#: Namespace for the keys stamped onto an emitted model's ``metadata_props``.
METADATA_PREFIX = "ocio2onnx."


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
    except OCIO.Exception as exc:
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
        processor=config.getProcessor(src, dst),
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
        processor=config.getProcessor(transform),
        config_name=config.getName(),
        config_uri=uri,
        endpoints=f"{src} -> {display} / {view}",
    )


def op_names(processor: OCIO.Processor) -> list[str]:
    """The op types a processor is built from, as bare names."""
    return [
        type(t).__name__.removesuffix("Transform")
        for t in processor.createGroupTransform()
    ]


def enumerate_transforms(
    config: OCIO.Config, reference: str | None = None
) -> Iterator[tuple[str, OCIO.Processor]]:
    """Every transform worth measuring: each color space both ways against the
    reference, then each display view."""
    reference = reference if reference is not None else reference_space(config)

    for space in config.getColorSpaces():
        name = space.getName()
        if name == reference:
            continue
        yield f"{name} -> ref", config.getProcessor(name, reference)
        yield f"ref -> {name}", config.getProcessor(reference, name)

    for display in config.getDisplays():
        for view in config.getViews(display):
            transform = OCIO.DisplayViewTransform()
            transform.setSrc(reference)
            transform.setDisplay(display)
            transform.setView(view)
            yield f"view {display}/{view}", config.getProcessor(transform)


def _require_colorspace(config: OCIO.Config, name: str, *, uri: str, role: str) -> None:
    """Reject a name the loaded config does not carry."""
    if config.getColorSpace(name) is None:
        raise AddressError(
            f"{role} color space {name!r} is not in {_label(config, uri)}"
        )


def _label(config: OCIO.Config, uri: str) -> str:
    """How an error message names the config a check was made against."""
    name = config.getName()
    return f"config {name!r} ({uri})" if name else f"config {uri}"
