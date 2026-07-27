#!/usr/bin/env python3
"""Measure a config's op coverage: what the compiler must emit, and what it
must refuse (§spec:op-coverage, §road:coverage-report).

Walks every color space in both directions against the reference, plus every
display view, and reports three things:

- the op census — which op types appear, and how often
- the LUT partition — which transforms OCIO cannot express analytically and
  bakes into a sampled texture instead
- the supported/refused split against the op set the compiler implements

Run it against a new OCIO release to see what changed: a new op type shows up
as a refusal rather than as silence. This is the measurement the specification
quotes, not a separate estimate of it.

Requires ``opencolorio``. Usage::

    python tools/census.py [config-uri]

Note: this reports *coverage*, not correctness. Verifying an emitted graph
against OCIO's CPU processor is §road:oracle-harness, and is not this script.
"""

from __future__ import annotations

import collections
import sys

import PyOpenColorIO as OCIO

#: The config the specification's numbers were measured against. Versioned
#: deliberately: ``ocio://default`` moves between releases, so an artifact
#: built against it cannot say what produced it (§spec:emitted-graph).
DEFAULT_CONFIG = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5"

#: Ops the compiler emits directly (§spec:op-coverage). Everything else is
#: refused by name rather than approximated.
SUPPORTED = frozenset(
    {
        "Matrix",
        "Range",
        "Exponent",
        "ExponentWithLinear",
        "Log",
        "LogCamera",
        "Lut1D",
    }
)


def op_names(processor) -> list[str]:
    """The op types a processor is built from, as bare names."""
    return [
        type(t).__name__.removesuffix("Transform")
        for t in processor.createGroupTransform()
    ]


def sampled_textures(processor) -> tuple[int, int]:
    """``(1D, 3D)`` texture counts OCIO itself needs for this transform.

    Non-zero means OCIO could not express those ops analytically and baked
    them into a lattice — the honest measure of where a LUT is unavoidable,
    taken from OCIO's own GPU partition rather than assumed.
    """
    gpu = processor.getDefaultGPUProcessor()
    desc = OCIO.GpuShaderDesc.CreateShaderDesc(language=OCIO.GPU_LANGUAGE_GLSL_4_0)
    gpu.extractGpuShaderInfo(desc)
    return len(list(desc.getTextures())), len(list(desc.get3DTextures()))


def transforms(config, reference: str):
    """Every transform worth measuring: each color space both ways against the
    reference, then each display view."""
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


def main(argv: list[str]) -> int:
    uri = argv[1] if len(argv) > 1 else DEFAULT_CONFIG
    config = OCIO.Config.CreateFromFile(uri)
    reference = config.getCanonicalName(OCIO.ROLE_REFERENCE) or "ACES2065-1"

    census: collections.Counter[str] = collections.Counter()
    needs_lut: list[tuple[str, int, int]] = []
    refused: list[tuple[str, set[str]]] = []
    total = 0

    for label, processor in transforms(config, reference):
        total += 1
        ops = op_names(processor)
        census.update(ops)

        unsupported = set(ops) - SUPPORTED
        if unsupported:
            refused.append((label, unsupported))

        n1d, n3d = sampled_textures(processor)
        if n1d or n3d:
            needs_lut.append((label, n1d, n3d))

    print(f"config:    {uri}")
    print(f"reference: {reference}")
    print(f"transforms measured: {total}\n")

    print(f"op census ({len(census)} distinct types)")
    for op, count in census.most_common():
        mark = " " if op in SUPPORTED else "*"
        print(f"  {mark} {count:5d}  {op}")
    print("  (* = not emitted; refused by name)\n")

    lut3d = sum(1 for _, _, n3d in needs_lut if n3d)
    print(f"needs a sampled LUT: {len(needs_lut)}/{total}  (3D LUT: {lut3d})")

    print(f"\nrefused: {len(refused)}/{total}")
    for label, ops in refused[:10]:
        print(f"  {label}: {', '.join(sorted(ops))}")
    if len(refused) > 10:
        print(f"  ... and {len(refused) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
