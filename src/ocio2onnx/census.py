"""Measure a config's op coverage: what the compiler must emit, and what it
refuses (§spec:op-coverage, §road:coverage-report).

Walks every color space in both directions against the reference, plus every
display view, and reports three things:

- the op census — which op types appear, and how often
- the LUT partition — which transforms OCIO cannot express analytically and
  bakes into a sampled texture instead
- the refused split, taken from ``compiler.unsupported_ops`` rather than from
  a list kept here

That last point is the whole of this module's discipline. A private copy of
the supported set drifts the moment an emitter is added, and it had: it named
``Lut1D``, which no emitter implements, so the census reported 28 refusals
where the compiler makes 48.

Run it against a new OCIO release to see what changed: a new op type shows up
as a refusal rather than as silence. This is the measurement the specification
quotes, not a separate estimate of it.

Reachable as ``ocio2onnx census`` or as ``python tools/census.py``.

Note: this reports *coverage*, not correctness. Verifying an emitted graph
against OCIO's CPU processor is ``ocio2onnx verify`` (§spec:verification).
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Iterable, Sequence

import PyOpenColorIO as OCIO

from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    enumerate_transforms,
    load_config,
    reference_space,
)
from ocio2onnx.compiler import op_names, unsupported_ops
from ocio2onnx.emitters import supported_ops

#: How many refusals the report lists before summarising the rest.
LISTED = 10


@dataclasses.dataclass(frozen=True)
class Census:
    """What one config asks of the compiler."""

    uri: str
    reference: str
    total: int
    supported: frozenset[str]
    ops: collections.Counter[str]
    refusals: list[tuple[str, list[str]]]
    needs_lut: list[tuple[str, int, int]]


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


def take(config, uri: str) -> Census:
    """Measure one config."""
    reference = reference_space(config)
    ops: collections.Counter[str] = collections.Counter()
    refusals: list[tuple[str, list[str]]] = []
    needs_lut: list[tuple[str, int, int]] = []
    total = 0

    for label, processor in enumerate_transforms(config, reference):
        total += 1
        ops.update(op_names(processor))

        refused = unsupported_ops(processor)
        if refused:
            refusals.append((label, refused))

        n1d, n3d = sampled_textures(processor)
        if n1d or n3d:
            needs_lut.append((label, n1d, n3d))

    return Census(
        uri=uri,
        reference=reference,
        total=total,
        supported=supported_ops(),
        ops=ops,
        refusals=refusals,
        needs_lut=needs_lut,
    )


def group_by_op(
    refusals: Iterable[tuple[str, list[str]]],
) -> list[tuple[str, int]]:
    """Refused transforms per op, commonest first.

    A transform carrying two unimplemented ops counts under both, so these sum
    above the number of refused transforms. That is the useful reading: a
    consumer asks which op is blocking them, not how the blame divides.
    """
    counts: collections.Counter[str] = collections.Counter()
    for _, ops in refusals:
        counts.update(ops)
    return counts.most_common()


def report(uri: str = DEFAULT_CONFIG) -> int:
    """Print the census for one config."""
    measured = take(load_config(uri), uri)

    print(f"config:    {measured.uri}")
    print(f"reference: {measured.reference}")
    print(f"transforms measured: {measured.total}\n")

    print(f"op census ({len(measured.ops)} distinct types)")
    for op, count in measured.ops.most_common():
        mark = " " if op in measured.supported else "*"
        print(f"  {mark} {count:5d}  {op}")
    print("  (* = no emitter registered; refused by name)\n")

    lut3d = sum(1 for _, _, n3d in measured.needs_lut if n3d)
    print(
        f"needs a sampled LUT: {len(measured.needs_lut)}/{measured.total}  "
        f"(3D LUT: {lut3d})"
    )

    print(f"\nemits: {', '.join(sorted(measured.supported))}")
    print(
        f"refused: {len(measured.refusals)}/{measured.total}, by op "
        "(a transform may carry more than one)"
    )
    for op, count in group_by_op(measured.refusals):
        print(f"  {count:5d}  {op}")

    print()
    for label, ops in measured.refusals[:LISTED]:
        print(f"  {label}: {', '.join(ops)}")
    if len(measured.refusals) > LISTED:
        print(f"  ... and {len(measured.refusals) - LISTED} more")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python tools/census.py [config-uri]``."""
    return report(argv[0] if argv else DEFAULT_CONFIG)
