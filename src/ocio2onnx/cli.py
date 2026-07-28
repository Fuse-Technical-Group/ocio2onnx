"""Compile, verify, and measure from the command line (§spec:verification).

Three subcommands, one per question a consumer asks: emit this transform,
does the compiler agree with OCIO across a config, and what does that config
need. Each composes the same modules the Python API does and holds no
compiling logic of its own.

A refusal is an answer rather than a crash, so an unresolvable name and an
unimplemented op each leave by their own exit code with their message on
stderr and no traceback. A consumer scripting this distinguishes "that
transform does not exist" from "this compiler will not emit it" without
parsing text.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import onnx

from ocio2onnx import __version__, census
from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    AddressError,
    Resolved,
    enumerate_transforms,
    load_config,
    reference_space,
    resolve_colorspaces,
    resolve_display_view,
)
from ocio2onnx.compiler import compile_processor, unsupported_ops
from ocio2onnx.emitters import UnsupportedOpError

#: Exit codes. The two refusals are distinct because they call for different
#: responses: one is a typo, the other is a workstream.
OK = 0
FAILED = 1
USAGE = 2  # argparse's own, restated so the tests can name it
CANNOT_RESOLVE = 3
WILL_NOT_EMIT = 4


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.run(parser, args)
    except AddressError as exc:
        return _refuse(exc, CANNOT_RESOLVE)
    except UnsupportedOpError as exc:
        return _refuse(exc, WILL_NOT_EMIT)


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface."""
    parser = argparse.ArgumentParser(
        prog="ocio2onnx",
        description="Compile an OpenColorIO transform into an ONNX graph.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    emit = subcommands.add_parser(
        "compile",
        help="emit one transform as an ONNX graph",
        description=(
            "Emit one transform as an ONNX graph. Name either a color space "
            "pair or a display view."
        ),
    )
    _add_config(emit)
    # A mutually exclusive group takes one argument at a time, so the two
    # forms are told apart by their target: --to or --display. The source is
    # shared — required by the first form, optional in the second.
    form = emit.add_mutually_exclusive_group(required=True)
    form.add_argument(
        "--to", dest="dst", metavar="SPACE", help="target color space, with --from"
    )
    form.add_argument(
        "--display", metavar="DISPLAY", help="target display, with --view"
    )
    emit.add_argument(
        "--from",
        dest="src",
        metavar="SPACE",
        help="source color space; required with --to, and defaults to the "
        "config's reference space with --display",
    )
    emit.add_argument("--view", metavar="VIEW", help="view on --display")
    emit.add_argument(
        "-o", "--output", required=True, metavar="PATH", help="where to write the graph"
    )
    emit.add_argument(
        "--verify",
        action="store_true",
        help="hold the graph against OCIO's CPU processor before writing it",
    )
    emit.set_defaults(run=_compile)

    sweep = subcommands.add_parser(
        "verify",
        help="hold every transform in a config against OCIO's CPU processor",
        description=(
            "Compile every transform in a config and hold it against OCIO's "
            "CPU processor, or report it refused by name. Nothing is skipped."
        ),
    )
    _add_config(sweep)
    sweep.set_defaults(run=_verify)

    report = subcommands.add_parser(
        "census",
        help="report a config's op coverage",
        description="Report which ops a config uses and which the compiler refuses.",
    )
    _add_config(report)
    report.set_defaults(run=_census)

    return parser


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        metavar="URI",
        help=f"built-in URI or file path (default: {DEFAULT_CONFIG})",
    )


def _refuse(exc: Exception, code: int) -> int:
    """A refusal is an answer: the message, no traceback."""
    print(str(exc), file=sys.stderr)
    return code


def _resolve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Resolved:
    """Bind the named form to a processor, or say which half is missing."""
    config = load_config(args.config)
    if args.dst is not None:
        if args.src is None:
            parser.error("--to needs --from")
        return resolve_colorspaces(config, args.src, args.dst, uri=args.config)
    if args.view is None:
        parser.error("--display needs --view")
    return resolve_display_view(
        config, args.display, args.view, src=args.src, uri=args.config
    )


def _compile(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    resolved = _resolve(parser, args)
    model = compile_processor(resolved)

    if args.verify:
        # The oracle executes a graph, which needs a runtime; compiling does
        # not, so the import stays inside the branch that uses it.
        from ocio2onnx.oracle import verify

        result = verify(resolved, model)
        print(f"{'verified' if result.ok else 'FAILED'}: {result}")
        if not result.ok:
            return FAILED

    onnx.save(model, args.output)
    print(f"wrote {args.output}")
    return OK


def _verify(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Compile and verify a whole config, partitioning every transform.

    Every transform lands in exactly one bucket. The summary prints the count
    that landed in none rather than trusting that none did: a transform
    quietly dropped from the sweep is the one failure the sweep cannot report
    any other way (§spec:verification).
    """
    from ocio2onnx.oracle import TOLERANCE, verify

    config = load_config(args.config)
    reference = reference_space(config)

    verified: list[str] = []
    refusals: list[tuple[str, list[str]]] = []
    failures: list[str] = []
    worst_abs = worst_rel = 0.0
    total = 0

    for label, processor in enumerate_transforms(config, reference):
        total += 1
        refused = unsupported_ops(processor)
        if refused:
            refusals.append((label, refused))
            continue

        resolved = Resolved(
            processor=processor,
            config_name=config.getName(),
            config_uri=args.config,
            endpoints=label,
        )
        try:
            result = verify(resolved)
        except UnsupportedOpError as exc:
            # An emitter refused a parameter rather than an op type, which the
            # check above cannot see. Still a refusal, not a failure.
            refusals.append((label, [str(exc)]))
            continue

        worst_abs = max(worst_abs, result.max_abs)
        worst_rel = max(worst_rel, result.max_rel)
        if result.ok:
            verified.append(label)
        else:
            failures.append(f"{label}: {result}")

    for failure in failures:
        print(failure, file=sys.stderr)

    compiled = len(verified) + len(failures)
    skipped = total - compiled - len(refusals)

    print(f"config:    {args.config}")
    print(f"reference: {reference}\n")
    for label, count in (
        ("verified", len(verified)),
        ("refused", len(refusals)),
        ("failed", len(failures)),
        ("skipped", skipped),
        ("total", total),
    ):
        print(f"{label + ':':10}{count:>3d}")

    if refusals:
        print("\nrefused, by op (a transform may carry more than one):")
        for op, count in census.group_by_op(refusals):
            print(f"  {count:5d}  {op}")

    print(
        f"\nworst deviation over the {compiled} compiled transforms: "
        f"abs {worst_abs:.3g}, rel {worst_rel:.3g}"
    )
    print(
        f"  every sample held the looser of abs {TOLERANCE.absolute:g} and rel "
        f"{TOLERANCE.relative:g},\n  which is why neither figure above is a "
        "bound on its own (§spec:verification)"
    )

    return FAILED if failures or skipped else OK


def _census(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return census.report(args.config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
