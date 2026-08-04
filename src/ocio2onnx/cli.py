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

from ocio2onnx import __version__, bench, census
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
from ocio2onnx.builder import parameters
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
    except OSError as exc:
        # An unwritable output path or an unreadable config file is the
        # caller's answer too, not a crash.
        return _refuse(exc, FAILED)


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
    _add_endpoints(emit)
    emit.add_argument(
        "-o", "--output", required=True, metavar="PATH", help="where to write the graph"
    )
    emit.add_argument(
        "--verify",
        action="store_true",
        help="hold the graph against OCIO's CPU processor before writing it",
    )
    _add_provider(emit, "with --verify, ")
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
    _add_provider(sweep)
    sweep.set_defaults(run=_verify)

    against = subcommands.add_parser(
        "shader",
        help="hold OCIO's own generated shader against the same oracle",
        description=(
            "Score OCIO's generated GLSL and this compiler's graph against the "
            "same CPU processor, over the same lattice and the same tolerance. "
            "Needs a GL 4.0 driver."
        ),
    )
    _add_config(against)
    against.set_defaults(run=_shader)

    speed = subcommands.add_parser(
        "bench",
        help="time OCIO's shader against the emitted graph on a GPU",
        description=(
            "Time one transform both ways at display resolutions. Needs a GL "
            "4.0 driver for OCIO's side and a GPU execution provider for the "
            "graph's. Nothing crosses PCIe inside a measurement, each frame is "
            "finished before the next is timed, and a time is reported only for "
            "a frame that matched OCIO's CPU processor."
        ),
    )
    _add_config(speed)
    _add_endpoints(speed)
    speed.add_argument(
        "--size",
        action="append",
        metavar="WxH",
        help="a frame size to time, repeatable (default: 1920x1080 and 3840x2160)",
    )
    speed.add_argument(
        "--provider",
        action="append",
        metavar="NAME",
        help="a GPU execution provider to time, repeatable (default: every "
        "non-CPU provider this onnxruntime offers)",
    )
    speed.add_argument(
        "--iterations",
        type=int,
        default=bench.ITERATIONS,
        metavar="N",
        help=f"timed frames per size (default: {bench.ITERATIONS})",
    )
    speed.set_defaults(run=_bench)

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


def _add_endpoints(parser: argparse.ArgumentParser) -> None:
    """Which transform, named either way.

    A mutually exclusive group takes one argument at a time, so the two forms
    are told apart by their target: --to or --display. The source is shared —
    required by the first form, optional in the second.
    """
    form = parser.add_mutually_exclusive_group(required=True)
    form.add_argument(
        "--to", dest="dst", metavar="SPACE", help="target color space, with --from"
    )
    form.add_argument(
        "--display", metavar="DISPLAY", help="target display, with --view"
    )
    parser.add_argument(
        "--from",
        dest="src",
        metavar="SPACE",
        help="source color space; required with --to, and defaults to the "
        "config's reference space with --display",
    )
    parser.add_argument("--view", metavar="VIEW", help="view on --display")


def _add_provider(parser: argparse.ArgumentParser, when: str = "") -> None:
    """Where the graph runs. The reference does not move with it.

    Left open rather than given `choices`, because the set an onnxruntime build
    offers is a property of that build; naming one it does not have is answered
    by the oracle with the list it does (`oracle.resolve_provider`).
    """
    parser.add_argument(
        "--provider",
        metavar="NAME",
        help=f"{when}run the graph on this execution provider — cpu, cuda, "
        "tensorrt, dml, or an onnxruntime provider's full name (default: cpu, "
        "which is the arithmetic the reference itself runs on)",
    )
    parser.add_argument(
        "--strict-provider",
        action="store_true",
        help="refuse the run unless --provider takes every node; without it "
        "onnxruntime may still place some on the CPU, as it does deliberately "
        "for shape ops",
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
        from ocio2onnx.oracle import ProviderError, verify

        try:
            result = verify(
                resolved, model, provider=args.provider, strict=args.strict_provider
            )
        except ProviderError as exc:
            return _refuse(exc, FAILED)
        print(f"{'verified' if result.ok else 'FAILED'}: {result}")
        if not result.ok:
            return FAILED

    onnx.save(model, args.output)
    print(f"wrote {args.output}")
    _report_parameters(model)
    return OK


def _report_parameters(model: onnx.ModelProto) -> None:
    """Say what the graph takes beyond the image (§spec:dynamic-properties).

    A live parameter is the capability that distinguishes an emitted graph from
    a baked table, and it is invisible in the file a caller just wrote — an
    input backed by a default looks like any other initializer until something
    names it. Silent where the transform carries none.
    """
    live = parameters(model)
    if not live:
        return
    width = max(len(name) for name in live)
    print("\nlive parameters (bind to vary per frame; unbound reads the default):")
    for name, default in live.items():
        print(f"  {name:{width}}  {', '.join(f'{value:g}' for value in default)}")


def _verify(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Compile and verify a whole config, partitioning every transform.

    Every transform lands in exactly one bucket. The summary prints the count
    that landed in none rather than trusting that none did: a transform
    quietly dropped from the sweep is the one failure the sweep cannot report
    any other way (§spec:verification).
    """
    from ocio2onnx.oracle import TOLERANCE, ProviderError, resolve_provider, verify

    config = load_config(args.config)
    reference = reference_space(config)

    verified: list[str] = []
    refusals: list[tuple[str, list[str]]] = []
    failures: list[str] = []
    worst_abs = worst_rel = 0.0
    total = 0

    for label, processor in enumerate_transforms(config, reference, uri=args.config):
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
            result = verify(
                resolved, provider=args.provider, strict=args.strict_provider
            )
        except UnsupportedOpError as exc:
            # An emitter refused a parameter rather than an op type, which the
            # check above cannot see. Still a refusal, not a failure.
            refusals.append((label, [str(exc)]))
            continue
        except ProviderError as exc:
            # The runtime is a property of the sweep, not of this transform,
            # so the rest would refuse identically. Leaving here is what keeps
            # the summary from reporting a partition of a config it abandoned.
            return _refuse(exc, FAILED)

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
    print(f"reference: {reference}")
    if args.provider is not None:
        # Only when it moved. A sweep run somewhere other than the default is
        # a different claim, and the report shall not read like the usual one.
        print(f"provider:  {resolve_provider(args.provider)}")
    print()
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


def _shader(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Two candidates, one oracle (§spec:op-coverage).

    OCIO's GPU path bakes a texture where this compiler evaluates, which is a
    claim about accuracy rather than about taste. Both are scored against the
    same CPU processor, over the same lattice, at the same tolerance — so the
    only thing that differs between the two columns is what runs the transform.

    A shader that misses the tolerance is a measurement, not this command's
    failure: the exit code still answers for the compiler alone.
    """
    from ocio2onnx.emitters import op_label
    from ocio2onnx.oracle import TOLERANCE, compare, cpu_reference, lattice, run_graph
    from ocio2onnx.shader import (
        LANGUAGE_NAME,
        ShaderError,
        sampled_textures,
        shader_reference,
    )

    config = load_config(args.config)
    reference = reference_space(config)

    CANDIDATES = ("graph", "shader")
    verified = dict.fromkeys(CANDIDATES, 0)
    worst_abs = dict.fromkeys(CANDIDATES, 0.0)
    worst_rel = dict.fromkeys(CANDIDATES, 0.0)
    margin = dict.fromkeys(CANDIDATES, 0.0)
    failures: list[str] = []
    refusals: list[tuple[str, list[str]]] = []
    unrunnable: list[str] = []
    sampled = closed_form_sampled = 0
    total = compiled = 0

    for label, processor in enumerate_transforms(config, reference, uri=args.config):
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
            model = compile_processor(resolved)
        except UnsupportedOpError as exc:
            refusals.append((label, [str(exc)]))
            continue
        compiled += 1

        # One lattice and one reference for both candidates. Computing them
        # twice would let the two columns disagree about what they measured.
        samples = lattice(processor)
        want = cpu_reference(processor, samples)

        textures = sampled_textures(processor)
        if textures:
            sampled += 1
            ops = processor.createGroupTransform()
            if not any(op_label(op) == "Lut1D" for op in ops):
                closed_form_sampled += 1

        candidates = {"graph": compare(want, run_graph(model, samples), samples)}
        try:
            candidates["shader"] = compare(
                want, shader_reference(processor, samples), samples
            )
        except ShaderError as exc:
            unrunnable.append(f"{label}: {exc}")

        for name, result in candidates.items():
            verified[name] += int(result.ok)
            worst_abs[name] = max(worst_abs[name], result.max_abs)
            worst_rel[name] = max(worst_rel[name], result.max_rel)
            margin[name] = max(margin[name], result.margin)
            if name == "graph" and not result.ok:
                failures.append(f"{label}: {result}")

    for failure in failures:
        print(failure, file=sys.stderr)

    print(f"config:    {args.config}")
    print(f"reference: {reference}")
    print(f"language:  {LANGUAGE_NAME}\n")

    ran = {"graph": compiled, "shader": compiled - len(unrunnable)}
    columns = ("this compiler", "OCIO's shader")
    print(f"{'':12}{columns[0]:>16}{columns[1]:>16}")
    for title, row in (
        ("verified:", verified),
        ("of:", ran),
        ("worst abs:", worst_abs),
        ("worst rel:", worst_rel),
        ("margin:", margin),
    ):
        cells = "".join(
            f"{row[name]:>16d}" if isinstance(row[name], int) else f"{row[name]:>16.3g}"
            for name in CANDIDATES
        )
        print(f"{title:12}{cells}")
    print(
        f"\nboth held the looser of abs {TOLERANCE.absolute:g} and rel "
        f"{TOLERANCE.relative:g} (§spec:verification), so neither worst figure\n"
        "  is a bound on its own. `margin` is: every deviation as a fraction "
        "of the\n  bound that governed it, so 1.0 is exactly at tolerance and "
        "the two\n  columns are comparable."
    )

    print(
        f"\nOCIO's shader sampled a texture for {sampled} of the {compiled} "
        f"compiled transforms;\n  {closed_form_sampled} of those carry no "
        "Lut1D op to account for it — their tables\n  belong to a fixed "
        "function OCIO's GPU path cannot evaluate in closed\n  form. This "
        "compiler samples for none of them."
    )

    if refusals:
        print(f"\nrefused by this compiler: {len(refusals)} of {total}")
        for op, count in census.group_by_op(refusals):
            print(f"  {count:5d}  {op}")
    for line in unrunnable:
        print(f"\nshader could not be run — {line}")

    return FAILED if failures else OK


def _bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Time one transform both ways, at every size asked for.

    The graph is compiled once and timed on each provider, because a consumer
    ships one artifact rather than one per runtime. A candidate that cannot run
    says so in its row and the rest of the table still stands — an absent
    TensorRT is a fact about the machine, not a reason to report nothing.
    """
    from ocio2onnx.oracle import ProviderError
    from ocio2onnx.shader import ShaderError

    resolved = _resolve(parser, args)
    model = compile_processor(resolved)

    try:
        wanted = bench.sizes(args.size)
    except ValueError as exc:
        parser.error(str(exc))

    providers = args.provider or _gpu_providers()
    if not providers:
        return _refuse(
            RuntimeError(
                "this onnxruntime offers no GPU provider to compare against; "
                "install onnxruntime-gpu"
            ),
            FAILED,
        )

    print(f"transform: {resolved.endpoints}")
    print(f"config:    {args.config}")
    print(f"gpu:       {bench.gpu_state()}")
    print(f"frames:    {args.iterations} timed, {bench.WARMUP} warmup")
    print(f"  pin the clocks or these are not reproducible:\n    {bench.PINNING}\n")

    measured = 0
    for width, height in wanted:
        planar, pixels = bench.frame(width, height)
        print(f"=== {width}x{height} ({width * height / 1e6:.2f} Mpix) ===")

        try:
            timing = bench.time_shader(
                resolved.processor, width, height, pixels, args.iterations
            )
            print(f"  {timing}")
            measured += 1
        except (bench.BenchError, ShaderError) as exc:
            print(f"  OCIO GLSL not measured: {exc}", file=sys.stderr)

        for provider in providers:
            try:
                timing = bench.time_graph(
                    model,
                    resolved.processor,
                    provider,
                    width,
                    height,
                    planar,
                    args.iterations,
                )
                print(f"  {timing}")
                measured += 1
            except (bench.BenchError, ProviderError) as exc:
                print(f"  {provider} not measured: {exc}", file=sys.stderr)
        print()

    # One row is not a comparison, and a table with a single survivor reads
    # like one anyway.
    return OK if measured > 1 else FAILED


def _gpu_providers() -> list[str]:
    """Every provider this onnxruntime offers that is not the CPU."""
    import onnxruntime as ort

    from ocio2onnx.oracle import DEFAULT_PROVIDER

    return [
        name
        for name in ort.get_available_providers()
        if name != DEFAULT_PROVIDER and not name.startswith("Azure")
    ]


def _census(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return census.report(args.config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
