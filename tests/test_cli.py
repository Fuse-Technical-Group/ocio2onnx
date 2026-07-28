"""The command line (§spec:op-coverage).

The section's integration surface, and its acceptance test: a named pair
compiles and agrees with the CPU processor, and the whole pinned config
partitions with nothing skipped and nothing refused.

The refusal path is exercised against a config written here rather than
against the pinned one, which no longer carries an op this compiler will not
emit. That is the point of a separate exit code — a caller scripting this
tells "that transform does not exist" from "this compiler will not emit it" —
so it is held on the arbitrary config the boundary exists for.

Driven through ``main`` rather than through a subprocess, so a failure lands
as a traceback in the test rather than as an exit code and a blob of text. One
test still writes and reloads a real file, because that is the part a caller
actually consumes.
"""

import importlib.metadata
import re

import onnx
import pytest

from ocio2onnx.addressing import DEFAULT_CONFIG, METADATA_PREFIX
from ocio2onnx.builder import PRECISION
from ocio2onnx.cli import (
    CANNOT_RESOLVE,
    FAILED,
    OK,
    USAGE,
    WILL_NOT_EMIT,
    main,
)

#: The pair §spec:verification names.
PAIR = ("Log3G10 REDWideGamutRGB", "ACES2065-1")

#: The display view that used to be the section's refusal, and the style that
#: blocked it. It is here for the opposite reason now: it is the heaviest
#: transform the command emits.
ACES_VIEW = ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")
ACES_STYLE = "FixedFunction[ACES_OUTPUT_TRANSFORM_20]"

#: A closed-form display view.
OPEN_VIEW = ("sRGB - Display", "Un-tone-mapped")

#: An HLG display view, which reaches a display through `REC2100_SURROUND`.
HLG_VIEW = ("Rec.2100-HLG - Display", "Video (colorimetric)")

#: The partition ``verify`` reproduces: every one of the 159 transforms
#: compiles and agrees with the CPU processor (§spec:op-coverage). ``verify``
#: and the census reach the refusal count by different routes — one compiles,
#: the other reads the op set — so their agreeing on zero is a measured fact
#: about this config, not an identity.
VERIFIED = 159
REFUSED = 0
TOTAL = 159


@pytest.fixture
def graph(tmp_path):
    return str(tmp_path / "graph.onnx")


def test_the_console_script_points_at_this_entry_point():
    (script,) = [
        entry
        for entry in importlib.metadata.entry_points(group="console_scripts")
        if entry.name == "ocio2onnx"
    ]
    assert script.load() is main


def test_compile_writes_a_graph_recording_its_config_and_precision(graph):
    """The ROADMAP's first Verify item, end to end: a caller runs the console
    script and reads what came out."""
    assert main(["compile", "--from", PAIR[0], "--to", PAIR[1], "-o", graph]) == OK

    model = onnx.load(graph)
    recorded = {entry.key: entry.value for entry in model.metadata_props}
    assert recorded[f"{METADATA_PREFIX}config.name"]
    assert recorded[f"{METADATA_PREFIX}config.uri"] == DEFAULT_CONFIG
    assert recorded[f"{METADATA_PREFIX}precision"] == PRECISION == "float32"
    assert recorded[f"{METADATA_PREFIX}endpoints"] == f"{PAIR[0]} -> {PAIR[1]}"


def test_compile_verify_holds_the_graph_against_the_cpu_processor(graph, capsys):
    argv = ["compile", "--from", PAIR[0], "--to", PAIR[1], "-o", graph, "--verify"]
    assert main(argv) == OK
    printed = capsys.readouterr().out
    assert "0 of" in printed
    assert "outside tolerance" in printed


@pytest.mark.parametrize(("display", "view"), (OPEN_VIEW, HLG_VIEW))
def test_compile_takes_a_display_view(display, view, graph):
    """The HLG view is here because a consumer asking for HLG output used to
    meet a refusal naming `REC2100_SURROUND` (§spec:op-coverage)."""
    assert main(["compile", "--display", display, "--view", view, "-o", graph]) == OK
    recorded = {entry.key: entry.value for entry in onnx.load(graph).metadata_props}
    assert recorded[f"{METADATA_PREFIX}endpoints"].endswith(f"{display} / {view}")


def test_compile_takes_a_config(graph):
    argv = ["compile", "--from", PAIR[0], "--to", PAIR[1], "-o", graph]
    assert main([*argv, "--config", DEFAULT_CONFIG]) == OK


def test_a_display_view_carrying_the_aces_output_transform_verifies(graph, capsys):
    """The view this command used to refuse on `ACES_OUTPUT_TRANSFORM_20`. It
    is the acceptance criterion read the other way round, and it is checked
    with ``--verify`` because the op is the image's look: a graph that writes
    but disagrees is the failure worth catching here."""
    display, view = ACES_VIEW
    argv = ["compile", "--display", display, "--view", view, "-o", graph, "--verify"]
    assert main(argv) == OK
    assert "0 of" in capsys.readouterr().out

    recorded = {entry.key: entry.value for entry in onnx.load(graph).metadata_props}
    assert recorded[f"{METADATA_PREFIX}endpoints"].endswith(f"{display} / {view}")


def test_an_unresolvable_name_exits_differently_from_a_refusal(graph, capsys):
    code = main(
        ["compile", "--from", "Not A Color Space", "--to", "ACEScg", "-o", graph]
    )
    assert code == CANNOT_RESOLVE != WILL_NOT_EMIT
    err = capsys.readouterr().err
    assert "not in config" in err
    assert "Traceback" not in err


def test_an_unloadable_config_is_refused_without_a_traceback(graph, capsys):
    code = main(
        [
            "compile",
            "--from",
            PAIR[0],
            "--to",
            PAIR[1],
            "-o",
            graph,
            "--config",
            "ocio://default",
        ]
    )
    assert code == CANNOT_RESOLVE
    assert "Traceback" not in capsys.readouterr().err


#: A config carrying an op no emitter is registered for. Written here rather
#: than found: every op the pinned config carries now emits, and the exit code
#: a refusal leaves by is what a caller scripting this reads.
CONFIG_WITH_AN_UNEMITTABLE_OP = """ocio_profile_version: 2
roles:
  default: ref
  reference: ref
colorspaces:
  - !<ColorSpace>
    name: ref
  - !<ColorSpace>
    name: unemittable
    from_scene_reference: !<FixedFunctionTransform> {style: ACES_RedMod03}
"""

#: The style that config carries, as a refusal spells it. ``FixedFunction``
#: alone cannot say which behaviour of the class blocked the caller.
UNEMITTABLE_STYLE = "FixedFunction[ACES_RED_MOD_03]"


@pytest.fixture
def config_with_an_unemittable_op(tmp_path):
    path = tmp_path / "unemittable.ocio"
    path.write_text(CONFIG_WITH_AN_UNEMITTABLE_OP)
    return str(path)


def test_an_unemittable_op_refuses_by_name_without_a_traceback(
    config_with_an_unemittable_op, graph, capsys
):
    """A refusal is an answer, not a crash: the message names the style and the
    endpoints, it leaves by its own exit code, and no traceback reaches the
    caller."""
    code = main(
        [
            "compile",
            "--from",
            "unemittable",
            "--to",
            "ref",
            "-o",
            graph,
            "--config",
            config_with_an_unemittable_op,
        ]
    )
    assert code == WILL_NOT_EMIT

    err = capsys.readouterr().err
    assert UNEMITTABLE_STYLE in err
    assert "unemittable -> ref" in err
    assert "Traceback" not in err


#: A config whose transform is non-finite over the whole lattice. A NaN
#: coefficient in a declared matrix is one step from an arbitrary config, and
#: the graph reproduces it faithfully — so every sample is matched by class and
#: none is held against the tolerance.
CONFIG_WITH_A_NON_FINITE_TRANSFORM = """ocio_profile_version: 2
roles:
  default: ref
  reference: ref
colorspaces:
  - !<ColorSpace>
    name: ref
  - !<ColorSpace>
    name: nonfinite
    from_scene_reference: !<MatrixTransform>
      matrix: [.nan, 0, 0, 0, 0, .nan, 0, 0, 0, 0, .nan, 0, 0, 0, 0, 1]
"""


@pytest.fixture
def config_with_a_non_finite_transform(tmp_path):
    path = tmp_path / "nonfinite.ocio"
    path.write_text(CONFIG_WITH_A_NON_FINITE_TRANSFORM)
    return str(path)


def test_compile_verify_refuses_to_certify_a_graph_on_no_evidence(
    config_with_a_non_finite_transform, graph, capsys
):
    """``--verify`` used to print ``verified: 0 of 0 samples outside
    tolerance`` over a 1248-sample lattice. A verified graph needs at least one
    finite sample compared (§spec:evidence-floor)."""
    code = main(
        [
            "compile",
            "--from",
            "nonfinite",
            "--to",
            "ref",
            "-o",
            graph,
            "--config",
            config_with_a_non_finite_transform,
            "--verify",
        ]
    )
    assert code == FAILED

    printed = capsys.readouterr().out
    assert printed.startswith("FAILED:")
    assert "no finite sample" in printed
    assert "verified" not in printed


#: A config that parses but names a LUT that is not there. OCIO resolves a
#: `FileTransform` lazily, at getProcessor rather than at load, so this is the
#: refusal that escapes if only load_config is guarded.
CONFIG_WITH_A_MISSING_LUT = """ocio_profile_version: 2
search_path: luts
roles:
  default: ref
  reference: ref
colorspaces:
  - !<ColorSpace>
    name: ref
  - !<ColorSpace>
    name: broken
    from_scene_reference: !<FileTransform> {src: absent.spi1d}
"""


@pytest.fixture
def config_with_a_missing_lut(tmp_path):
    path = tmp_path / "broken.ocio"
    path.write_text(CONFIG_WITH_A_MISSING_LUT)
    return str(path)


def test_a_config_naming_a_missing_lut_is_refused_without_a_traceback(
    config_with_a_missing_lut, graph, capsys
):
    code = main(
        [
            "compile",
            "--from",
            "broken",
            "--to",
            "ref",
            "-o",
            graph,
            "--config",
            config_with_a_missing_lut,
        ]
    )
    assert code == CANNOT_RESOLVE
    err = capsys.readouterr().err
    assert "cannot build a processor" in err
    assert "Traceback" not in err


def test_an_unwritable_output_path_is_refused_without_a_traceback(tmp_path, capsys):
    missing = str(tmp_path / "no-such-directory" / "graph.onnx")
    code = main(["compile", "--from", PAIR[0], "--to", PAIR[1], "-o", missing])
    assert code == FAILED
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--from", PAIR[0]],
        ["--from", PAIR[0], "--to", PAIR[1], "--display", "sRGB - Display"],
    ],
)
def test_the_two_forms_are_mutually_exclusive_and_one_is_required(argv, graph, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["compile", *argv, "-o", graph])
    assert caught.value.code == USAGE
    assert "usage:" in capsys.readouterr().err


def test_a_half_stated_form_says_which_half_is_missing(graph, capsys):
    with pytest.raises(SystemExit):
        main(["compile", "--display", "sRGB - Display", "-o", graph])
    assert "--view" in capsys.readouterr().err


def test_compile_needs_somewhere_to_write():
    with pytest.raises(SystemExit) as caught:
        main(["compile", "--from", PAIR[0], "--to", PAIR[1]])
    assert caught.value.code == USAGE


def buckets(printed):
    """The partition, read back off the summary the command printed."""
    found = re.findall(r"^(\w+): +(\d+)$", printed, flags=re.MULTILINE)
    return {label: int(count) for label, count in found}


def test_verify_partitions_the_whole_config_with_nothing_skipped(capsys):
    """The section's acceptance test. Every one of the 159 transforms lands in
    exactly one bucket, and the identity is printed rather than assumed —
    ``skipped`` is what a transform quietly dropped from the sweep shows up
    as, and it is the one failure the sweep cannot report any other way."""
    assert main(["verify"]) == OK
    printed = capsys.readouterr().out

    assert buckets(printed) == {
        "verified": VERIFIED,
        "refused": REFUSED,
        "failed": 0,
        "skipped": 0,
        "total": TOTAL,
    }
    assert VERIFIED == TOTAL

    assert "refused, by op" not in printed
    assert ACES_STYLE not in printed
    assert "REC2100_SURROUND" not in printed
    assert "Lut1D" not in printed
    assert "worst deviation" in printed


def test_census_is_reachable_from_the_cli(capsys):
    assert main(["census"]) == OK
    printed = capsys.readouterr().out
    assert f"transforms measured: {TOTAL}" in printed
    assert f"refused: {REFUSED}/{TOTAL}" in printed


def test_an_unknown_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["decompile"])
    assert caught.value.code == USAGE


def test_failure_is_distinct_from_refusal_and_from_success():
    assert len({OK, FAILED, USAGE, CANNOT_RESOLVE, WILL_NOT_EMIT}) == 5
