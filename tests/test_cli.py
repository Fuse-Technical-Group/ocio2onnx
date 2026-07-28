"""The command line (§spec:op-coverage).

The section's integration surface, and its acceptance test. The ROADMAP's
Verify block asks for three things — a named pair compiles and agrees with the
CPU processor, a display view carrying `ACES_OUTPUT_TRANSFORM_20` refuses
naming the op, and the whole pinned config partitions with nothing skipped —
and each is a test here.

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

#: The display view it names as its refusal, and the style that blocks it.
ACES_VIEW = ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")
ACES_STYLE = "FixedFunction[ACES_OUTPUT_TRANSFORM_20]"

#: A closed-form display view.
OPEN_VIEW = ("sRGB - Display", "Un-tone-mapped")

#: The partition ``verify`` reproduces: the 111 closed-form transforms
#: §spec:op-coverage measured, plus the 20 carrying a ``Lut1D`` over either
#: domain in either direction. What is left refuses on ``FixedFunction``.
VERIFIED = 131
REFUSED = 28
TOTAL = 159

#: ``verify`` and the census now refuse the same transforms. They did not
#: while the ``Lut1D`` emitter refused a direction: direction is a parameter
#: of a registered op, so the census counted 28 where ``verify`` counted 36.
CENSUS_REFUSED = REFUSED


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


def test_compile_takes_a_display_view(graph):
    display, view = OPEN_VIEW
    assert main(["compile", "--display", display, "--view", view, "-o", graph]) == OK
    recorded = {entry.key: entry.value for entry in onnx.load(graph).metadata_props}
    assert recorded[f"{METADATA_PREFIX}endpoints"].endswith(f"{display} / {view}")


def test_compile_takes_a_config(graph):
    argv = ["compile", "--from", PAIR[0], "--to", PAIR[1], "-o", graph]
    assert main([*argv, "--config", DEFAULT_CONFIG]) == OK


def test_a_display_view_carrying_the_aces_output_transform_refuses(graph, capsys):
    """The ROADMAP's second Verify item. A refusal is an answer, not a crash:
    the message names the style, the transform, and its endpoints, and no
    traceback reaches the caller."""
    display, view = ACES_VIEW
    code = main(["compile", "--display", display, "--view", view, "-o", graph])
    assert code == WILL_NOT_EMIT

    err = capsys.readouterr().err
    assert ACES_STYLE in err
    assert f"{display} / {view}" in err
    assert "Traceback" not in err


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
    """The ROADMAP's third Verify item. Every one of the 159 transforms lands
    in exactly one bucket, and the identity is printed rather than assumed."""
    assert main(["verify"]) == OK
    printed = capsys.readouterr().out

    assert buckets(printed) == {
        "verified": VERIFIED,
        "refused": REFUSED,
        "failed": 0,
        "skipped": 0,
        "total": TOTAL,
    }
    assert VERIFIED + REFUSED == TOTAL

    assert f"24  {ACES_STYLE}" in printed
    assert "5  FixedFunction[REC2100_SURROUND]" in printed
    assert "Lut1D" not in printed
    assert "worst deviation" in printed


def test_census_is_reachable_from_the_cli(capsys):
    assert main(["census"]) == OK
    printed = capsys.readouterr().out
    assert f"transforms measured: {TOTAL}" in printed
    assert f"refused: {CENSUS_REFUSED}/{TOTAL}" in printed


def test_an_unknown_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["decompile"])
    assert caught.value.code == USAGE


def test_failure_is_distinct_from_refusal_and_from_success():
    assert len({OK, FAILED, USAGE, CANNOT_RESOLVE, WILL_NOT_EMIT}) == 5
