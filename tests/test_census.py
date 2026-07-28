"""The coverage report (§spec:op-coverage, §road:coverage-report).

The census exists to be quoted, so its numbers are pinned here: a change in
what OCIO ships, or in what the compiler emits, moves them and says so. It
reports the compiler's own supported set rather than a second copy of it,
because a second copy drifts the moment an emitter is added.
"""

import pathlib
import subprocess
import sys

import pytest

from ocio2onnx import census
from ocio2onnx.addressing import DEFAULT_CONFIG, enumerate_transforms
from ocio2onnx.compiler import unsupported_ops
from ocio2onnx.emitters import supported_ops

#: Measured across the pinned config (§spec:op-coverage).
OP_CENSUS = {
    "Matrix": 284,
    "Range": 52,
    "Lut1D": 40,
    "FixedFunction": 29,
    "Exponent": 26,
    "ExponentWithLinear": 24,
    "LogCamera": 24,
    "Log": 4,
}
TOTAL_TRANSFORMS = 159

#: The census refuses by op type, so a transform whose ``Lut1D`` an emitter
#: refuses for its domain or its direction is not counted here. That refusal
#: is ``ocio2onnx verify``'s, which compiles (§spec:op-coverage).
REFUSED_TRANSFORMS = 28
NEEDS_LUT = 48

#: The script SPEC.md §spec:op-coverage names by path.
SHIM = pathlib.Path(__file__).parents[1] / "tools" / "census.py"


@pytest.fixture(scope="module")
def taken(config):
    return census.take(config, DEFAULT_CONFIG)


def test_the_op_census_is_the_number_the_specification_quotes(taken):
    assert dict(taken.ops) == OP_CENSUS
    assert taken.total == TOTAL_TRANSFORMS


def test_the_lut_partition_is_ocios_own(taken):
    """Where OCIO itself bakes a sampled texture. No transform in this config
    needs a 3D one."""
    assert len(taken.needs_lut) == NEEDS_LUT
    assert [entry for entry in taken.needs_lut if entry[2]] == []


def test_the_census_refuses_exactly_what_the_compiler_refuses(taken, config):
    """One supported set, not two. ``SUPPORTED`` used to live here and had
    already drifted: it listed ``Lut1D`` while no emitter implemented one, so
    the census reported 28 refusals where the compiler made 48."""
    expected = [
        (label, unsupported_ops(processor))
        for label, processor in enumerate_transforms(config)
    ]
    assert taken.refusals == [entry for entry in expected if entry[1]]
    assert len(taken.refusals) == REFUSED_TRANSFORMS


def test_the_census_marks_the_ops_the_compiler_emits(taken):
    assert taken.supported == supported_ops()
    assert "Lut1D" in taken.supported
    assert "FixedFunction" not in taken.supported


def test_refusals_group_by_op(taken):
    """A transform carrying two unimplemented ops counts under both, so these
    sum above the number of refused transforms."""
    grouped = dict(census.group_by_op(taken.refusals))
    assert grouped == {
        "FixedFunction[ACES_OUTPUT_TRANSFORM_20]": 24,
        "FixedFunction[REC2100_SURROUND]": 5,
    }
    assert sum(grouped.values()) > REFUSED_TRANSFORMS


def test_the_report_prints_the_partition(capsys):
    assert census.report(DEFAULT_CONFIG) == 0
    printed = capsys.readouterr().out
    assert f"transforms measured: {TOTAL_TRANSFORMS}" in printed
    assert f"refused: {REFUSED_TRANSFORMS}/{TOTAL_TRANSFORMS}" in printed


def test_the_script_spec_names_by_path_still_runs():
    result = subprocess.run(
        [sys.executable, str(SHIM)], capture_output=True, text=True, check=True
    )
    assert f"refused: {REFUSED_TRANSFORMS}/{TOTAL_TRANSFORMS}" in result.stdout
