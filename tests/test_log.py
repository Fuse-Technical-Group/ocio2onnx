"""``Log`` emits a logarithm forward and a power inverse (§spec:op-emission).

Both ``Log`` ops in the pinned config sit beside a ``Lut1D`` — ADX10 and
ADX16 — so none of the 111 closed-form transforms exercises this emitter, and
the config's own parameters are verified one op at a time instead.

The floor under the logarithm is the whole subtlety: OCIO clamps its argument
to the smallest normal float32, not to some round number, and a floor an order
of magnitude away is visible against the oracle at the first negative input.
"""

import math

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import enumerate_transforms
from ocio2onnx.oracle import cpu_reference, run_graph

BASES = (2.0, math.e, 10.0)


def log_transform(base, *, inverse=False):
    """A bare ``LogTransform`` at one base."""
    transform = OCIO.LogTransform()
    transform.setBase(base)
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def row(*values):
    """A lattice-shaped sample carrying ``values`` in every channel."""
    return np.array([[[list(values)]] * 3], dtype=np.float32)


@pytest.fixture(scope="module")
def config_log_ops(config):
    """Every ``Log`` op the pinned config carries, both directions."""
    return [
        transform
        for _, processor in enumerate_transforms(config)
        for transform in processor.createGroupTransform()
        if type(transform).__name__ == "LogTransform"
    ]


def test_the_registry_carries_log():
    assert "LogTransform" in emitters.REGISTRY


def test_no_closed_form_transform_carries_log(config_log_ops):
    """Recorded because it is why this module builds its own transforms: the
    sweep over the closed-form set cannot reach this emitter."""
    assert len(config_log_ops) == 4


def test_every_log_in_the_pinned_config_verifies(config_log_ops, check_transform):
    failures = [
        str(result)
        for transform in config_log_ops
        if not (result := check_transform(transform)).ok
    ]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("inverse", (False, True))
def test_a_bare_log_verifies(base, inverse, check_transform):
    result = check_transform(log_transform(base, inverse=inverse))
    assert result.ok, str(result)


def test_the_forward_direction_is_the_logarithm_in_its_base(config, compile_bare):
    transform = log_transform(10.0)
    got = run_graph(compile_bare(config.getProcessor(transform)), row(1.0, 10.0, 100.0))
    assert got.ravel()[:3] == pytest.approx([0.0, 1.0, 2.0], abs=1e-5)


def test_the_inverse_direction_raises_the_base(config, compile_bare):
    transform = log_transform(10.0, inverse=True)
    got = run_graph(compile_bare(config.getProcessor(transform)), row(0.0, 1.0, 2.0))
    assert got.ravel()[:3] == pytest.approx([1.0, 10.0, 100.0], rel=1e-5)


def test_the_floor_matches_the_one_ocio_clamps_to(config, compile_bare):
    """A floor of 1e-30 would read as nearly 8 units of error at the first
    non-positive input; OCIO clamps to the smallest normal float32."""
    processor = config.getProcessor(log_transform(10.0))
    samples = row(-100.0, -1e-3, 0.0)
    want = cpu_reference(processor, samples)
    got = run_graph(compile_bare(processor), samples)

    assert np.isfinite(got).all()
    assert got == pytest.approx(want, abs=2e-5)
    assert got.ravel()[0] == pytest.approx(math.log10(emitters.LOG_FLOOR), abs=1e-4)


def test_the_forward_direction_declares_zero_as_its_breakpoint():
    assert emitters.breakpoints(log_transform(10.0)) == [0.0]


def test_the_inverse_direction_declares_no_breakpoint():
    """``base ** x`` is one expression over the whole line."""
    assert emitters.breakpoints(log_transform(10.0, inverse=True)) == []


def test_zero_is_where_the_forward_branch_switches(config, compile_bare):
    """Below the breakpoint the clamp holds the output at the floor; above it
    the output tracks the input."""
    got = run_graph(
        compile_bare(config.getProcessor(log_transform(10.0))),
        row(-1e-6, -1e-9, 1e-6, 1e-3),
    ).ravel()[:4]

    assert got[0] == pytest.approx(got[1], abs=1e-5)
    assert got[2] == pytest.approx(-6.0, abs=1e-5)
    assert got[3] == pytest.approx(-3.0, abs=1e-5)
    assert got[0] < got[2]
