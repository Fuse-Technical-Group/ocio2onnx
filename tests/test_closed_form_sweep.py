"""Every closed-form transform in the pinned config, against the oracle
(§road:closed-form-emitters, §spec:verification).

This is the workstream's acceptance test. Coverage is asserted rather than
sampled: the count is fixed at the number §spec:op-coverage measured, every
one of them must verify, and none may be skipped. A transform that stopped
being closed-form, or an emitter that stopped emitting, moves the count.
"""

import pytest

from ocio2onnx.addressing import enumerate_transforms, op_names

#: The six ops this workstream emits (§spec:op-coverage).
CLOSED_FORM_OPS = frozenset(
    {"Matrix", "Range", "Exponent", "ExponentWithLinear", "Log", "LogCamera"}
)

#: Measured across the pinned config: 111 of its 159 transforms are built from
#: those six alone (§spec:op-coverage).
CLOSED_FORM_TRANSFORMS = 111
TOTAL_TRANSFORMS = 159

#: Two of the six are never the whole of a transform: every ``Range`` and
#: every ``Log`` in this config sits beside a ``Lut1D``. Their emitters are
#: verified op by op in their own modules, not by this sweep.
OPS_REACHED_BY_THE_SWEEP = frozenset(
    {"Matrix", "Exponent", "ExponentWithLinear", "LogCamera"}
)


@pytest.fixture(scope="module")
def closed_form(config):
    """Every transform in the pinned config built from the six ops alone."""
    return [
        (label, processor, ops)
        for label, processor in enumerate_transforms(config)
        if set(ops := op_names(processor)) <= CLOSED_FORM_OPS
    ]


def test_the_config_holds_the_transforms_the_specification_counted(config):
    assert sum(1 for _ in enumerate_transforms(config)) == TOTAL_TRANSFORMS


def test_the_closed_form_partition_is_the_counted_size(closed_form):
    assert len(closed_form) == CLOSED_FORM_TRANSFORMS


def test_the_sweep_reaches_the_ops_it_claims_to(closed_form):
    """Named so a change in what the sweep covers is visible rather than
    silent: an op that vanished from the partition would still let every
    remaining transform pass."""
    reached = {op for _, _, ops in closed_form for op in ops}
    assert reached == OPS_REACHED_BY_THE_SWEEP


def test_every_closed_form_transform_verifies(closed_form, check):
    """111 of 111, none skipped. A failure names the transform, the input, and
    both values, so the reading at fault is identifiable without a debugger."""
    verified = 0
    failures = []
    for label, processor, _ in closed_form:
        result = check(processor, label)
        if result.ok:
            verified += 1
        else:
            failures.append(f"{label}: {result}")

    assert not failures, "\n".join(failures)
    assert verified == len(closed_form) == CLOSED_FORM_TRANSFORMS
