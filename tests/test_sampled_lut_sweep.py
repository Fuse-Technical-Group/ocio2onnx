"""Every transform in the pinned config carrying a ``Lut1D``, against the
oracle (§spec:op-coverage, §spec:verification).

The sampled-lookup acceptance test, and the counterpart to
``test_closed_form_sweep``: between them the two modules assert the whole
partition. 111 transforms are closed-form, 20 carry a table, 28 refuse on
`FixedFunction`, and 111 + 20 + 28 is the config. So "all 131
non-``FixedFunction`` transforms verify" is asserted rather than sampled, and
the 28 that refuse are shown to refuse on that op alone rather than on
anything this compiler could have emitted.

The sweep is over the 20 alone. The 111 are their own module's subject, and
`ocio2onnx verify` runs all 159 once in ``test_cli``; a third full sweep would
buy nothing but runtime.
"""

import pytest

from ocio2onnx.addressing import enumerate_transforms
from ocio2onnx.compiler import op_names, unsupported_ops
from ocio2onnx.emitters import supported_ops

#: Measured across the pinned config (§spec:op-coverage): the transforms built
#: from the six closed-form ops alone, those carrying a table, and those
#: refused. Nothing else is left.
CLOSED_FORM_TRANSFORMS = 111
SAMPLED_TRANSFORMS = 20
REFUSED_TRANSFORMS = 28
TOTAL_TRANSFORMS = 159

#: Of the 20, the ones whose ``Lut1D`` arrives inverse, which the compiler
#: inverts at compile time rather than searching at run time
#: (§spec:op-emission).
INVERSE_TRANSFORMS = 8

#: The one op every refusal names. A refusal naming anything else would mean
#: the compiler lost coverage it claims (§spec:op-coverage).
REFUSED_OP = "FixedFunction"


def is_inverse(transform):
    return str(transform.getDirection()).endswith("INVERSE")


@pytest.fixture(scope="module")
def partition(config):
    """The config split into transforms that carry a table, transforms that
    refuse, and everything else."""
    sampled, refused, closed_form = [], [], []
    for label, processor in enumerate_transforms(config):
        if unsupported_ops(processor):
            refused.append((label, processor))
        elif "Lut1D" in op_names(processor):
            sampled.append((label, processor))
        else:
            closed_form.append((label, processor))
    return sampled, refused, closed_form


def test_the_partition_accounts_for_every_transform(partition):
    sampled, refused, closed_form = partition
    assert len(sampled) == SAMPLED_TRANSFORMS
    assert len(refused) == REFUSED_TRANSFORMS
    assert len(closed_form) == CLOSED_FORM_TRANSFORMS
    assert len(sampled) + len(refused) + len(closed_form) == TOTAL_TRANSFORMS


def test_the_sweep_reaches_both_directions(partition, config):
    """Named so a direction vanishing from the partition is visible rather
    than silent: the forward ops alone would still let the sweep pass."""
    directions = [
        is_inverse(transform)
        for _, processor in partition[0]
        for transform in processor.createGroupTransform()
        if type(transform).__name__ == "Lut1DTransform"
    ]
    assert directions.count(True) == INVERSE_TRANSFORMS
    assert directions.count(False) == SAMPLED_TRANSFORMS - INVERSE_TRANSFORMS


def test_every_transform_carrying_a_table_verifies(partition, check):
    """20 of 20, none skipped. A failure names the transform, the input, and
    both values, so the reading at fault is identifiable without a debugger."""
    verified = 0
    failures = []
    for label, processor in partition[0]:
        result = check(processor, label)
        if result.ok:
            verified += 1
        else:
            failures.append(f"{label}: {result}")

    assert not failures, "\n".join(failures)
    assert verified == SAMPLED_TRANSFORMS


def test_the_refused_transforms_refuse_on_one_op_and_it_is_not_emittable(partition):
    """The section's second criterion, stated as what is left rather than as
    what passed: nothing refuses on an op the compiler claims to emit, so the
    28 are the display rendering work (§road:display-rendering) and nothing
    else."""
    named = {
        op.split("[")[0]
        for _, processor in partition[1]
        for op in unsupported_ops(processor)
    }
    assert named == {REFUSED_OP}
    assert REFUSED_OP not in supported_ops()
