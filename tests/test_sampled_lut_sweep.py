"""Every transform in the pinned config carrying a ``Lut1D``, against the
oracle (§spec:op-coverage, §spec:verification).

The sampled-lookup acceptance test, and the counterpart to
``test_closed_form_sweep``: between them the two modules and
``test_aces_output_transform`` assert the whole partition. 111 transforms are
closed-form, 40 carry a table, 8 carry a fixed function and no table, and
111 + 40 + 8 is the config. So "all 159 verify" is asserted rather than
sampled, and nothing is left over to be quietly dropped.

Four of the 40 are the HLG transforms `REC2100_SURROUND` unblocked, and
seventeen more are ACES display renderings reaching a PQ or HLG display
(§spec:op-coverage). They land here rather than in their own modules because
each pairs a fixed function with a curve OCIO ships as a table.

The sweep is over the 40 alone. The 111 are their own module's subject, and
`ocio2onnx verify` runs all 159 once in ``test_cli``; a third full sweep would
buy nothing but runtime.
"""

import pytest

from ocio2onnx.addressing import enumerate_transforms
from ocio2onnx.compiler import op_names, unsupported_ops

#: Measured across the pinned config (§spec:op-coverage): the transforms built
#: from the closed-form ops alone, those carrying a table, and those carrying a
#: fixed function without one. Nothing else is left, and nothing refuses.
CLOSED_FORM_TRANSFORMS = 111
SAMPLED_TRANSFORMS = 40
FIXED_FUNCTION_TRANSFORMS = 8
TOTAL_TRANSFORMS = 159

#: Every ``Lut1D`` reaching the compiler is forward and half-domain but three,
#: which are uniform. None arrives inverse: `OPTIMIZATION_LUT_INV_FAST` is set,
#: so OCIO rewrites an invertible inverse table into a forward half-domain one
#: before the compiler sees it (§spec:op-emission).
HALF_DOMAIN_TRANSFORMS = 37
UNIFORM_TRANSFORMS = 3


def luts(processor):
    return [
        transform
        for transform in processor.createGroupTransform()
        if type(transform).__name__ == "Lut1DTransform"
    ]


@pytest.fixture(scope="module")
def partition(config):
    """The config split into transforms that carry a table, transforms that
    carry a fixed function and no table, and everything else."""
    sampled, fixed_function, closed_form = [], [], []
    for label, processor in enumerate_transforms(config):
        ops = op_names(processor)
        if "Lut1D" in ops:
            sampled.append((label, processor))
        elif any(op.startswith("FixedFunction") for op in ops):
            fixed_function.append((label, processor))
        else:
            closed_form.append((label, processor))
    return sampled, fixed_function, closed_form


def test_the_partition_accounts_for_every_transform(partition):
    sampled, fixed_function, closed_form = partition
    assert len(sampled) == SAMPLED_TRANSFORMS
    assert len(fixed_function) == FIXED_FUNCTION_TRANSFORMS
    assert len(closed_form) == CLOSED_FORM_TRANSFORMS
    assert len(sampled) + len(fixed_function) + len(closed_form) == TOTAL_TRANSFORMS


def test_nothing_in_the_pinned_config_refuses(partition):
    """Op coverage is complete for this config (§spec:op-coverage). Asserted
    over the partition rather than over the sweep's own subset, so a refusal
    reappearing anywhere in the config fails here rather than going unnoticed
    in a bucket nothing walks."""
    named = {
        op
        for group in partition
        for _, processor in group
        for op in unsupported_ops(processor)
    }
    assert named == set()


def test_the_sweep_reaches_both_domains(partition):
    """Named so a domain vanishing from the partition is visible rather than
    silent: the half-domain ops alone would still let the sweep pass, and they
    are indexed by an arithmetic reconstruction the uniform ones never run."""
    domains = [
        bool(lut.getInputHalfDomain())
        for _, processor in partition[0]
        for lut in luts(processor)
    ]
    assert domains.count(True) == HALF_DOMAIN_TRANSFORMS
    assert domains.count(False) == UNIFORM_TRANSFORMS


def test_no_table_reaches_the_compiler_inverse(partition):
    """The finding this workstream turned up. The compiler inverts a table at
    compile time rather than searching at run time (§spec:op-emission), but no
    table in this config arrives needing it: the compiler and the oracle now
    read one optimized op list, and that optimization pre-inverts every
    invertible table. The compile-time inversion serves op lists that arrive
    inverse, which ``test_lut1d`` builds directly."""
    directions = [
        str(lut.getDirection()).endswith("INVERSE")
        for _, processor in partition[0]
        for lut in luts(processor)
    ]
    assert directions.count(True) == 0
    assert directions.count(False) == SAMPLED_TRANSFORMS


def test_every_transform_carrying_a_table_verifies(partition, check):
    """40 of 40, none skipped. A failure names the transform, the input, and
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


def test_every_transform_carrying_a_fixed_function_alone_verifies(partition, check):
    """The eight the ACES output transform unblocked that need no table at all
    — OCIO's own shader bakes their encoding into a texture and this compiler
    evaluates it (§spec:op-coverage)."""
    failures = [
        f"{label}: {result}"
        for label, processor in partition[1]
        if not (result := check(processor, label)).ok
    ]
    assert not failures, "\n".join(failures)
