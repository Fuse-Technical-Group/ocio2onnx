"""Fixtures shared across the suite.

Loading a built-in config costs enough to be worth doing once per session.
The oracle wrappers live here too: every emitter test asks the same question —
does this processor's emitted graph agree with OCIO's CPU processor — and only
the processor changes.
"""

import numpy as np
import pytest

from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    OPTIMIZATION_FLAGS,
    Resolved,
    enumerate_transforms,
    load_config,
)
from ocio2onnx.builder import CHANNELS
from ocio2onnx.compiler import compile_processor
from ocio2onnx.emitters import op_label
from ocio2onnx.oracle import compare, cpu_reference, lattice, run_graph


@pytest.fixture(scope="session")
def config_uri() -> str:
    """The URI of the config the specification's numbers were measured against."""
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def config(config_uri):
    """The pinned config, loaded once."""
    return load_config(config_uri)


@pytest.fixture(scope="session")
def compile_bare(config_uri):
    """Compile a processor a test discovered rather than resolved.

    Optimized at ``OPTIMIZATION_FLAGS`` first, which is what ``resolve`` hands
    the compiler. A test that compiled the unoptimized op list would exercise
    ops no caller reaches and measure them against an oracle that optimizes
    (``oracle.cpu_reference``) — two different transforms, agreeing only where
    the optimizer happens to be value-preserving.
    """

    def compile(processor, label="bare"):
        return compile_processor(
            Resolved(
                processor=processor.getOptimizedProcessor(OPTIMIZATION_FLAGS),
                config_name="bare",
                config_uri=config_uri,
                endpoints=label,
            )
        )

    return compile


@pytest.fixture(scope="session")
def emitted_ops():
    """The ops a processor compiles from, which is not the list it reports.

    A test that reads an op's own table or coefficients has to read them off
    the op the compiler emitted. ``OPTIMIZATION_LUT_INV_FAST`` turns an inverse
    ``Lut1D`` into a forward one carrying a different table, so the reported
    transform and the emitted one answer the same pixel from different numbers.
    """

    def emitted_ops(processor, label=None):
        transforms = processor.getOptimizedProcessor(
            OPTIMIZATION_FLAGS
        ).createGroupTransform()
        return [t for t in transforms if label is None or op_label(t) == label]

    return emitted_ops


@pytest.fixture(scope="session")
def check(compile_bare):
    """Hold one processor's emitted graph against the oracle.

    Compiles one unless the caller hands over the model it already holds, as
    ``oracle.verify`` does — so a test of the public API verifies the artifact
    that API returned rather than a second compilation of the same request.

    ``parameters`` binds the graph's live inputs (§spec:dynamic-properties).
    A sweep hands over one graph and a processor OCIO built at the swept value,
    which is the claim a live parameter makes: the same artifact, not a
    recompilation.
    """

    def check(processor, label="bare", model=None, parameters=None):
        samples = lattice(processor)
        return compare(
            cpu_reference(processor, samples),
            run_graph(
                compile_bare(processor, label) if model is None else model,
                samples,
                parameters,
            ),
            samples,
        )

    return check


@pytest.fixture(scope="session")
def check_transform(config, check):
    """Hold one bare transform's emitted graph against the oracle.

    Where the pinned config supplies an op in only one direction, the other is
    reached by handing OCIO a transform object and comparing against the
    processor OCIO builds from it.
    """

    def check_transform(transform):
        return check(config.getProcessor(transform), type(transform).__name__)

    return check_transform


@pytest.fixture(scope="session")
def config_ops(config):
    """Every op the pinned config carries under one ``op_label``.

    Walking all 159 transforms is the expensive part, so each label is walked
    once per session and the answer kept.
    """
    found: dict[str, list] = {}

    def config_ops(label):
        if label not in found:
            found[label] = [
                transform
                for _, processor in enumerate_transforms(config)
                for transform in processor.createGroupTransform()
                if op_label(transform) == label
            ]
        return found[label]

    return config_ops


@pytest.fixture(scope="session")
def op_in(config):
    """The one op under a label inside a single config transform."""

    def op_in(label, src, dst):
        ops = [
            transform
            for transform in config.getProcessor(src, dst).createGroupTransform()
            if op_label(transform) == label
        ]
        assert len(ops) == 1
        return ops[0]

    return op_in


@pytest.fixture
def row():
    """A lattice-shaped sample carrying its arguments in every channel."""

    def row(*values):
        return np.array([[[list(values)]] * CHANNELS], dtype=np.float32)

    return row
