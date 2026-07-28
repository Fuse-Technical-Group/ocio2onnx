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
    Resolved,
    enumerate_transforms,
    load_config,
)
from ocio2onnx.builder import CHANNELS
from ocio2onnx.compiler import compile_processor
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
    """Compile a processor a test discovered rather than resolved."""

    def compile(processor, label="bare"):
        return compile_processor(
            Resolved(
                processor=processor,
                config_name="bare",
                config_uri=config_uri,
                endpoints=label,
            )
        )

    return compile


@pytest.fixture(scope="session")
def check(compile_bare):
    """Hold one processor's emitted graph against the oracle.

    Compiles one unless the caller hands over the model it already holds, as
    ``oracle.verify`` does — so a test of the public API verifies the artifact
    that API returned rather than a second compilation of the same request.
    """

    def check(processor, label="bare", model=None):
        samples = lattice(processor)
        return compare(
            cpu_reference(processor, samples),
            run_graph(
                compile_bare(processor, label) if model is None else model, samples
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
    """Every op of one OCIO transform class the pinned config carries.

    Walking all 159 transforms is the expensive part, so each class is walked
    once per session and the answer kept.
    """
    found: dict[str, list] = {}

    def config_ops(class_name):
        if class_name not in found:
            found[class_name] = [
                transform
                for _, processor in enumerate_transforms(config)
                for transform in processor.createGroupTransform()
                if type(transform).__name__ == class_name
            ]
        return found[class_name]

    return config_ops


@pytest.fixture(scope="session")
def op_in(config):
    """The one op of a class inside a single config transform."""

    def op_in(class_name, src, dst):
        ops = [
            transform
            for transform in config.getProcessor(src, dst).createGroupTransform()
            if type(transform).__name__ == class_name
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
