"""Fixtures shared across the suite.

Loading a built-in config costs enough to be worth doing once per session.
The oracle wrappers live here too: every emitter test asks the same question —
does this processor's emitted graph agree with OCIO's CPU processor — and only
the processor changes.
"""

import pytest

from ocio2onnx.addressing import DEFAULT_CONFIG, Resolved, load_config
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
    """Hold one processor's emitted graph against the oracle."""

    def check(processor, label="bare"):
        samples = lattice(processor)
        return compare(
            cpu_reference(processor, samples),
            run_graph(compile_bare(processor, label), samples),
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
