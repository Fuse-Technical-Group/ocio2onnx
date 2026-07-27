"""Fixtures shared across the suite.

Loading a built-in config costs enough to be worth doing once per session.
"""

import pytest

from ocio2onnx.addressing import DEFAULT_CONFIG, load_config


@pytest.fixture(scope="session")
def config_uri() -> str:
    """The URI of the config the specification's numbers were measured against."""
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def config(config_uri):
    """The pinned config, loaded once."""
    return load_config(config_uri)
