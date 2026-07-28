"""A compile request resolves to a processor, or is refused by name
(§spec:emitted-graph).
"""

import PyOpenColorIO as OCIO
import pytest

import ocio2onnx
from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    AddressError,
    enumerate_transforms,
    load_config,
    reference_space,
    resolve_colorspaces,
    resolve_display_view,
)
from ocio2onnx.compiler import op_names

PINNED_NAME = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
DISPLAY = "sRGB - Display"
VIEW = "Un-tone-mapped"


def test_address_error_is_a_value_error():
    assert issubclass(AddressError, ValueError)


def test_default_config_is_a_versioned_built_in():
    assert DEFAULT_CONFIG.startswith("ocio://")
    assert DEFAULT_CONFIG.endswith(PINNED_NAME)


def test_load_config_by_uri_resolves_the_pinned_name(config):
    assert config.getName() == PINNED_NAME


def test_ocio_default_is_refused():
    with pytest.raises(AddressError, match="ocio://default"):
        load_config("ocio://default")


def test_unresolvable_uri_names_the_uri():
    with pytest.raises(AddressError, match="ocio://no-such-config"):
        load_config("ocio://no-such-config")


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "absent.ocio"
    with pytest.raises(AddressError, match=str(missing)):
        load_config(str(missing))


def test_reference_space_falls_back_when_the_role_is_empty(config):
    assert config.getCanonicalName(OCIO.ROLE_REFERENCE) == ""
    assert reference_space(config) == "ACES2065-1"


def test_resolve_colorspaces_returns_a_processor_with_ops(config, config_uri):
    resolved = resolve_colorspaces(
        config, "Log3G10 REDWideGamutRGB", "ACES2065-1", uri=config_uri
    )
    assert op_names(resolved.processor)
    assert resolved.config_name == PINNED_NAME
    assert resolved.config_uri == config_uri
    assert resolved.endpoints == "Log3G10 REDWideGamutRGB -> ACES2065-1"


def test_unknown_source_names_the_name_and_the_config(config, config_uri):
    with pytest.raises(AddressError) as excinfo:
        resolve_colorspaces(config, "Not A Space", "ACES2065-1", uri=config_uri)
    message = str(excinfo.value)
    assert "Not A Space" in message
    assert PINNED_NAME in message


def test_unknown_target_names_the_name_and_the_config(config, config_uri):
    with pytest.raises(AddressError) as excinfo:
        resolve_colorspaces(config, "ACES2065-1", "Not A Space", uri=config_uri)
    message = str(excinfo.value)
    assert "Not A Space" in message
    assert PINNED_NAME in message


def test_a_role_resolves(config, config_uri):
    resolved = resolve_colorspaces(config, "scene_linear", "ACES2065-1", uri=config_uri)
    assert op_names(resolved.processor)


def test_resolve_display_view_returns_a_processor(config, config_uri):
    resolved = resolve_display_view(config, DISPLAY, VIEW, uri=config_uri)
    assert op_names(resolved.processor)
    assert resolved.config_name == PINNED_NAME


def test_display_view_defaults_src_to_the_reference(config, config_uri):
    resolved = resolve_display_view(config, DISPLAY, VIEW, uri=config_uri)
    assert resolved.endpoints.startswith(f"{reference_space(config)} -> ")
    assert DISPLAY in resolved.endpoints
    assert VIEW in resolved.endpoints


def test_display_view_honors_an_explicit_src(config, config_uri):
    resolved = resolve_display_view(config, DISPLAY, VIEW, src="ACEScg", uri=config_uri)
    assert resolved.endpoints.startswith("ACEScg -> ")


def test_unknown_display_names_the_display(config, config_uri):
    with pytest.raises(AddressError) as excinfo:
        resolve_display_view(config, "Not A Display", VIEW, uri=config_uri)
    message = str(excinfo.value)
    assert "Not A Display" in message
    assert PINNED_NAME in message


def test_unknown_view_names_the_view_and_its_display(config, config_uri):
    with pytest.raises(AddressError) as excinfo:
        resolve_display_view(config, DISPLAY, "Not A View", uri=config_uri)
    message = str(excinfo.value)
    assert "Not A View" in message
    assert DISPLAY in message


def test_unknown_display_view_src_names_the_src(config, config_uri):
    with pytest.raises(AddressError, match="Not A Space"):
        resolve_display_view(config, DISPLAY, VIEW, src="Not A Space", uri=config_uri)


def test_metadata_records_both_the_resolved_name_and_the_uri(config, config_uri):
    resolved = resolve_colorspaces(
        config, "Log3G10 REDWideGamutRGB", "ACES2065-1", uri=config_uri
    )
    metadata = resolved.metadata
    assert all(key.startswith("ocio2onnx.") for key in metadata)
    assert all(isinstance(value, str) for value in metadata.values())
    values = set(metadata.values())
    assert PINNED_NAME in values
    assert config_uri in values
    assert resolved.endpoints in values
    assert OCIO.GetVersion() in values
    assert ocio2onnx.__version__ in values


def test_metadata_never_says_ocio_default(config, config_uri):
    resolved = resolve_colorspaces(config, "ACEScg", "ACES2065-1", uri=config_uri)
    assert "ocio://default" not in "".join(resolved.metadata.values())


def test_metadata_leaves_precision_to_a_later_workstream(config, config_uri):
    resolved = resolve_colorspaces(config, "ACEScg", "ACES2065-1", uri=config_uri)
    assert not any("precision" in key for key in resolved.metadata)


def test_enumerate_transforms_yields_the_measured_census(config):
    labels = [label for label, _ in enumerate_transforms(config)]
    assert len(labels) == 159
    assert len(set(labels)) == 159
