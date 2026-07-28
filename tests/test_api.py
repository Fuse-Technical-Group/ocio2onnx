"""The package's public surface (§road:closed-form-emitters).

Two entry points, each loading a config, resolving a request against it, and
compiling the processor OCIO returns. They hold no logic of their own, so what
is worth testing is that they wire the right modules together and that what
they hand back verifies.
"""

import onnx
import pytest

import ocio2onnx
from ocio2onnx import (
    AddressError,
    UnsupportedOpError,
    compile_colorspaces,
    compile_display_view,
)
from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    METADATA_PREFIX,
    Resolved,
    op_names,
    reference_space,
    resolve_colorspaces,
    resolve_display_view,
)
from ocio2onnx.builder import PRECISION
from ocio2onnx.compiler import compile_processor
from ocio2onnx.oracle import compare, cpu_reference, lattice, run_graph
from ocio2onnx.oracle import verify as oracle_verify

#: The pair §road:compiler-core names as its verification: a camera log
#: encoding back to the reference.
PAIR = ("Log3G10 REDWideGamutRGB", "ACES2065-1")

#: A closed-form display view. The tone-mapped views on the same display carry
#: a FixedFunction and are refused.
DISPLAY_VIEW = ("sRGB - Display", "Un-tone-mapped")


def metadata(model):
    return {entry.key: entry.value for entry in model.metadata_props}


def verified(processor, model):
    """Hold a model handed back by the public API against the oracle."""
    samples = lattice(processor)
    return compare(
        cpu_reference(processor, samples), run_graph(model, samples), samples
    )


def test_the_package_exports_what_it_says_it_does():
    assert set(ocio2onnx.__all__) == {
        "AddressError",
        "DEFAULT_CONFIG",
        "UnsupportedOpError",
        "__version__",
        "compile_colorspaces",
        "compile_display_view",
        "verify",
    }
    for name in ocio2onnx.__all__:
        assert getattr(ocio2onnx, name) is not None


def test_the_re_exports_are_the_originals():
    assert ocio2onnx.DEFAULT_CONFIG == DEFAULT_CONFIG
    assert ocio2onnx.verify is oracle_verify
    assert issubclass(ocio2onnx.AddressError, ValueError)
    assert issubclass(ocio2onnx.UnsupportedOpError, NotImplementedError)


def test_compile_colorspaces_returns_a_model_that_verifies(config):
    model = compile_colorspaces(*PAIR)
    assert isinstance(model, onnx.ModelProto)
    result = verified(config.getProcessor(*PAIR), model)
    assert result.ok, str(result)
    assert result.compared > 0


def test_compile_colorspaces_records_its_provenance_and_precision():
    recorded = metadata(compile_colorspaces(*PAIR))
    assert recorded[f"{METADATA_PREFIX}config.uri"] == DEFAULT_CONFIG
    assert recorded[f"{METADATA_PREFIX}endpoints"] == f"{PAIR[0]} -> {PAIR[1]}"
    assert recorded[f"{METADATA_PREFIX}precision"] == PRECISION == "float32"


def test_compile_display_view_returns_a_model_that_verifies(config):
    model = compile_display_view(*DISPLAY_VIEW)
    processor = resolve_display_view(
        config, *DISPLAY_VIEW, uri=DEFAULT_CONFIG
    ).processor
    result = verified(processor, model)
    assert result.ok, str(result)
    assert result.compared > 0


def test_compile_display_view_defaults_its_source_to_the_reference(config):
    recorded = metadata(compile_display_view(*DISPLAY_VIEW))
    display, view = DISPLAY_VIEW
    expected = f"{reference_space(config)} -> {display} / {view}"
    assert recorded[f"{METADATA_PREFIX}endpoints"] == expected


def test_compile_display_view_takes_an_explicit_source(config):
    recorded = metadata(compile_display_view(*DISPLAY_VIEW, src="ACEScg"))
    display, view = DISPLAY_VIEW
    assert recorded[f"{METADATA_PREFIX}endpoints"] == f"ACEScg -> {display} / {view}"


def test_both_entry_points_accept_a_config(config_uri):
    assert compile_colorspaces(*PAIR, config=config_uri) is not None
    assert compile_display_view(*DISPLAY_VIEW, config=config_uri) is not None


def test_an_unresolvable_color_space_is_refused_against_the_loaded_config():
    with pytest.raises(AddressError, match="not in config"):
        compile_colorspaces("Not A Color Space", "ACES2065-1")


def test_an_unresolvable_view_is_refused_against_the_loaded_config():
    with pytest.raises(AddressError, match="not a view"):
        compile_display_view("sRGB - Display", "Not A View")


def test_an_unimplemented_op_is_refused_rather_than_approximated():
    with pytest.raises(UnsupportedOpError, match="FixedFunction"):
        compile_display_view("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")


def test_a_view_with_no_ops_compiles_to_a_pass_through(config):
    """``Raw`` resolves to an empty op list, which the builder closes over with
    an Identity rather than failing to make a graph."""
    processor = resolve_display_view(
        config, "sRGB - Display", "Raw", uri=DEFAULT_CONFIG
    ).processor
    assert op_names(processor) == []
    result = verified(processor, compile_display_view("sRGB - Display", "Raw"))
    assert result.ok, str(result)


def test_the_entry_points_hold_no_logic_of_their_own(config, config_uri):
    """Each is the composition of loading, resolving, and compiling — so a
    model built by hand out of those three parts is the same model."""
    resolved = resolve_colorspaces(config, *PAIR, uri=config_uri)
    assert isinstance(resolved, Resolved)
    assert (
        compile_colorspaces(*PAIR).SerializeToString()
        == compile_processor(resolved).SerializeToString()
    )
