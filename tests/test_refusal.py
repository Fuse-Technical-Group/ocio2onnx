"""Refusing an op before anything is emitted (§spec:op-coverage).

The check is written against the supported set rather than against a list of
known-bad ops, so it is a statement about what the compiler emits. An op a
future OCIO release adds is refused because nothing registered an emitter for
it, not dropped because nothing recognised it (§spec:op-coverage).
"""

import PyOpenColorIO as OCIO
import pytest

from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    Resolved,
    resolve_colorspaces,
    resolve_display_view,
)
from ocio2onnx.compiler import compile_processor, unsupported_ops
from ocio2onnx.emitters import REGISTRY, UnsupportedOpError, op_label, supported_ops

#: The pair the compiler emits.
PAIR = ("Log3G10 REDWideGamutRGB", "ACES2065-1")

#: The display view the compiler refuses, and the style that blocks it.
ACES_VIEW = ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")
ACES_STYLE = "FixedFunction[ACES_OUTPUT_TRANSFORM_20]"

#: A view carrying three unimplemented ops, the first of them four ops in.
#: Emitting until something unsupported turns up would name one and stop.
CROWDED_VIEW = ("Rec.2100-HLG - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)")
CROWDED_OPS = [ACES_STYLE, "FixedFunction[REC2100_SURROUND]", "Lut1D"]


@pytest.fixture
def resolve(config):
    """Resolve a display view against the pinned config."""

    def resolve(display, view):
        return resolve_display_view(config, display, view, uri=DEFAULT_CONFIG)

    return resolve


def test_a_transform_the_compiler_emits_refuses_nothing(config):
    resolved = resolve_colorspaces(config, *PAIR, uri=DEFAULT_CONFIG)
    assert unsupported_ops(resolved.processor) == []


def test_the_supported_set_is_read_off_the_registry():
    assert supported_ops() == {name.removesuffix("Transform") for name in REGISTRY}
    assert supported_ops() == {
        "Matrix",
        "Range",
        "Exponent",
        "ExponentWithLinear",
        "Log",
        "LogCamera",
    }


def test_an_op_absent_from_the_registry_is_refused(config, monkeypatch):
    """The check derives from the registry, not from a list of known-bad ops.

    Removing an emitter is the cheapest stand-in for an op OCIO has not
    shipped yet: nothing else in the compiler mentions ``Matrix``, so a
    transform built from it can only be refused by reading the registry.
    """
    resolved = resolve_colorspaces(config, *PAIR, uri=DEFAULT_CONFIG)
    monkeypatch.delitem(REGISTRY, "MatrixTransform")
    assert "Matrix" in unsupported_ops(resolved.processor)


def test_an_op_type_this_compiler_has_never_seen_is_refused(config):
    """An OCIO op no workstream has reached yet. ``GRADING_*`` is deferred
    behind a named trigger and refuses by this mechanism until then
    (§road:grading-curves)."""
    processor = config.getProcessor(OCIO.GradingPrimaryTransform())
    assert unsupported_ops(processor) == ["GradingPrimary"]


def test_the_refusal_names_the_fixed_function_style(resolve):
    """`FixedFunction` alone cannot say which of the two styles blocked the
    caller, and §spec:op-coverage sequences them into separate workstreams."""
    assert unsupported_ops(resolve(*ACES_VIEW).processor) == [ACES_STYLE]


def test_an_op_with_no_distinguishing_attribute_is_named_by_its_type(resolve):
    display, view = "Rec.2100-PQ - Display", "Un-tone-mapped"
    assert "Lut1D" in unsupported_ops(resolve(display, view).processor)


def test_op_label_reads_the_transform_rather_than_the_op_list():
    assert op_label(OCIO.RangeTransform()) == "Range"
    style = OCIO.FixedFunctionTransform(
        OCIO.FIXED_FUNCTION_REC2100_SURROUND, params=[1.2]
    )
    assert op_label(style) == "FixedFunction[REC2100_SURROUND]"


def test_every_unimplemented_op_is_named_not_only_the_first(resolve):
    """The refusal cannot depend on op order. Emission would stop at the
    fourth op and never reach the two behind it."""
    assert unsupported_ops(resolve(*CROWDED_VIEW).processor) == CROWDED_OPS


def test_the_refusal_names_the_ops_the_transform_and_its_endpoints(resolve):
    resolved = resolve(*ACES_VIEW)
    with pytest.raises(UnsupportedOpError) as caught:
        compile_processor(resolved)
    message = str(caught.value)
    assert ACES_STYLE in message
    assert resolved.endpoints in message
    assert "refused rather than approximated" in message


def test_the_refusal_names_several_ops_in_one_message(resolve):
    with pytest.raises(UnsupportedOpError) as caught:
        compile_processor(resolve(*CROWDED_VIEW))
    for op in CROWDED_OPS:
        assert op in str(caught.value)


def test_an_empty_op_list_is_a_graph_rather_than_a_refusal(config):
    resolved = resolve_display_view(config, "sRGB - Display", "Raw", uri=DEFAULT_CONFIG)
    assert unsupported_ops(resolved.processor) == []
    assert compile_processor(resolved) is not None


def test_a_parameter_refusal_is_the_same_exception_type(config):
    """An emitter refuses a parameter it has no path for. A caller catching a
    refusal should not have to care which layer raised it."""
    transform = OCIO.ExponentTransform()
    transform.setNegativeStyle(OCIO.NEGATIVE_CLAMP)
    resolved = Resolved(
        processor=config.getProcessor(transform),
        config_name=config.getName(),
        config_uri=DEFAULT_CONFIG,
        endpoints="bare",
    )
    assert unsupported_ops(resolved.processor) == []
    with pytest.raises(UnsupportedOpError, match="negative style CLAMP"):
        compile_processor(resolved)
