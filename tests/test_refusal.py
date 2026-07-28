"""Refusing an op before anything is emitted (§spec:op-coverage).

The check is written against the supported set rather than against a list of
known-bad ops, so it is a statement about what the compiler emits. An op a
future OCIO release adds is refused because nothing registered an emitter for
it, not dropped because nothing recognised it (§spec:op-coverage).

Every subject here is built rather than found. The pinned config carries
nothing this compiler refuses, so the boundary is not visible from inside it —
and the boundary is for the arbitrary config a caller hands over
(§spec:problem-statement), which is what these transforms stand in for.
"""

import PyOpenColorIO as OCIO
import pytest

from ocio2onnx.addressing import (
    DEFAULT_CONFIG,
    Resolved,
    resolve_colorspaces,
    resolve_display_view,
)
from ocio2onnx.compiler import compile_processor, op_names, unsupported_ops
from ocio2onnx.emitters import REGISTRY, UnsupportedOpError, op_label, supported_ops

#: The pair the compiler emits.
PAIR = ("Log3G10 REDWideGamutRGB", "ACES2065-1")

#: A hue adjustment the ``Lut1D`` emitter has no path for. ``Lut1D`` is
#: registered, so an op list sees nothing wrong — a refusal only the emitter
#: can raise.
LUT_HUE_ADJUST = OCIO.HUE_DW3

#: A `FixedFunction` style with no emitter, and how a refusal spells it. Both
#: styles the pinned config uses are emitted, so this is the class's third
#: behaviour standing in for the op an OCIO release has not shipped yet.
UNEMITTED_STYLE = OCIO.FIXED_FUNCTION_ACES_RED_MOD_03
UNEMITTED_LABEL = "FixedFunction[ACES_RED_MOD_03]"

#: Two unimplemented ops in one transform, the second behind the first.
#: Emitting until something unsupported turns up would name one and stop.
CROWDED_OPS = ["GradingPrimary", "GradingTone"]


@pytest.fixture
def bare(config):
    """Resolve a transform a test built rather than a config supplied."""

    def bare(transform, endpoints="bare"):
        return Resolved(
            processor=config.getProcessor(transform),
            config_name=config.getName(),
            config_uri=DEFAULT_CONFIG,
            endpoints=endpoints,
        )

    return bare


@pytest.fixture
def crowded(bare):
    """A transform carrying two ops the compiler has no emitter for."""
    group = OCIO.GroupTransform()
    group.appendTransform(OCIO.GradingPrimaryTransform())
    group.appendTransform(OCIO.GradingToneTransform())
    return bare(group, "crowded")


@pytest.fixture
def unemitted_style(bare):
    """A transform carrying the `FixedFunction` style nothing emits."""
    return bare(OCIO.FixedFunctionTransform(UNEMITTED_STYLE), "unemitted style")


def test_a_transform_the_compiler_emits_refuses_nothing(config):
    resolved = resolve_colorspaces(config, *PAIR, uri=DEFAULT_CONFIG)
    assert unsupported_ops(resolved.processor) == []


def test_the_supported_set_is_read_off_the_registry():
    assert supported_ops() == set(REGISTRY)
    assert supported_ops() == {
        "Matrix",
        "Range",
        "Exponent",
        "ExponentWithLinear",
        "Log",
        "LogCamera",
        "Lut1D",
        "FixedFunction[REC2100_SURROUND]",
        "FixedFunction[ACES_OUTPUT_TRANSFORM_20]",
        "ExposureContrast",
        "CDL",
    }


def test_an_op_absent_from_the_registry_is_refused(config, monkeypatch):
    """The check derives from the registry, not from a list of known-bad ops.

    Removing an emitter is the cheapest stand-in for an op OCIO has not
    shipped yet: nothing else in the compiler mentions ``Matrix``, so a
    transform built from it can only be refused by reading the registry.
    """
    resolved = resolve_colorspaces(config, *PAIR, uri=DEFAULT_CONFIG)
    monkeypatch.delitem(REGISTRY, "Matrix")
    assert "Matrix" in unsupported_ops(resolved.processor)


def test_an_op_type_this_compiler_has_never_seen_is_refused(config):
    """An OCIO op no workstream has reached yet. ``GRADING_*`` is deferred
    behind a named trigger and refuses by this mechanism until then
    (§road:grading-curves)."""
    processor = config.getProcessor(OCIO.GradingPrimaryTransform())
    assert unsupported_ops(processor) == ["GradingPrimary"]


def test_the_refusal_names_the_fixed_function_style(unemitted_style):
    """`FixedFunction` alone cannot say which behaviour of the class blocked
    the caller, and the class is emitted for two of them. Refusing by the class
    would refuse those too; refusing by the label refuses one style
    (§spec:op-coverage)."""
    assert unsupported_ops(unemitted_style.processor) == [UNEMITTED_LABEL]
    assert "FixedFunction" not in supported_ops()


def test_a_refusal_an_op_list_cannot_see_is_raised_at_emission(bare):
    """``Lut1D`` is registered, so the op list passes this transform; the
    emitter refuses a parameter of it on reading the op. A caller meets one
    refusal, whichever layer raised it."""
    lut = OCIO.Lut1DTransform(length=2)
    lut.setHueAdjust(LUT_HUE_ADJUST)
    resolved = bare(lut)
    assert unsupported_ops(resolved.processor) == []
    with pytest.raises(UnsupportedOpError, match="hue adjust"):
        compile_processor(resolved)


def test_op_names_are_the_labels_the_registry_is_keyed_on(config):
    processor = config.getProcessor("Log3G10 REDWideGamutRGB", "ACES2065-1")
    assert op_names(processor) == ["LogCamera", "Matrix"]
    assert set(op_names(processor)) <= supported_ops()


def test_op_label_reads_the_transform_rather_than_the_op_list():
    assert op_label(OCIO.RangeTransform()) == "Range"
    style = OCIO.FixedFunctionTransform(
        OCIO.FIXED_FUNCTION_REC2100_SURROUND, params=[1.2]
    )
    assert op_label(style) == "FixedFunction[REC2100_SURROUND]"


def test_every_unimplemented_op_is_named_not_only_the_first(crowded):
    """The refusal cannot depend on op order. Emission would stop at the first
    op and never reach the one behind it."""
    assert unsupported_ops(crowded.processor) == CROWDED_OPS


def test_the_refusal_names_the_ops_the_transform_and_its_endpoints(unemitted_style):
    with pytest.raises(UnsupportedOpError) as caught:
        compile_processor(unemitted_style)
    message = str(caught.value)
    assert UNEMITTED_LABEL in message
    assert unemitted_style.endpoints in message
    assert "refused rather than approximated" in message


def test_the_refusal_names_several_ops_in_one_message(crowded):
    with pytest.raises(UnsupportedOpError) as caught:
        compile_processor(crowded)
    for op in CROWDED_OPS:
        assert op in str(caught.value)


def test_an_empty_op_list_is_a_graph_rather_than_a_refusal(config):
    resolved = resolve_display_view(config, "sRGB - Display", "Raw", uri=DEFAULT_CONFIG)
    assert unsupported_ops(resolved.processor) == []
    assert compile_processor(resolved) is not None


def test_a_parameter_refusal_is_the_same_exception_type(bare):
    """An emitter refuses a parameter it has no path for. A caller catching a
    refusal should not have to care which layer raised it."""
    transform = OCIO.ExponentTransform()
    transform.setNegativeStyle(OCIO.NEGATIVE_CLAMP)
    resolved = bare(transform)
    assert unsupported_ops(resolved.processor) == []
    with pytest.raises(UnsupportedOpError, match="negative style CLAMP"):
        compile_processor(resolved)
