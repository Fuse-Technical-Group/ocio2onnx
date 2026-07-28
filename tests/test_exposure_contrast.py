"""``ExposureContrast`` emits its three scalars as graph inputs
(§spec:dynamic-properties).

The claim a live parameter makes is not that the graph is right at its default
— a baked table manages that — but that one compiled artifact tracks OCIO
across the knob's whole range. So the test that matters here sweeps: one graph,
compiled once, held against a processor OCIO built at each swept value.

All three styles are exercised in both directions. Their arithmetic is read off
OCIO's own shader, as `Lut1D`'s half index is (§spec:op-emission): ``linear``
and ``video`` pivot a power, ``log`` is an affine, and the contrast floor and
the ``contrast == 1`` short circuit are OCIO's rather than tidied away.
"""

import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.builder import parameters
from ocio2onnx.compiler import UnsupportedOpError
from ocio2onnx.oracle import run_graph

EXPOSURE_CONTRAST = "ExposureContrast"

STYLES = (
    OCIO.EXPOSURE_CONTRAST_LINEAR,
    OCIO.EXPOSURE_CONTRAST_VIDEO,
    OCIO.EXPOSURE_CONTRAST_LOGARITHMIC,
)

#: The values the graph's three inputs default to, and what a sweep of each
#: moves away from. Off the identity, so a knob wired to nothing cannot agree
#: by accident.
DEFAULTS = {"Exposure": 0.4, "Contrast": 1.3, "Gamma": 1.1}

#: Stops either side of the default. A grade knob's range, not a lattice.
EXPOSURE_SWEEP = (-4.0, -1.25, 0.0, 0.75, 4.0)

#: Multipliers. Contrast and gamma reach the arithmetic as their product, and
#: the sweeps stop where the reference does rather than where the knob does:
#: inverted, the pivoted styles raise the picture to ``1 / (contrast * gamma)``,
#: and OCIO's ExposureContrast renderer uses an approximate power whose base
#: error that exponent multiplies. Measured against the CPU processor, the
#: product verifies at 0.11 and misses the tolerance by 1.1e-4 at 0.088
#: (`test_the_pivoted_inverse_gives_out_where_ocios_own_power_does`).
CONTRAST_SWEEP = (0.1, 0.4, 1.0, 2.5)
GAMMA_SWEEP = (0.1, 0.5, 1.0, 2.2)

#: Graph input, OCIO setter, and the range each is swept over.
SWEEPS = (
    ("EXPOSURE", "setExposure", EXPOSURE_SWEEP),
    ("CONTRAST", "setContrast", CONTRAST_SWEEP),
    ("GAMMA", "setGamma", GAMMA_SWEEP),
)

#: A contrast that puts ``contrast * gamma`` below OCIO's 0.001 floor. Not a
#: malformed request: the floor is a branch of the op, and where the reference
#: can be measured there it is.
FLOORED = -1.0

#: The pivoted styles, whose inverse is where the sweep above stops.
PIVOTED = (OCIO.EXPOSURE_CONTRAST_LINEAR, OCIO.EXPOSURE_CONTRAST_VIDEO)


def exposure_contrast(style, *, inverse=False, **overrides):
    """A bare ``ExposureContrastTransform`` at the defaults, minus overrides."""
    transform = OCIO.ExposureContrastTransform()
    transform.setStyle(style)
    for name, value in {**DEFAULTS, **overrides}.items():
        getattr(transform, f"set{name}")(value)
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def test_the_registry_carries_exposure_contrast():
    assert EXPOSURE_CONTRAST in emitters.REGISTRY


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_the_graph_declares_one_input_per_scalar_property(
    style, inverse, config, compile_bare
):
    """OCIO's three scalar dynamic property types, spelled as OCIO spells them
    (§spec:dynamic-properties), defaulting to the values it reports."""
    model = compile_bare(config.getProcessor(exposure_contrast(style, inverse=inverse)))
    assert parameters(model) == {
        "EXPOSURE": pytest.approx([DEFAULTS["Exposure"]]),
        "CONTRAST": pytest.approx([DEFAULTS["Contrast"]]),
        "GAMMA": pytest.approx([DEFAULTS["Gamma"]]),
    }


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_a_bare_exposure_contrast_verifies_at_its_defaults(
    style, inverse, check_transform
):
    result = check_transform(exposure_contrast(style, inverse=inverse))
    assert result.ok, str(result)
    assert result.compared > 0


@pytest.mark.parametrize(("name", "setter", "sweep"), SWEEPS)
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_one_graph_tracks_ocio_across_a_swept_parameter(
    name, setter, sweep, style, inverse, config, check, compile_bare
):
    """The acceptance test for a live parameter (§spec:dynamic-properties).

    The graph is compiled once, at the defaults, and every swept value is held
    against the processor OCIO builds for it — which is the claim a live
    parameter makes: no recompilation.
    """
    model = compile_bare(config.getProcessor(exposure_contrast(style, inverse=inverse)))

    for value in sweep:
        reference = exposure_contrast(style, inverse=inverse)
        getattr(reference, setter)(value)
        result = check(
            config.getProcessor(reference), model=model, parameters={name: [value]}
        )
        assert result.ok, f"{name}={value}: {result}"
        assert result.compared > 0


@pytest.mark.parametrize("style", STYLES)
def test_a_swept_parameter_moves_the_picture(style, config, compile_bare, row):
    """A knob wired to nothing agrees with OCIO at the default and is ignored
    everywhere else, which every agreement test above would pass."""
    model = compile_bare(config.getProcessor(exposure_contrast(style)))
    samples = row(0.18, 0.5)
    dark = run_graph(model, samples, {"EXPOSURE": [-3.0]})
    bright = run_graph(model, samples, {"EXPOSURE": [3.0]})
    assert (dark < bright).all()


def test_a_property_ocio_declares_dynamic_is_the_same_knob(config, compile_bare, check):
    """OCIO exposes a run-time handle for a property a transform declares
    dynamic. The graph input is that handle, and the graph does not need the
    declaration to offer it: an input left unbound is its default."""
    dynamic = exposure_contrast(OCIO.EXPOSURE_CONTRAST_LINEAR)
    dynamic.makeExposureDynamic()
    model = compile_bare(config.getProcessor(dynamic))
    assert list(parameters(model)) == ["EXPOSURE", "CONTRAST", "GAMMA"]

    swept = 2.5
    result = check(
        config.getProcessor(
            exposure_contrast(OCIO.EXPOSURE_CONTRAST_LINEAR, Exposure=swept)
        ),
        model=model,
        parameters={"EXPOSURE": [swept]},
    )
    assert result.ok, str(result)


def test_two_ops_carry_a_knob_each(config, compile_bare):
    """OCIO's run time allows one dynamic property of each type per processor
    and drops the rest with a warning. A graph has no such limit, so each op
    keeps its own default rather than one of them being discarded."""
    group = OCIO.GroupTransform()
    for exposure in (1.0, 2.0):
        group.appendTransform(
            exposure_contrast(OCIO.EXPOSURE_CONTRAST_LINEAR, Exposure=exposure)
        )
    defaults = parameters(compile_bare(config.getProcessor(group)))

    assert defaults["EXPOSURE"] == pytest.approx([1.0])
    assert defaults["EXPOSURE_2"] == pytest.approx([2.0])


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("inverse", (False, True))
def test_the_declared_breakpoint_is_where_the_pivot_clamp_engages(style, inverse):
    """``linear`` and ``video`` floor the pivoted value at zero; ``log`` is an
    affine over the whole line and declares none."""
    transform = exposure_contrast(style, inverse=inverse)
    expected = [] if style == OCIO.EXPOSURE_CONTRAST_LOGARITHMIC else [0.0]
    assert emitters.breakpoints(transform) == expected


def test_a_style_this_compiler_has_no_path_for_is_refused_by_name(
    config, compile_bare, monkeypatch
):
    """Following ``_negative_style``: an unrecognised style is refused rather
    than approximated by the nearest one implemented. Narrowing the supported
    set stands in for the style an OCIO release has not shipped yet."""
    monkeypatch.setattr(emitters, "EXPOSURE_CONTRAST_STYLES", ("VIDEO",))
    with pytest.raises(UnsupportedOpError, match="LINEAR"):
        compile_bare(
            config.getProcessor(exposure_contrast(OCIO.EXPOSURE_CONTRAST_LINEAR))
        )


@pytest.mark.parametrize("style", STYLES)
def test_the_contrast_floor_verifies_forward(style, config, check, compile_bare):
    """``contrast * gamma`` is held at 0.001 rather than let through. Forward
    the floor is an exponent of 0.001, which damps rather than amplifies, so
    the reference is measurable there and the graph agrees with it."""
    model = compile_bare(config.getProcessor(exposure_contrast(style)))
    result = check(
        config.getProcessor(exposure_contrast(style, Contrast=FLOORED)),
        model=model,
        parameters={"CONTRAST": [FLOORED]},
    )
    assert result.ok, str(result)


def test_the_log_inverse_floors_the_reciprocal_rather_than_the_product(
    config, check, compile_bare
):
    """Which side of the reciprocal the floor sits on is the reference's, and
    the two orders part company below the floor: flooring the product would put
    the log inverse's slope at 1000 where OCIO puts it at 0.001. Measurable,
    because that style carries no power to amplify anything."""
    style = OCIO.EXPOSURE_CONTRAST_LOGARITHMIC
    model = compile_bare(config.getProcessor(exposure_contrast(style, inverse=True)))
    result = check(
        config.getProcessor(exposure_contrast(style, inverse=True, Contrast=FLOORED)),
        model=model,
        parameters={"CONTRAST": [FLOORED]},
    )
    assert result.ok, str(result)
    assert result.compared > 0


def test_a_contrast_of_zero_leaves_the_log_inverse_with_nothing_finite(
    config, check, compile_bare
):
    """The reciprocal of zero is infinite, so the reference is non-finite
    across the whole lattice. Every sample agrees on its class and none is held
    against the tolerance, which is a graph certified on no evidence rather
    than a graph that disagrees (§spec:evidence-floor)."""
    style = OCIO.EXPOSURE_CONTRAST_LOGARITHMIC
    model = compile_bare(config.getProcessor(exposure_contrast(style, inverse=True)))
    result = check(
        config.getProcessor(exposure_contrast(style, inverse=True, Contrast=0.0)),
        model=model,
        parameters={"CONTRAST": [0.0]},
    )
    assert result.disagreed == 0
    assert result.compared == 0
    assert not result.ok


@pytest.mark.parametrize("style", PIVOTED)
def test_the_pivoted_inverse_gives_out_where_ocios_own_power_does(
    style, config, check, compile_bare
):
    """Where the sweep stops, and why it stops there rather than at the knob's
    end. OCIO's ExposureContrast renderer uses an approximate power — one
    ``OPTIMIZATION_FAST_LOG_EXP_POW`` does not govern, so §spec:verification's
    remedy of clearing a flag does not reach it. Its base error is around 1e-5
    relative, which the inverse's exponent of ``1 / (contrast * gamma)``
    multiplies: floored, that exponent is 1000 and the reference parts company
    with an accurate power by 9e-3.

    Pinned rather than worked around, because the boundary is the reference's
    and not the graph's. An OCIO release that made the renderer accurate would
    fail this test, which is how the sweep learns it can widen.
    """
    model = compile_bare(config.getProcessor(exposure_contrast(style, inverse=True)))
    result = check(
        config.getProcessor(exposure_contrast(style, inverse=True, Contrast=FLOORED)),
        model=model,
        parameters={"CONTRAST": [FLOORED]},
    )
    assert not result.ok
    assert result.max_rel > 1e-3


def test_a_contrast_of_one_leaves_the_negative_half_alone(config, compile_bare, row):
    """OCIO short circuits at ``contrast == 1``, and the short circuit is not
    cosmetic: the pivoted power floors its base at zero, so evaluating it at an
    exponent of one would clip every negative value to black."""
    transform = exposure_contrast(
        OCIO.EXPOSURE_CONTRAST_LINEAR, Exposure=0.0, Contrast=1.0, Gamma=1.0
    )
    model = compile_bare(config.getProcessor(transform))
    samples = row(-0.5, -0.25, 0.5)
    assert run_graph(model, samples) == pytest.approx(samples, abs=1e-6)
