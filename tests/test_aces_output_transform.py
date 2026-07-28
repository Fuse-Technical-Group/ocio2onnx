"""``ACES_OUTPUT_TRANSFORM_20`` emits the ACES 2.0 display rendering.

The largest op in the pinned config and the only one that is *the look*, so the
misreadings available here are both numerous and hard to see: a tone scale
coefficient derived from the wrong peak, a hue table interval off by one, a
``std::max`` that swallows a NaN in OCIO and propagates it here. None of them
would show up on a grey ramp. The tests below therefore lean on the oracle
across the whole lattice, and pin the few derived quantities that the oracle
alone would leave ambiguous.
"""

import re

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import aces2, emitters
from ocio2onnx.addressing import enumerate_transforms
from ocio2onnx.compiler import unsupported_ops
from ocio2onnx.emitters import op_label, supported_ops
from ocio2onnx.oracle import lattice, run_graph

#: How the registry, the census, and a refusal all spell this op.
ACES = "FixedFunction[ACES_OUTPUT_TRANSFORM_20]"

#: Measured across the pinned config: every transform carrying the op is a
#: display view, and there are 24 of them (§spec:op-coverage).
CARRYING = 24

#: The op's nine parameters for a 100 nit Rec.709 display, which is what the
#: SDR views resolve to.
REC709_SDR = (100.0, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329)

#: A second parameterisation, so nothing passes by holding one peak's
#: constants: 1000 nits onto P3-D65.
P3_HDR = (1000.0, 0.68, 0.32, 0.265, 0.69, 0.15, 0.06, 0.3127, 0.329)


def aces(params=REC709_SDR, *, inverse=False):
    """A bare ``ACES_OUTPUT_TRANSFORM_20`` at one parameterisation."""
    transform = OCIO.FixedFunctionTransform(
        OCIO.FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20, params=list(params)
    )
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def pixel(*channels):
    """One channels-first pixel, shaped as the graph's interface."""
    return np.array([[[[value]] for value in channels]], dtype=np.float32)


@pytest.fixture(scope="module")
def carrying_the_op(config):
    """Every transform in the pinned config that carries the op, by endpoint."""
    return {
        label: processor
        for label, processor in enumerate_transforms(config)
        if ACES in {op_label(op) for op in processor.createGroupTransform()}
    }


@pytest.fixture(scope="module")
def resolved():
    """The op's parameters, resolved once for the SDR parameterisation."""
    return aces2.output_transform(aces())


def test_the_registry_keys_on_the_style_rather_than_the_class():
    """Both ``FixedFunction`` styles are emitted now, and still under separate
    keys: they remain unrelated transforms sharing a class."""
    assert ACES in emitters.REGISTRY
    assert ACES in supported_ops()
    assert "FixedFunction" not in supported_ops()


def test_the_config_carries_the_op_where_the_specification_counted_it(carrying_the_op):
    assert len(carrying_the_op) == CARRYING
    assert all(label.startswith("view ") for label in carrying_the_op)


def test_every_config_transform_the_op_unblocks_verifies(carrying_the_op, check):
    """The workstream's acceptance test: the 24 display views that refused on
    this op alone now compile and agree with OCIO's CPU processor.

    Every sample is compared, because a display rendering compresses to a
    display range and the reference is finite across the whole lattice — a
    verified result over a suspiciously small sample count would mean the
    graph, not the reference, had gone non-finite.
    """
    failures = []
    for label, processor in carrying_the_op.items():
        assert unsupported_ops(processor) == []
        result = check(processor, label)
        if not result.ok:
            failures.append(f"{label}: {result}")
        elif result.nonfinite:
            failures.append(f"{label}: {result.nonfinite} samples went uncompared")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("params", (REC709_SDR, P3_HDR))
def test_a_bare_output_transform_verifies(params, check_transform):
    """The op alone, away from the matrices and the encoding that surround it
    in a view, so a failure here is the op rather than its neighbours."""
    result = check_transform(aces(params))
    assert result.ok, str(result)
    assert result.compared > 0


def test_an_inverse_output_transform_is_refused_by_name(config, compile_bare):
    """The inverse is not this path run backwards — its gamut compression
    solves an approximation twice — so it is refused rather than approximated
    (§spec:op-coverage)."""
    with pytest.raises(emitters.UnsupportedOpError, match=r"inverse.*ACES_OUTPUT"):
        compile_bare(config.getProcessor(aces(inverse=True)))


def test_the_graph_is_finite_across_the_lattice(config, compile_bare):
    """A display rendering compresses to a display range, so nothing in the
    lattice has anywhere to overflow to. Every ``Pow`` in the chain would
    answer NaN on a negative base, and every clamp that prevents that is read
    off OCIO rather than added defensively."""
    processor = config.getProcessor(aces())
    got = run_graph(compile_bare(processor), lattice(processor))
    assert np.isfinite(got).all()


def test_black_maps_to_black(config, compile_bare):
    """A non-positive achromatic response has no hue to report, and OCIO
    answers black rather than carrying a hue of zero through the rendering."""
    graph = compile_bare(config.getProcessor(aces()))
    assert run_graph(graph, pixel(0.0, 0.0, 0.0)).ravel() == pytest.approx([0.0] * 3)


def test_a_negative_pixel_maps_to_black_rather_than_to_a_nan(config, compile_bare):
    """OCIO's tone scale raises a negative luminance to a fractional power and
    every step after it is NaN; its own ``std::max(0, ...)`` turns that back
    into black. ONNX ``Max`` would propagate the NaN instead, which is why
    ``_std_max`` emits the comparison (§spec:op-emission)."""
    graph = compile_bare(config.getProcessor(aces()))
    got = run_graph(graph, pixel(-1.0, -1.0, -1.0)).ravel()
    assert np.isfinite(got).all()
    assert got == pytest.approx([0.0] * 3)


def test_a_nan_pixel_leaves_as_nan_without_indexing_a_table(config, compile_bare):
    """A NaN takes the achromatic arm, which OCIO's ``<= 0`` test does not, so
    the hue it carries into the table lookup is zero rather than NaN — a NaN
    ``Gather`` index is undefined. The pixel is unaffected: the tone scale
    reads the achromatic response, so the NaN survives the substitution and
    the graph agrees with OCIO on the kind of value it returns."""
    graph = compile_bare(config.getProcessor(aces()))
    got = run_graph(graph, pixel(np.nan, np.nan, np.nan)).ravel()
    assert np.isnan(got).all()


def test_the_declared_breakpoints_are_black_and_the_peak():
    """Stated on the diagonal, because the op branches in lightness rather
    than in the channels."""
    assert emitters.breakpoints(aces()) == pytest.approx([0.0, 1.0])
    assert emitters.breakpoints(aces(P3_HDR)) == pytest.approx([0.0, 10.0])


def test_a_neutral_pixel_at_the_peak_reaches_the_top_of_the_tone_scale(resolved):
    """What makes ``peak / 100`` a breakpoint: 1.0 is 100 nits, so a neutral
    pixel there carries exactly the luminance the tone scale tops out at."""
    reached = aces2.luminance_to_lightness(resolved.peak_luminance)
    assert float(reached) == pytest.approx(float(resolved.limit_lightness))


def test_the_derived_parameters_are_the_ones_ocio_derives(resolved):
    """Spot checks against OCIO's own generated shader for this
    parameterisation, which inlines the constants its renderer was built with.

    The oracle would catch a wrong constant, but not say which; these name the
    three the rest of the derivation hangs off.
    """
    assert float(aces2.CZ) == pytest.approx(1.1370559, rel=1e-6)
    assert float(aces2.A_W_J) == pytest.approx(0.0323680267, rel=1e-6)
    assert float(resolved.limit_lightness) == pytest.approx(100.0, rel=1e-6)
    assert float(resolved.mid_lightness) == pytest.approx(34.096539, rel=1e-6)
    assert float(resolved.lower_hull_gamma_inv) == pytest.approx(1.0 / 1.14, rel=1e-6)


@pytest.mark.parametrize("params", (REC709_SDR, P3_HDR))
def test_the_derived_hue_table_is_the_one_ocio_built(params):
    """The two big tables are read off OCIO's shader description; the hue table
    is derived here because OCIO publishes it only as generated shader *text*
    (§spec:op-emission). This holds the derivation against that text.

    The bound is not equality. One corner hue lands a few ulps from OCIO's,
    which moves a knot by under 1e-4 of a degree — three orders of magnitude
    below the spacing, and far below what the oracle can see.
    """
    transform = aces(params)
    processor = OCIO.Config.CreateRaw().getProcessor(transform)
    description = OCIO.GpuShaderDesc.CreateShaderDesc(
        language=OCIO.GPU_LANGUAGE_GLSL_4_0
    )
    processor.getDefaultGPUProcessor().extractGpuShaderInfo(description)
    declared = re.search(
        r"hues_array\[(\d+)\] = float\[\d+\]\(([^)]*)\)", description.getShaderText()
    )
    assert declared, "OCIO no longer declares the hue table in its shader text"
    assert int(declared.group(1)) == aces2.TABLE_SIZE

    want = np.array([float(v) for v in declared.group(2).split(",")], dtype=np.float32)
    got = aces2.output_transform(transform).hues
    assert np.abs(got - want).max() < 1e-4


@pytest.mark.parametrize("params", (REC709_SDR, P3_HDR))
def test_the_hue_table_rises_across_every_entry(params):
    """The interval search bisects on that ordering, and the wrap entries
    either end are what keep it true across 0 and 360."""
    hues = aces2.output_transform(aces(params)).hues
    assert (np.diff(hues) > 0.0).all()
    assert hues[aces2.FIRST_NOMINAL] == 0.0

    # The three wrap entries are the nominal ones displaced a full turn, which
    # is what keeps the table increasing across 0 and 360 rather than folding.
    assert hues[0] == hues[aces2.LAST_NOMINAL] - aces2.HUE_LIMIT
    assert hues[aces2.UPPER_WRAP] == hues[aces2.FIRST_NOMINAL] + aces2.HUE_LIMIT
    assert hues[aces2.UPPER_WRAP + 1] == hues[aces2.FIRST_NOMINAL + 1] + aces2.HUE_LIMIT


@pytest.mark.parametrize("params", (REC709_SDR, P3_HDR))
def test_the_tables_are_the_size_ocio_publishes(params):
    resolved = aces2.output_transform(aces(params))
    assert resolved.reach.shape == (aces2.TABLE_SIZE,)
    assert resolved.cusps.shape == (aces2.TABLE_SIZE, 3)
    assert np.isfinite(resolved.cusps).all()
    assert (resolved.reach > 0.0).all()


def test_the_hue_search_window_brackets_the_table(resolved):
    """The window is measured off the table rather than assumed, and it is
    what makes the interval lookup a couple of steps instead of nine."""
    low, high = resolved.search_range
    assert low <= 0 < high
    assert high - low <= 4
