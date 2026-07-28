"""``REC2100_SURROUND`` emits a surround adjustment (§spec:op-emission).

The op scales all three channels by a power of their *luminance*, so the
misreading available here is a per-channel gamma: it agrees wherever the three
channels coincide, which is most of a naive test, and disagrees everywhere
else. The tests below hold a coloured pixel rather than a grey one for that
reason.

It is also the first op whose emitter is chosen by a style rather than by a
class. ``FixedFunction`` carries two unrelated transforms (§spec:op-coverage);
this workstream emits one of them and the other has to go on refusing, so the
registry keys on the style.
"""

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import emitters
from ocio2onnx.addressing import enumerate_transforms
from ocio2onnx.compiler import unsupported_ops
from ocio2onnx.emitters import op_label, supported_ops
from ocio2onnx.oracle import lattice, run_graph

#: The two styles that share the ``FixedFunction`` class, as a refusal and the
#: registry now spell them.
SURROUND = "FixedFunction[REC2100_SURROUND]"
ACES = "FixedFunction[ACES_OUTPUT_TRANSFORM_20]"

#: The display whose transforms this workstream unblocks (§spec:op-coverage).
HLG_DISPLAY = "Rec.2100-HLG - Display"

#: Four of the five config transforms carrying the op, which compile once it is
#: emitted. Both directions are here: the display's encode and its decode.
UNBLOCKED = (
    f"{HLG_DISPLAY} -> ref",
    f"ref -> {HLG_DISPLAY}",
    f"view {HLG_DISPLAY}/Un-tone-mapped",
    f"view {HLG_DISPLAY}/Video (colorimetric)",
)

#: The fifth, which goes on refusing because it carries the other style too.
STILL_REFUSED = f"view {HLG_DISPLAY}/ACES 2.0 - HDR 1000 nits (P3 D65)"

#: Rec.2020's luminance weights, which the op reads Y through. They sum to one,
#: so a grey pixel is raised to the gamma and nothing else.
LUMA = (0.2627, 0.6780, 0.0593)

#: The luminance OCIO's own renderer floors the forward op at.
MIN_LUM = 1e-4

#: A gamma either side of one, so no test passes on the identity. The pinned
#: config carries 1/1.2.
GAMMA = 1.2

#: A pixel whose three channels differ, so a per-channel gamma cannot agree
#: with a luminance scale.
COLOURED = (1.0, 0.5, 0.25)


def surround(gamma=GAMMA, *, inverse=False):
    """A bare ``REC2100_SURROUND`` at one gamma."""
    transform = OCIO.FixedFunctionTransform(
        OCIO.FIXED_FUNCTION_REC2100_SURROUND, params=[gamma]
    )
    if inverse:
        transform.setDirection(OCIO.TRANSFORM_DIR_INVERSE)
    return transform


def pixel(*channels):
    """One channels-first pixel, shaped as the graph's interface."""
    return np.array([[[[value]] for value in channels]], dtype=np.float32)


def luminance(*channels):
    """What the op reads Y as, before the fold and the floor."""
    return sum(weight * value for weight, value in zip(LUMA, channels, strict=True))


@pytest.fixture
def apply(config, compile_bare):
    """Run one bare transform's emitted graph over one pixel."""

    def apply(transform, *channels):
        return run_graph(
            compile_bare(config.getProcessor(transform)), pixel(*channels)
        ).ravel()

    return apply


@pytest.fixture(scope="module")
def carrying_the_surround(config):
    """Every transform in the pinned config that carries the op, by endpoint."""
    return {
        label: processor
        for label, processor in enumerate_transforms(config)
        if SURROUND in {op_label(op) for op in processor.createGroupTransform()}
    }


def test_the_registry_keys_on_the_style_rather_than_the_class():
    """The class alone cannot be registered: it would claim the other style
    too, and a refusal is what §road:aces-output-transform is deferred behind."""
    assert SURROUND in emitters.REGISTRY
    assert "FixedFunctionTransform" not in emitters.REGISTRY
    assert SURROUND in supported_ops()
    assert ACES not in supported_ops()


def test_the_config_carries_the_op_where_the_specification_counted_it(
    carrying_the_surround,
):
    assert set(carrying_the_surround) == {*UNBLOCKED, STILL_REFUSED}


def test_the_transform_carrying_both_styles_refuses_on_the_one_left(
    carrying_the_surround,
):
    """The op is gone from the refusal and the transform is not: one style
    emitted does not make the other emittable."""
    assert unsupported_ops(carrying_the_surround[STILL_REFUSED]) == [ACES]


def test_every_config_transform_the_op_unblocks_verifies(carrying_the_surround, check):
    """The workstream's acceptance test: the four HLG transforms that refused
    on this op alone now compile and agree with the CPU processor."""
    failures = []
    for label in UNBLOCKED:
        processor = carrying_the_surround[label]
        assert unsupported_ops(processor) == []
        result = check(processor, label)
        if not result.ok:
            failures.append(f"{label}: {result}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("inverse", (False, True))
def test_a_bare_surround_verifies(inverse, check_transform):
    result = check_transform(surround(inverse=inverse))
    assert result.ok, str(result)
    assert result.compared > 0


def test_the_scale_is_one_luminance_rather_than_three_gammas(apply):
    """A per-channel gamma agrees on grey and disagrees on everything else, so
    the check is that the three channels take the *same* factor — and that the
    factor is Rec.2020's luminance raised to ``gamma - 1``."""
    got = apply(surround(), *COLOURED)
    scale = luminance(*COLOURED) ** (GAMMA - 1.0)
    assert got == pytest.approx([value * scale for value in COLOURED], rel=1e-5)

    ratios = [out / value for out, value in zip(got, COLOURED, strict=True)]
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-5)
    assert ratios[1] == pytest.approx(ratios[2], rel=1e-5)


def test_a_grey_pixel_is_raised_to_the_gamma(apply):
    """The weights sum to one, so on the diagonal the op is ``v ** gamma``."""
    assert apply(surround(), 0.5, 0.5, 0.5) == pytest.approx([0.5**GAMMA] * 3, rel=1e-5)


def test_luminance_below_the_floor_is_held_at_it(apply):
    """Unfloored, the scale runs away as the pixel approaches black."""
    channels = (1e-9, 2e-9, 3e-9)
    assert luminance(*channels) < MIN_LUM
    scale = MIN_LUM ** (GAMMA - 1.0)
    assert apply(surround(), *channels) == pytest.approx(
        [value * scale for value in channels], rel=1e-5
    )


def test_negative_luminance_folds_through_its_magnitude(apply):
    """OCIO takes ``abs(Y)`` before the floor, so a negative pixel keeps the
    scale its magnitude earns rather than collapsing onto the floor."""
    channels = tuple(-value for value in COLOURED)
    scale = abs(luminance(*channels)) ** (GAMMA - 1.0)
    assert apply(surround(), *channels) == pytest.approx(
        [value * scale for value in channels], rel=1e-5
    )
    assert scale != pytest.approx(MIN_LUM ** (GAMMA - 1.0), rel=1e-3)


def test_the_inverse_reciprocates_the_gamma(apply):
    """``1/gamma``, not ``-gamma`` and not ``2 - gamma``."""
    assert apply(surround(inverse=True), 0.5, 0.5, 0.5) == pytest.approx(
        [0.5 ** (1.0 / GAMMA)] * 3, rel=1e-5
    )


def test_the_inverse_floors_at_the_forward_floors_image(apply):
    """The inverse's clamp lives in the forward's output domain, so its floor
    is ``1e-4 ** gamma`` rather than ``1e-4``."""
    floor = MIN_LUM**GAMMA
    channels = (floor / 10.0,) * 3
    scale = floor ** (1.0 / GAMMA - 1.0)
    assert apply(surround(inverse=True), *channels) == pytest.approx(
        [value * scale for value in channels], rel=1e-5
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_the_declared_breakpoints_are_zero_and_the_floor(inverse):
    """Where the fold and the clamp switch, stated on the diagonal: the three
    weights sum to one, so a grey pixel's luminance is its own value."""
    floor = MIN_LUM**GAMMA if inverse else MIN_LUM
    assert emitters.breakpoints(surround(inverse=inverse)) == pytest.approx(
        [0.0, floor, -floor]
    )


@pytest.mark.parametrize("inverse", (False, True))
def test_the_graph_is_finite_across_the_lattice(inverse, config, compile_bare):
    """``Pow`` on a non-positive base is NaN in ONNX, and the lattice runs to
    -65504. The floor is what keeps the base positive everywhere."""
    processor = config.getProcessor(surround(inverse=inverse))
    got = run_graph(compile_bare(processor), lattice(processor))
    assert np.isfinite(got).all()
