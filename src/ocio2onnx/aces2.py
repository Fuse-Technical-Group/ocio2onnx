"""Resolve one ACES 2.0 output transform's parameters (§spec:op-emission).

`ACES_OUTPUT_TRANSFORM_20` carries nine numbers — a peak luminance and the
limiting gamut's chromaticities — and OCIO's renderer expands them into a CAM16
colour appearance model, a tone scale, a chroma compression, and a gamut
compression, each with its own derived constants and three hue-indexed tables.
This module performs that expansion once, at compile time, so `emitters` has
only arithmetic left to emit.

Every step here is read off OCIO 2.5's own renderer — `ACES2/Transform.cpp` and
`FixedFunctionOpCPU.cpp` — rather than off the published ACES 2.0 description,
which differs from it in details that reach the picture. Nothing here is a
choice; this module defines no transform (§spec:non-goals).

**The arithmetic runs at float32 because OCIO's does.** The derivation is not a
smooth function of its inputs: OCIO builds the hue table by binary search, and
a comparison decided the other way moves a knot by a whole sample rather than
by an ulp. float64 would be more accurate and less faithful, and the oracle
measures faithfulness.

**Two tables are read off OCIO rather than rebuilt.** The gamut cusp table
costs a bisection per hue against the limiting gamut hull, and its third column
a second, nested search for the upper hull's exponent; the reach table costs a
third. OCIO computes all of them to build its own renderer and hands them over
through its GPU shader description, which is a structured accessor rather than
generated source. Rebuilding them would be slower, and would put a
search tolerance between this compiler and its oracle for no gain — the same
reasoning that makes OCIO the oracle at all (§spec:verification).

The hue table is derived here because OCIO exposes it only inside generated
shader *text*, which is not an interface. It costs six bisections rather than
seven hundred, and `tests/test_aces_output_transform.py` holds the result
against the array OCIO's own shader declares.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import PyOpenColorIO as OCIO

#: Every derived parameter is float32, as OCIO's renderer computes it.
F32 = np.float32

#: Chromaticities as OCIO names them: red, green, blue, white, each ``(x, y)``.
Primaries = tuple[tuple[float, float], ...]

#: The colour space the transform reads. Fixed: OCIO's renderer builds the
#: input side from ACES AP0 whatever the op's parameters say.
AP0: Primaries = ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.077), (0.32168, 0.33767))

#: The gamut the chroma compression reaches towards, also fixed.
AP1: Primaries = ((0.713, 0.293), (0.165, 0.83), (0.128, 0.044), (0.32168, 0.33767))

#: The CAM16 cone response primaries. Not a display gamut — two of its
#: chromaticities lie outside the spectral locus, which is what makes the
#: matrix a cone response rather than a colour space.
CAM16: Primaries = (
    (0.8336, 0.1735),
    (2.3854, -1.4659),
    (0.087, -0.125),
    (0.333, 0.333),
)

#: What a value of 1.0 means in nits, and the appearance model's viewing
#: conditions: adapting field luminance, background luminance, and a dim
#: surround's three coefficients.
REFERENCE_LUMINANCE = F32(100.0)
ADAPTING_LUMINANCE = F32(100.0)
BACKGROUND_LUMINANCE = F32(20.0)
SURROUND = (F32(0.9), F32(0.59), F32(0.9))

#: Lightness is reported on [0, 100], and the model is run backwards as often
#: as forwards, so the reciprocal is named alongside it.
J_SCALE = F32(100.0)
INV_J_SCALE = F32(1.0) / J_SCALE

#: The post-adaptation cone response compression, ``x**0.42 / (27.13 +
#: x**0.42)``, and its inverse. The inverse holds its argument below 0.99
#: because the compression's own range is [0, 1) and its inverse diverges at
#: the top; OCIO clamps there rather than returning an infinity.
CONE_EXPONENT = F32(0.42)
INV_CONE_EXPONENT = F32(1.0) / CONE_EXPONENT
CONE_OFFSET = F32(0.2713) * REFERENCE_LUMINANCE
CONE_LIMIT = F32(0.99)

#: What the achromatic and chromatic rows of the cone response are scaled by.
CONE_SCALE = F32(4.0) * REFERENCE_LUMINANCE
CHROMATIC_SCALE = F32(43.0) * SURROUND[2]

#: Hue is carried in degrees over [0, 360), and the tables hold one entry per
#: degree plus one below and two above, so a lookup near either end reads a
#: neighbour rather than a bound.
HUE_LIMIT = F32(360.0)
NOMINAL_HUES = 360
TABLE_SIZE = 363

#: Where the nominal entries sit in a table of ``TABLE_SIZE``.
FIRST_NOMINAL = 1
LAST_NOMINAL = FIRST_NOMINAL + NOMINAL_HUES - 1
UPPER_WRAP = FIRST_NOMINAL + NOMINAL_HUES

#: The six corners of the limiting gamut cube the hue table is anchored to —
#: red, yellow, green, cyan, blue, magenta, in that order so hues rotate
#: forwards — and the two wrap entries either side of them.
CUSP_CORNERS = 6

#: How far the reach corner search runs and how tightly it closes. Both are
#: OCIO's.
REACH_TOLERANCE = F32(1e-3)

#: The chroma compression's normalising curve: three harmonics of hue plus a
#: constant, weighted. Read off OCIO's ``chroma_compress_norm``.
CHROMA_NORM_COS = (F32(11.34072), F32(16.46899), F32(7.88380))
CHROMA_NORM_SIN = (F32(14.66441), F32(-6.37224), F32(9.19364))
CHROMA_NORM_OFFSET = F32(77.12896)

#: The chroma compression's shape, before the peak luminance moves it.
CHROMA_COMPRESS = F32(2.4)
CHROMA_COMPRESS_FACT = F32(3.3)
CHROMA_EXPAND = F32(1.3)
CHROMA_EXPAND_FACT = F32(0.69)
CHROMA_EXPAND_THRESHOLD = F32(0.5)
CHROMA_SATURATION_FLOOR = F32(0.2)
CHROMA_SCALE_GAIN = F32(0.03379)
CHROMA_SCALE_EXPONENT = F32(0.30596)
CHROMA_SCALE_OFFSET = F32(0.45135)

#: The gamut compression's shape, likewise.
SMOOTH_CUSPS = F32(0.12)
CUSP_MID_BLEND = F32(1.3)
FOCUS_GAIN_BLEND = F32(0.3)
FOCUS_DISTANCE = F32(1.35)
FOCUS_DISTANCE_SCALING = F32(1.75)
COMPRESSION_THRESHOLD = F32(0.75)
LOWER_HULL_GAMMA = F32(1.14)
LOWER_HULL_GAMMA_SCALING = F32(0.07)
FOCUS_GAIN_FLOOR = F32(0.0001)

#: The toe's floor on its own softness parameter.
TOE_MIN_K2 = F32(0.001)

#: The tone scale's fixed shape: normalised white in nits, contrast, the 18%
#: grey anchor and the luminance it lands on, how that grey moves with peak
#: luminance, the shadow toe, and the scene-referred values that hit the roof.
TONE_WHITE = F32(100.0)
TONE_CONTRAST = F32(1.15)
TONE_GREY = F32(0.18)
TONE_GREY_OUT = F32(10.013)
TONE_GREY_SHIFT = F32(0.14)
TONE_TOE = F32(0.04)
TONE_HIT_MIN = F32(128.0)
TONE_HIT_MAX = F32(896.0)

#: How far past the roof the reach corner search looks.
TONE_FORWARD_LIMIT_SCALE = F32(8.0)


def _rgb_to_xyz(primaries: Primaries) -> np.ndarray:
    """RGB to CIE XYZ for one set of chromaticities, at float64.

    OCIO's ``rgb2xyz_from_xy`` computes this in double and casts once, so the
    cast happens at the caller rather than here.
    """
    red, green, blue, white = primaries
    matrix = np.array(
        [
            [red[0], green[0], blue[0]],
            [red[1], green[1], blue[1]],
            [1.0 - sum(red), 1.0 - sum(green), 1.0 - sum(blue)],
        ],
        dtype=np.float64,
    )
    white_xyz = np.array(
        [white[0] / white[1], 1.0, (1.0 - white[0] - white[1]) / white[1]],
        dtype=np.float64,
    )
    return matrix * (np.linalg.inv(matrix) @ white_xyz)


def _invert(matrix: np.ndarray) -> np.ndarray:
    """Invert at float64 and return float32, as OCIO's ``invert_f33`` does."""
    return np.linalg.inv(matrix.astype(np.float64)).astype(np.float32)


def _cone_compress(value: F32) -> F32:
    """One channel's post-adaptation cone response, unsigned."""
    compressed = np.power(value, CONE_EXPONENT, dtype=np.float32)
    return F32(compressed / (CONE_OFFSET + compressed))


#: The achromatic and two chromatic rows of the cone response, before scaling.
_BASE_CONE_TO_AAB = np.array(
    [
        [2.0, 1.0, 1.0 / 20.0],
        [1.0, -12.0 / 11.0, 1.0 / 11.0],
        [1.0 / 9.0, 1.0 / 9.0, -2.0 / 9.0],
    ],
    dtype=np.float32,
)


def _model_gamma() -> F32:
    """The appearance model's lightness exponent, ``c * z``.

    It depends on the surround and the background alone, so it is the same for
    every parameterisation of this op — which is why the emitter carries it as
    a constant rather than as a field.
    """
    return SURROUND[1] * (
        F32(1.48) + F32(math.sqrt(BACKGROUND_LUMINANCE / REFERENCE_LUMINANCE))
    )


def _adapting_response() -> F32:
    """``F_L``, the luminance level adaptation factor."""
    k = F32(1.0) / (F32(5.0) * ADAPTING_LUMINANCE + F32(1.0))
    k4 = k * k * k * k
    return F32(0.2) * k4 * (F32(5.0) * ADAPTING_LUMINANCE) + F32(0.1) * np.power(
        F32(1.0) - k4, F32(2.0), dtype=np.float32
    ) * np.power(F32(5.0) * ADAPTING_LUMINANCE, F32(1.0 / 3.0), dtype=np.float32)


#: The lightness exponent and its reciprocal.
CZ = _model_gamma()
INV_CZ = F32(1.0) / CZ

#: The adaptation factor, normalised, and the achromatic response of the
#: adapting white. Both are independent of the op's parameters for the same
#: reason ``CZ`` is: nothing in them reads a chromaticity or a peak luminance.
F_L_N = F32(_adapting_response() / REFERENCE_LUMINANCE)
A_W_J = _cone_compress(_adapting_response())
INV_A_W_J = F32(1.0) / A_W_J


@dataclasses.dataclass(frozen=True)
class Appearance:
    """One gamut's half of the appearance model.

    Four matrices: into the chromatically adapted cone responses and back, and
    from those responses into ``(A, a, b)`` and back. The scalars that would
    otherwise live here — the lightness exponent, the adaptation factor — do
    not depend on the gamut, so they are module constants.
    """

    rgb_to_cone: np.ndarray
    cone_to_rgb: np.ndarray
    cone_to_aab: np.ndarray
    aab_to_cone: np.ndarray


def appearance(primaries: Primaries) -> Appearance:
    """OCIO's ``init_JMhParams`` for one gamut.

    The chromatic adaptation is folded into the cone matrix rather than applied
    as a second step, and the achromatic row is divided by the adapting white's
    own response so that white lands at ``A = 1``.
    """
    xyz_to_cam16 = _invert(_rgb_to_xyz(CAM16).astype(np.float32))
    rgb_to_xyz = _rgb_to_xyz(primaries).astype(np.float32)
    white_xyz = rgb_to_xyz @ np.full(3, REFERENCE_LUMINANCE, dtype=np.float32)
    white_cone = xyz_to_cam16 @ white_xyz
    to_cam16 = (xyz_to_cam16 @ rgb_to_xyz) * REFERENCE_LUMINANCE

    adaptation = np.array(
        [F_L_N * white_xyz[1] / channel for channel in white_cone], dtype=np.float32
    )
    adapted_white = np.array(
        [_cone_compress(value) for value in adaptation * white_cone], dtype=np.float32
    )

    cone_to_aab = CONE_SCALE * _BASE_CONE_TO_AAB
    achromatic_white = F32(
        cone_to_aab[0, 0] * adapted_white[0]
        + cone_to_aab[0, 1] * adapted_white[1]
        + cone_to_aab[0, 2] * adapted_white[2]
    )
    cone_to_aab = np.stack(
        [
            cone_to_aab[0] / achromatic_white,
            cone_to_aab[1] * CHROMATIC_SCALE,
            cone_to_aab[2] * CHROMATIC_SCALE,
        ]
    ).astype(np.float32)

    rgb_to_cone = (adaptation[:, None] * to_cam16).astype(np.float32)

    return Appearance(
        rgb_to_cone=rgb_to_cone,
        cone_to_rgb=_invert(rgb_to_cone),
        cone_to_aab=cone_to_aab,
        aab_to_cone=_invert(cone_to_aab),
    )


@dataclasses.dataclass(frozen=True)
class ToneScale:
    """The tone scale curve's four coefficients and its output scale.

    ``m_2`` and ``s_2`` place the curve so that 18% grey lands on the display
    luminance the peak asks for; ``g`` is the contrast and ``t_1`` the shadow
    toe. ``forward_limit`` and ``log_peak`` are not curve coefficients — they
    are how far the reach search looks and how far this peak sits from 100
    nits, and the compressions below read both.
    """

    g: F32
    t_1: F32
    c_t: F32
    s_2: F32
    m_2: F32
    n_r: F32
    forward_limit: F32
    log_peak: F32


def tone_scale(peak_luminance: F32) -> ToneScale:
    """OCIO's ``init_ToneScaleParams``."""
    n = F32(peak_luminance)
    n_r = TONE_WHITE
    g = TONE_CONTRAST
    t_1 = TONE_TOE

    r_hit = TONE_HIT_MIN + (TONE_HIT_MAX - TONE_HIT_MIN) * F32(
        math.log(n / n_r) / math.log(10000.0 / 100.0)
    )
    m_0 = F32(n / n_r)
    m_1 = F32(0.5) * (m_0 + F32(math.sqrt(m_0 * (m_0 + F32(4.0) * t_1))))
    m = m_1 / np.power((r_hit / m_1) / ((r_hit / m_1) + F32(1.0)), g, dtype=np.float32)

    w_i = F32(math.log(n / 100.0) / math.log(2.0))
    c_t = F32(TONE_GREY_OUT / n_r * (F32(1.0) + w_i * TONE_GREY_SHIFT))
    grey_in = F32(0.5) * (c_t + F32(math.sqrt(c_t * (c_t + F32(4.0) * t_1))))
    scaled = np.power(grey_in / m, F32(1.0) / g, dtype=np.float32)
    w_2 = F32(TONE_GREY / (-(m_1 * scaled) / (scaled - F32(1.0))))

    u_2 = np.power((r_hit / m_1) / ((r_hit / m_1) + w_2), g, dtype=np.float32)
    return ToneScale(
        g=g,
        t_1=t_1,
        c_t=c_t,
        s_2=F32(w_2 * m_1 * REFERENCE_LUMINANCE),
        m_2=F32(m_1 / u_2),
        n_r=n_r,
        forward_limit=TONE_FORWARD_LIMIT_SCALE * r_hit,
        log_peak=F32(math.log10(n / n_r)),
    )


def luminance_to_lightness(luminance: F32) -> F32:
    """``Y_to_J`` for a non-negative luminance, in nits over 100."""
    return J_SCALE * np.power(
        _cone_compress(F32(luminance) * F_L_N) * INV_A_W_J, CZ, dtype=np.float32
    )


def _rgb_to_aab(rgb: np.ndarray, model: Appearance) -> np.ndarray:
    """One colour's achromatic and two chromatic responses, at float32.

    Only the table construction below calls this; the emitted graph does the
    same arithmetic in ONNX.
    """
    cone = _apply(model.rgb_to_cone, rgb)
    compressed = np.array(
        [np.copysign(_cone_compress(abs(value)), value) for value in cone],
        dtype=np.float32,
    )
    return _apply(model.cone_to_aab, compressed)


def _rgb_to_jmh(rgb: np.ndarray, model: Appearance) -> np.ndarray:
    """One colour's lightness, colourfulness, and hue."""
    aab = _rgb_to_aab(rgb, model)
    if aab[0] <= 0.0:
        return np.zeros(3, dtype=np.float32)
    hue = F32(180.0) * np.arctan2(aab[2], aab[1], dtype=np.float32) / F32(math.pi)
    return np.array(
        [
            J_SCALE * np.power(aab[0], CZ, dtype=np.float32),
            np.sqrt(aab[1] * aab[1] + aab[2] * aab[2], dtype=np.float32),
            hue + HUE_LIMIT if hue < 0.0 else hue,
        ],
        dtype=np.float32,
    )


def _apply(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """``mult_f3_f33``: three float32 products summed left to right.

    Written out rather than left to ``@`` so the summation order is OCIO's.
    """
    return np.array(
        [
            matrix[row, 0] * vector[0]
            + matrix[row, 1] * vector[1]
            + matrix[row, 2] * vector[2]
            for row in range(3)
        ],
        dtype=np.float32,
    )


def _cube_corner(index: int) -> np.ndarray:
    """One corner of the unit cube, in the order R, Y, G, C, B, M.

    That order is what makes the six hues rotate forwards, which the merge
    below relies on.
    """
    return np.array(
        [float(((index + shift) % CUSP_CORNERS) < 3) for shift in (1, 5, 3)],
        dtype=np.float32,
    )


def _cycle(corners: list[np.ndarray]) -> list[np.ndarray]:
    """Rotate six corners so the lowest hue sits at index 1, then wrap.

    The entries either end are copies displaced by a full turn, so a hue near
    0 or 360 finds an interval rather than an edge, and the hues stay
    increasing across the whole table.
    """
    lowest = min(range(CUSP_CORNERS), key=lambda i: corners[i][2])
    rotated = [corners[(i + lowest) % CUSP_CORNERS].copy() for i in range(CUSP_CORNERS)]
    below, above = rotated[-1].copy(), rotated[0].copy()
    below[2] -= HUE_LIMIT
    above[2] += HUE_LIMIT
    return [below, *rotated, above]


def _limiting_corners(model: Appearance, peak_luminance: F32) -> list[np.ndarray]:
    """The limiting gamut's six cube corners, in JMh."""
    scale = F32(peak_luminance) / REFERENCE_LUMINANCE
    return _cycle(
        [
            _rgb_to_jmh((scale * _cube_corner(i)).astype(np.float32), model)
            for i in range(CUSP_CORNERS)
        ]
    )


def _reach_corners(
    model: Appearance, limit_lightness: F32, maximum: F32
) -> list[np.ndarray]:
    """The reach gamut's six corners at the top of the lightness range.

    The reach gamut is unbounded, so a corner is where its edge crosses
    ``limit_lightness`` rather than a vertex. The search runs on the achromatic
    response instead of on lightness because the two are monotonically related
    and the response costs one power fewer.
    """
    limit = np.power(limit_lightness * INV_J_SCALE, INV_CZ, dtype=np.float32)
    found = []
    for index in range(CUSP_CORNERS):
        direction = _cube_corner(index)
        low, high = F32(0.0), F32(maximum)
        while (high - low) > REACH_TOLERANCE:
            middle = F32((low + high) / F32(2.0))
            achromatic = _rgb_to_aab((middle * direction).astype(np.float32), model)[0]
            if achromatic < limit:
                low = middle
            else:
                high = middle
            if achromatic == limit:
                break
        found.append(_rgb_to_jmh((high * direction).astype(np.float32), model))
    return _cycle(found)


def _merge_hues(reach: list[np.ndarray], limiting: list[np.ndarray]) -> list[F32]:
    """The two corner sets' hues, merged into one increasing list.

    Both are already sorted, so this is a merge rather than a sort, and an
    exactly shared hue is taken once.
    """
    merged: list[F32] = []
    i = j = 1
    while i <= CUSP_CORNERS or j <= CUSP_CORNERS:
        left = reach[i][2] if i <= CUSP_CORNERS else math.inf
        right = limiting[j][2] if j <= CUSP_CORNERS else math.inf
        if left == right:
            merged.append(left)
            i += 1
            j += 1
        elif left < right:
            merged.append(left)
            i += 1
        else:
            merged.append(right)
            j += 1
    return merged


def _hue_table(corner_hues: list[F32]) -> np.ndarray:
    """360 hues, spaced as evenly as landing on every corner allows.

    A uniform degree-per-entry table would interpolate the gamut cusp across a
    corner, where the cusp has a crease. This table instead samples every
    corner exactly and distributes what is left evenly between them, so each
    interval is straight and the crease falls on a knot.

    The result is near enough to uniform that a lookup starts from the entry a
    uniform table would have used and searches a handful of entries either
    side (`search_range`), rather than bisecting all 360.
    """
    table = np.zeros(TABLE_SIZE, dtype=np.float32)
    positions: list[int] = []
    previous: int | None = None
    lowest = 0 if corner_hues[0] == 0.0 else 1
    for index, hue in enumerate(corner_hues):
        # Round to the entry this hue would ideally occupy, then keep the
        # positions strictly increasing: two corners rounding to one entry
        # would leave an interval with no samples in it at all.
        #
        # Half rounds up, as C's ``round`` does. Python's rounds to even, which
        # would put a corner falling exactly on a half in the entry below.
        position = min(max(math.floor(float(hue) + 0.5), lowest), NOMINAL_HUES - 1)
        if previous == position:
            if index > 1 and positions[index - 2] != positions[index - 1] - 1:
                positions[index - 1] -= 1
            else:
                position += 1
        positions.append(min(position, NOMINAL_HUES - 1))
        previous = lowest = position

    filled = 0
    lower = F32(0.0)
    for index, hue in enumerate(corner_hues):
        count = positions[index] - (positions[index - 1] if index else 0)
        _fill(table, filled + FIRST_NOMINAL, count, lower, hue)
        filled += count
        lower = hue
    _fill(table, filled + FIRST_NOMINAL, NOMINAL_HUES - filled, lower, HUE_LIMIT)

    table[0] = table[LAST_NOMINAL] - HUE_LIMIT
    table[UPPER_WRAP] = table[FIRST_NOMINAL] + HUE_LIMIT
    table[UPPER_WRAP + 1] = table[FIRST_NOMINAL + 1] + HUE_LIMIT
    return table


def _fill(table: np.ndarray, base: int, count: int, lower: F32, upper: F32) -> None:
    """One evenly spaced run, closed below and open above."""
    if count <= 0:
        return
    delta = F32((F32(upper) - F32(lower)) / F32(count))
    for offset in range(count):
        table[base + offset] = F32(lower) + F32(offset) * delta


def _search_range(hues: np.ndarray) -> tuple[int, int]:
    """How far a hue lookup strays from the entry a uniform table would use.

    OCIO's ``determine_hue_linearity_search_range``, and the reason the lookup
    is a two-step bisection rather than a nine-step one. The window is padded
    above by one because the lookup returns the interval's *upper* index.
    """
    low, high = 0, 1
    for index in range(FIRST_NOMINAL, UPPER_WRAP):
        delta = index - (FIRST_NOMINAL + int(hues[index]))
        low = min(low, delta)
        high = max(high, delta + 1)
    return low, high


def _ocio_tables(transform: Any) -> tuple[np.ndarray, np.ndarray]:
    """The reach and gamut cusp tables, read off OCIO's own shader description.

    OCIO builds both to construct any renderer for this op, and publishes them
    as textures on a `GpuShaderDesc`. A processor is built here from the op
    alone rather than reusing the caller's, so the tables belong to this op
    whatever else the caller's processor holds.
    """
    processor = OCIO.Config.CreateRaw().getProcessor(transform)
    description = OCIO.GpuShaderDesc.CreateShaderDesc(
        language=OCIO.GPU_LANGUAGE_GLSL_4_0
    )
    processor.getDefaultGPUProcessor().extractGpuShaderInfo(description)

    found: dict[str, np.ndarray] = {}
    for texture in description.getTextures():
        for key in ("reach_m_table", "gamut_cusp_table"):
            if key in texture.textureName:
                found[key] = np.asarray(texture.getValues(), dtype=np.float32)
    if set(found) != {"reach_m_table", "gamut_cusp_table"}:
        raise LookupError(
            "OCIO published no reach and gamut cusp tables for "
            "ACES_OUTPUT_TRANSFORM_20; this compiler reads both off its shader "
            f"description and found {sorted(found)}"
        )
    return found["reach_m_table"], found["gamut_cusp_table"].reshape(-1, 3)


@dataclasses.dataclass(frozen=True)
class OutputTransform:
    """Everything the emitter needs for one `ACES_OUTPUT_TRANSFORM_20`."""

    peak_luminance: F32
    inward: Appearance
    outward: Appearance
    tone: ToneScale

    #: Where the tone scale tops out, in lightness. Both compressions read it
    #: as their ceiling; the shape parameter they pair it with is ``INV_CZ``,
    #: which is a module constant because it depends on nothing here.
    limit_lightness: F32

    #: The chroma compression: how hard, how saturated, and where saturation
    #: gives way, plus the scale on the hue-dependent norm.
    compress: F32
    saturation: F32
    saturation_threshold: F32
    chroma_scale: F32

    #: The gamut compression: the lightness the focus point leans towards, how
    #: far the compression vectors converge, and the lower hull's exponent.
    mid_lightness: F32
    focus_distance: F32
    lower_hull_gamma_inv: F32

    #: The hue-indexed tables and the window a hue lookup searches.
    hues: np.ndarray
    cusps: np.ndarray
    reach: np.ndarray
    search_range: tuple[int, int]


def output_transform(transform: Any) -> OutputTransform:
    """Resolve one op's nine parameters into everything its graph needs."""
    params = [float(value) for value in transform.getParams()]
    peak_luminance = F32(params[0])
    limiting: Primaries = tuple(
        (params[1 + 2 * i], params[2 + 2 * i]) for i in range(4)
    )

    inward = appearance(AP0)
    outward = appearance(limiting)
    tone = tone_scale(peak_luminance)
    log_peak = tone.log_peak

    limit_lightness = luminance_to_lightness(peak_luminance)
    reach, cusps = _ocio_tables(transform)
    hues = _hue_table(
        _merge_hues(
            _reach_corners(appearance(AP1), limit_lightness, tone.forward_limit),
            _limiting_corners(outward, peak_luminance),
        )
    )

    return OutputTransform(
        peak_luminance=peak_luminance,
        inward=inward,
        outward=outward,
        tone=tone,
        limit_lightness=limit_lightness,
        compress=CHROMA_COMPRESS + (CHROMA_COMPRESS * CHROMA_COMPRESS_FACT) * log_peak,
        saturation=max(
            CHROMA_SATURATION_FLOOR,
            CHROMA_EXPAND - (CHROMA_EXPAND * CHROMA_EXPAND_FACT) * log_peak,
        ),
        saturation_threshold=F32(CHROMA_EXPAND_THRESHOLD / peak_luminance),
        chroma_scale=np.power(
            CHROMA_SCALE_GAIN * peak_luminance, CHROMA_SCALE_EXPONENT, dtype=np.float32
        )
        - CHROMA_SCALE_OFFSET,
        mid_lightness=luminance_to_lightness(tone.c_t * REFERENCE_LUMINANCE),
        focus_distance=FOCUS_DISTANCE
        + FOCUS_DISTANCE * FOCUS_DISTANCE_SCALING * log_peak,
        lower_hull_gamma_inv=F32(1.0)
        / (LOWER_HULL_GAMMA + LOWER_HULL_GAMMA_SCALING * log_peak),
        hues=hues,
        cusps=cusps,
        reach=reach,
        search_range=_search_range(hues),
    )
