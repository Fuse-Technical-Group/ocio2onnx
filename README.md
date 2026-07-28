# ocio2onnx

> [SPEC.md](SPEC.md) (system specification)
> | [ROADMAP.md](ROADMAP.md) (planned work)

Compile an [OpenColorIO](https://opencolorio.org) transform into an ONNX
graph.

OCIO is the industry's color transform database — every camera vendor's
log encoding, every ACES working space, every display transform. Its
execution targets are all shading languages: GLSL, HLSL, Cg, MSL, OSL.
Each one needs a graphics context.

A CUDA-resident real-time video pipeline has no graphics context. Today it
has to choose between an OpenGL interop round trip on every frame, a baked
LUT that cannot carry a live parameter, or reimplementing vendor curves by
hand. `ocio2onnx` compiles the transform instead, to a portable graph that
runs under ONNX Runtime, TensorRT, or anything else that reads ONNX — with
no graphics context, and no OCIO dependency at execution time.

OCIO stays the authority for what a transform *is*. This project only
changes what can execute it.

## How it works

OCIO reports a transform as a list of typed ops with readable parameters.
Measured across the ACES Studio config — 159 transforms, every color space
in both directions plus every display view — that list uses **nine op
labels, and needs no 3D LUT at all**. `FixedFunction` is counted as its two
styles, which are unrelated transforms sharing a class:

| Op | Occurrences |
| --- | --- |
| `Matrix` | 186 |
| `Range` | 52 |
| `Lut1D` | 40 |
| `Exponent` | 26 |
| `ExponentWithLinear` | 24 |
| `LogCamera` | 24 |
| `FixedFunction[ACES_OUTPUT_TRANSFORM_20]` | 24 |
| `FixedFunction[REC2100_SURROUND]` | 5 |
| `Log` | 4 |

Counting ops understates the reach, though. What matters is how the
transforms partition:

| Transform class | Count |
| --- | --- |
| closed-form ops only | 111 |
| `Lut1D`, half-domain | 37 |
| `Lut1D`, uniform | 3 |
| a fixed function, no table | 8 |

**Six closed-form emitters reach 111 of the 159 transforms with no lookup
table at all** — every camera log encoding, every ACES working space,
sRGB and the gamma displays. `Lut1D` adds 40 more and costs
several times as much, because 37 of the 40 `Lut1D` ops are half-domain:
65536 entries indexed by the float16 *bit pattern* of the input, which
standard ONNX has no cast to reach. OCIO hits the same wall on GLSL 1.2
and reconstructs the index arithmetically; the compiler follows its
shaders rather than inventing a second answer. An inverse table inverts
once at compile time, onto that same half domain, so running a transform
backwards costs no run-time search.

Vendor differences are **coefficients, not code**: RED Log3G10, ARRI
LogC4, CanonLog3, Apple Log and BMDFilm are all one parametric
`LogCamera` op with different numbers, which OCIO supplies. Adding a
camera is a config update.

An op the compiler does not emit is **refused by name at compile**, not
approximated, and the name is the style rather than the class — so
`REC2100_SURROUND` and `ACES_OUTPUT_TRANSFORM_20` are two entries rather
than one `FixedFunction`. Nothing in the pinned config is refused. The
boundary is still an op set rather than a category, because that is what
makes it predictable for a config you supply: the answer names the op that
blocks you, not the kind of transform it belongs to.

Reproduce the split for a config of your choice:

```sh
ocio2onnx census [--config URI]
```

## Usage

Compile a color space pair or a display view. `--verify` holds the graph
against OCIO's CPU processor before writing it:

```sh
ocio2onnx compile --from "Log3G10 REDWideGamutRGB" --to ACES2065-1 -o graph.onnx --verify
ocio2onnx compile --display "sRGB - Display" --view "Un-tone-mapped" -o srgb.onnx
ocio2onnx compile --display "Rec.2100-HLG - Display" --view "Video (colorimetric)" -o hlg.onnx
ocio2onnx compile --display "sRGB - Display" --view "ACES 2.0 - SDR 100 nits (Rec.709)" -o aces.onnx
```

A transform carrying an op the compiler does not emit is refused, naming
the op, the transform and its endpoints, and exits non-zero:

```text
GradingPrimary is not emitted by this compiler, so 'ACES2065-1 -> a graded space' is refused rather than approximated
```

The same two entry points from Python:

```python
import onnx

from ocio2onnx import compile_colorspaces, compile_display_view

onnx.save(compile_colorspaces("Log3G10 REDWideGamutRGB", "ACES2065-1"), "graph.onnx")
model = compile_display_view("sRGB - Display", "Un-tone-mapped", src="ACEScg")
```

Both take `config=` — a built-in URI or a file path — and raise
`AddressError` for a name the config does not carry, `UnsupportedOpError`
for an op the compiler does not emit. The graph is float32, three
channels, with batch and spatial dimensions free, and records the config
that produced it in its `metadata_props`.

## Live parameters

A transform's dynamic properties compile to graph *inputs*, so a grade
varies per frame without recompiling. This is what a baked LUT cannot do,
and it is why the emitted artifact is a graph rather than a table.

`ExposureContrast` carries `EXPOSURE`, `CONTRAST`, and `GAMMA`; ASC CDL
carries `CDL_SLOPE`, `CDL_OFFSET`, `CDL_POWER`, and `CDL_SATURATION`. Each
is an ONNX input backed by an initializer holding the config's own value,
so an input left unbound reads that default. `compile` names them:

```text
live parameters (bind to vary per frame; unbound reads the default):
  EXPOSURE        0.5
  CONTRAST        1.2
  GAMMA           1
  CDL_SLOPE       1.1, 1, 0.9
  CDL_OFFSET      0, 0, 0
  CDL_POWER       1.2, 1, 0.8
  CDL_SATURATION  1.3
```

From Python, `parameters(model)` answers the same question off the artifact,
and any runtime binds them the way it binds the image:

```python
from ocio2onnx import compile_colorspaces, parameters

model = compile_colorspaces("ref", "graded", config="grade.ocio")
parameters(model)  # {"EXPOSURE": array([0.5], dtype=float32), ...}
session.run(None, {"input": image, "EXPOSURE": [1.5]})
```

A knob needs an op to hang on, and OCIO drops an op set to its own identity
before the compiler sees it. So a template config whose grade is neutral —
exposure 0, contrast 1, slope `[1, 1, 1]` — compiles to a graph with no
inputs at all. A grade already off its identity is unaffected. To keep a
neutral one, either author it at the value the grade starts from, or declare
it dynamic, which OCIO's config syntax spells by *omitting* the parameter:

```yaml
# exposure is dynamic; the op survives and the graph carries all three knobs
from_scene_reference: !<ExposureContrastTransform> {style: linear, contrast: 1, gamma: 1}
```

A CDL has no such declaration — OCIO defines no dynamic property for one —
so a neutral CDL meant to be driven has to be authored off its identity.

The four `GRADING_*` families are not emitted — their parameters are control
points rather than values — and refuse by name like any other unimplemented
op. See [ROADMAP.md](ROADMAP.md).

## Verification

OCIO's own CPU processor is the oracle. Every emitted graph is
property-tested against it over a generated input lattice — out-of-range
values, values either side of zero, and every op's breakpoints — and must
agree within 2e-5 absolute or 1e-4 relative, whichever is looser. Both
bounds are needed: these transforms cross zero, where a relative measure is
meaningless, and they also reach 1e8, where an absolute one is.

Coverage is asserted rather than sampled: every transform in a config
either verifies or refuses with a named op, and none are skipped. No sample
is skipped either — where a transform overflows float32, the graph must
return the same kind of answer the reference did, `+inf` against `+inf` and
not against `-inf`, and a graph with no finite sample to compare does not
verify at all.

```sh
ocio2onnx verify [--config URI]
```

Against the pinned ACES Studio config: **159 verified, 0 refused, 0
failed, 0 skipped, 159 total**.

## Trusting a config

Loading a config is trusting it. An OCIO config may reference LUT files by
relative path, and OCIO reads them from wherever the config's search path
points — so `--config` on an untrusted file is a local file read on that
file's terms, not this compiler's. Point it at configs you would run
`ociocheck` on.

## Status

Op coverage is complete for the pinned config: every one of its 159
transforms compiles and verifies against OCIO's CPU processor — the
closed-form ops, the sampled lookups, and both `FixedFunction` styles,
including the ACES 2.0 display renderings. A grade is live: exposure,
contrast, gamma, and the ASC CDL quartet emit as graph inputs, each swept
against the oracle rather than checked at its default. The curve-shaped
`GRADING_*` families and packaging are what is left in
[ROADMAP.md](ROADMAP.md).

## License

BSD-3-Clause, matching OpenColorIO. `onnx` is Apache-2.0. No copyleft
dependencies.
