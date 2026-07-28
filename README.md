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
in both directions plus every display view — that list uses **eight op
types, and needs no 3D LUT at all**:

| Op | Occurrences |
| --- | --- |
| `Matrix` | 284 |
| `Range` | 52 |
| `Lut1D` | 40 |
| `FixedFunction` | 29 |
| `Exponent` | 26 |
| `ExponentWithLinear` | 24 |
| `LogCamera` | 24 |
| `Log` | 4 |

Counting ops understates the reach, though. What matters is how the
transforms partition:

| Transform class | Count |
| --- | --- |
| closed-form ops only | 111 |
| `Lut1D`, half-domain | 14 |
| `Lut1D`, uniform | 6 |
| refused (`FixedFunction`) | 28 |

**Six closed-form emitters reach 111 of the 159 transforms with no lookup
table at all** — every camera log encoding, every ACES working space,
sRGB and the gamma displays. `Lut1D` adds the remaining 20 and costs
several times as much, because 34 of the 40 `Lut1D` ops are half-domain:
65536 entries indexed by the float16 *bit pattern* of the input, which
standard ONNX has no cast to reach. OCIO hits the same wall on GLSL 1.2
and reconstructs the index arithmetically; the compiler follows its
shaders rather than inventing a second answer.

Vendor differences are **coefficients, not code**: RED Log3G10, ARRI
LogC4, CanonLog3, Apple Log and BMDFilm are all one parametric
`LogCamera` op with different numbers, which OCIO supplies. Adding a
camera is a config update.

The other two are **refused by name at compile**, not approximated:
`Lut1D`, and `FixedFunction` — the ACES output transform and the
Rec.2100 surround adjustment. 48 transforms refuse; 28 of them carry a
`FixedFunction` and the other 20 only a `Lut1D`. `Rec.2100-HLG - Display`
is among the refusals, because HLG's surround adjustment is not a
transfer curve — not because display views are unsupported. Stating the
boundary as an op set is what keeps that case from being a surprise.

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
```

A transform carrying an op the compiler does not emit is refused, naming
the op, the transform and its endpoints, and exits non-zero:

```text
FixedFunction[ACES_OUTPUT_TRANSFORM_20] is not emitted by this compiler, so 'ACES2065-1 -> sRGB - Display / ACES 2.0 - SDR 100 nits (Rec.709)' is refused rather than approximated
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

Planned. A transform's dynamic properties — exposure, contrast, gamma,
ASC CDL — compile to graph *inputs*, so a grade varies per frame without
recompiling. This is what a baked LUT cannot do, and it is why the emitted
artifact is a graph rather than a table. See [ROADMAP.md](ROADMAP.md).

## Verification

OCIO's own CPU processor is the oracle. Every emitted graph is
property-tested against it over a generated input lattice — out-of-range
values, values either side of zero, and every op's breakpoints — and must
agree within 2e-5 absolute or 1e-4 relative, whichever is looser. Both
bounds are needed: these transforms cross zero, where a relative measure is
meaningless, and they also reach 1e8, where an absolute one is.

Coverage is asserted rather than sampled: every transform in a config
either verifies or refuses with a named op, and none are skipped.

```sh
ocio2onnx verify [--config URI]
```

Against the pinned ACES Studio config: **111 verified, 48 refused, 0
failed, 0 skipped, 159 total**.

## Status

The closed-form compiler ships, with the oracle harness. `Lut1D` and the
two `FixedFunction` styles are refused by name rather than approximated;
they are the next sections of [ROADMAP.md](ROADMAP.md), along with live
parameters.

## License

BSD-3-Clause, matching OpenColorIO. `onnx` is Apache-2.0. No copyleft
dependencies.
