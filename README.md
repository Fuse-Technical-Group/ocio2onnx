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

`FixedFunction` — the ACES output transform and the Rec.2100 surround
adjustment — is **refused by name at compile**, not approximated. Of the
159 transforms, 131 compile and 28 refuse; `Rec.2100-HLG - Display` is
among the refusals, because HLG's surround adjustment is not a transfer
curve.

Reproduce any of this for a config of your choice:

```sh
python tools/census.py [config-uri]
```

## Live parameters

A transform's dynamic properties — exposure, contrast, gamma, ASC CDL —
compile to graph *inputs*, so a grade varies per frame without
recompiling. This is what a baked LUT cannot do, and it is why the emitted
artifact is a graph rather than a table.

## Verification

OCIO's own CPU processor is the oracle. Every emitted graph is
property-tested against it over a generated input lattice, including
out-of-range and breakpoint values, and must agree within 2e-5 absolute
or 1e-4 relative, whichever is looser. Both bounds are needed: these
transforms cross zero, where a relative measure is meaningless, and they
also reach 1e8, where an absolute one is. Coverage is asserted rather
than sampled — every color space pair in a config either verifies or
refuses with a named op.

## Status

Early. The specification and roadmap are written; the compiler is not yet
implemented. See [ROADMAP.md](ROADMAP.md).

## License

BSD-3-Clause, matching OpenColorIO. `onnx` is Apache-2.0. No copyleft
dependencies.
