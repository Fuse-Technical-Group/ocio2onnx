# ocio2onnx — System Specification

## Problem statement §spec:problem-statement
*Status: complete*

OpenColorIO is the industry's color transform database: every camera
vendor's log encoding, every ACES working space, every display transform,
maintained by the people who define them. Its execution targets are
shading languages — OCIO 2.5 emits GLSL (six variants), HLSL, Cg, MSL, and
OSL. Every one of them needs a graphics context.

A CUDA-resident real-time video pipeline has no graphics context, and
acquiring one to run a 3×3 matrix is not a trade worth making. Such a
pipeline has three options today, all bad:

- **Interop with OpenGL** — a GL context beside CUDA and a texture round
  trip on every frame, to run arithmetic the GPU is already holding data
  for.
- **Bake a LUT** — approximates a transform that is mostly closed-form,
  and cannot carry a live parameter at all: a grade knob would mean
  re-baking every frame the operator touches it.
- **Reimplement the math** — unbounded work against per-vendor curves,
  and unverifiable against the reference.

`ocio2onnx` compiles an OCIO transform into an ONNX graph. The graph runs
wherever ONNX runs — ONNX Runtime, TensorRT, or a consumer's own executor
— with no graphics context, and with no OCIO dependency at execution
time. OCIO remains the authority for what a transform *is*; this project
only changes what it can be executed by.

## What the compiler emits §spec:emitted-graph
*Status: not started*

A single ONNX graph per transform: a channels-first float image tensor in,
the same shape out, with spatial dimensions free so one graph serves every
resolution. Dynamic properties become additional graph inputs
(§spec:dynamic-properties).

**ONNX, not a framework.** ONNX is a portable IR with multiple mature
executors, so a consumer picks its own runtime and inherits that runtime's
fusion. A 1×1 convolution followed by a short elementwise chain is
memory-bandwidth bound and fuses to a single pass — the same place a
hand-written kernel lands.

*Rejected*: emitting **torch**, which couples every consumer to one
framework for math that is framework-independent, and which has no
counterpart anywhere in OCIO's own architecture. *Rejected for now*: a
`GPU_LANGUAGE_CUDA` target contributed to OCIO itself, which is a coherent
idea and follows the path Metal took — but it is a C++ backend to
maintain, and it still leaves each consumer compiling and loading kernels.
That contribution stays open as a complement, not a substitute: it serves
consumers who want OCIO in-process, where this project serves consumers
who want the transform as a portable artifact.

## Op coverage §spec:op-coverage
*Status: not started*

Coverage is measured, not assumed. Across
`studio-config-v4.0.0_aces-v2.0_ocio-v2.5` — every color space in both
directions against the reference, plus every display view, 159 transforms
— eight op types appear:

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

**No transform in that config requires a 3D LUT.** Forty-eight require a
single 1D LUT. The compiler implements seven of the eight ops directly;
`Lut1D` emits as a gather and lerp, the rest as closed-form arithmetic.

**Vendor curves are data, not code.** `LogCamera` is one parametric op —
base, log-side slope and offset, lin-side slope and offset, and a linear
break. RED Log3G10, ARRI LogC3 and LogC4, CanonLog2 and CanonLog3, Apple
Log, and BMDFilm are all that op with different coefficients, which OCIO
hands over as plain floats. Supporting a new camera is a config update
with no change here. *Why this matters*: the alternative is per-vendor
implementations against curves whose definitions are not uniformly
published — unbounded work, and unverifiable.

**`FixedFunction` is refused, by name, at compile.** Only two styles occur
— `ACES_OUTPUT_TRANSFORM_20` and `REC2100_SURROUND` — and both are
tone-mapping and gamut-compression math that should not be hand-ported.
OCIO reports the op list before anything executes, so a transform
containing an unimplemented op fails naming the op, the transform, and its
endpoints, rather than emitting a graph that is quietly wrong. *Why refuse
rather than approximate*: an approximation of a display rendering
transform is a picture nobody signed off on.

**Refusal is by op, not by category.** 131 of the 159 transforms compile;
28 refuse. Most are the ACES output transforms, where a caller expects it
— but `Rec.2100-HLG - Display` refuses too, because HLG's surround
adjustment is `REC2100_SURROUND` rather than a transfer curve. A consumer
wanting plain HLG encoding is served; one wanting the ACES HLG display
rendering is not. Stating the boundary as an op set rather than as "view
transforms are unsupported" is what keeps that case from being a
surprise. `tools/census.py` reports the split for any config.

## Verification §spec:verification
*Status: not started*

OCIO's CPU processor is the oracle. For every transform the compiler
claims to support, a test generates an input lattice — including values
outside `[0, 1]`, near zero, and at each op's breakpoints — runs it
through both the CPU processor and the emitted graph, and asserts
agreement within a stated tolerance.

*Why an oracle rather than reference constants*: the failure mode here is
not a wrong constant, it is a subtly wrong *interpretation* — a direction
inverted, a breakpoint on the wrong side, a LUT indexed by `n` where the
reference uses `n-1`. OCIO's own FAQ records that last one costing roughly
3% gain against Nuke. Hand-checked values do not catch a systematic
misreading; a lattice against the reference does.

Coverage is asserted, not sampled: for every color space pair in the
pinned config, the compiler either emits a verified graph or refuses with
a named op.

## Dynamic properties §spec:dynamic-properties
*Status: not started*

OCIO exposes seven dynamic property types — `EXPOSURE`, `CONTRAST`,
`GAMMA`, and four `GRADING_*` families — plus ASC CDL as slope, offset,
power, and saturation. A transform carrying them compiles to a graph with
those values as **additional inputs**, so a consumer varies them per frame
without recompiling.

This is what a baked LUT cannot do, and it is the reason the emitted
artifact is a graph rather than a table: a live grade is the same compiled
transform with its parameters left free.

Scalar and per-channel properties — exposure, contrast, gamma, and the CDL
quartet — map to graph inputs directly. The curve-shaped `GRADING_*`
families do not, and are deferred: their parameters are control points
rather than values, so they arrive with a tensor-valued input or not at
all.

## Precision §spec:precision
*Status: not started*

The graph is emitted at float32 and declares that. A log or exponential
chain evaluated at float16 loses accuracy where the curve is steepest,
which is the toe — the part of the image a grade is most often reaching
for. A consumer whose executor selects reduced precision by default must
pin float32 for these graphs or accept a documented, measured error
against the oracle (§spec:verification).

## Non-goals §spec:non-goals
*Status: complete*

- **Not a LUT baker.** OCIO already has `Baker` and `ociobakelut`. This
  project exists because a table cannot carry a live parameter.
- **Not an OCIO fork, backend, or reimplementation.** It consumes OCIO's
  public API and holds no opinion its config does not already state.
- **Not a color science library.** It defines no transform. Every
  coefficient originates in an OCIO config.
- **Not an executor.** It emits an artifact; running it is the consumer's
  concern.
