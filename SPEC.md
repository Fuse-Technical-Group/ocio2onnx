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
*Status: complete*

A single ONNX graph per transform: a channels-first float image tensor in,
the same shape out, with spatial dimensions free so one graph serves every
resolution. Dynamic properties become additional graph inputs
(§spec:dynamic-properties).

The tensor carries three channels. No transform in the pinned config
touches alpha — every matrix carries an identity alpha row, every exponent
an alpha term of 1 — yet OCIO's own RGBA path still runs alpha through
that identity arithmetic and perturbs it by roughly 3e-6 at float32.
Alpha therefore bypasses the graph rather than flowing through it, and a
consumer holding RGBA carries it alongside, unchanged. Widening the
interface to four channels for a caller's convenience stays open; alpha
bypassing the arithmetic does not.

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
*Status: in progress*

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

**No transform in that config requires a 3D LUT.** The census counts ops;
what governs build order is how the 159 transforms partition:

| Transform class | Count |
| --- | --- |
| closed-form ops only | 111 |
| `Lut1D`, half-domain | 14 |
| `Lut1D`, uniform | 6 |
| refused (`FixedFunction`) | 28 |

Six closed-form emitters — `Matrix`, `Range`, `Exponent`,
`ExponentWithLinear`, `Log`, `LogCamera` — cover 111 of the 159 transforms
with no LUT machinery at all. `Lut1D` adds the remaining 20 and is by a
wide margin the most involved emitter (§spec:op-emission). Sequencing
follows that split rather than the op census, which understates how much
of the config is reachable without a table.

OCIO's own GPU path needs a sampled texture for 48 transforms — more than
this compiler does. Eight of those carry no `Lut1D` op at all: OCIO's
shader baked a closed-form `Exponent` or `ExponentWithLinear` into a
texture, and all eight are transforms this compiler refuses anyway. Where
OCIO's shaders sample, this compiler evaluates.

**Direction is a second axis.** Ops arrive both ways: 18 of 26 `Exponent`,
17 of 24 `ExponentWithLinear`, 12 of 24 `LogCamera`, and 2 of 4 `Log` are
inverse. `Matrix` and `Range` never are, because OCIO folds them. The
inverse of a closed-form op is closed-form, so the six emitters are about
eleven code paths — cheap, but not free, and not cheap at all for `Lut1D`.

**Vendor curves are data, not code.** `LogCamera` is one parametric op —
base, log-side slope and offset, lin-side slope and offset, and a linear
break. RED Log3G10, ARRI LogC3 and LogC4, CanonLog2 and CanonLog3, Apple
Log, and BMDFilm are all that op with different coefficients, which OCIO
hands over as plain floats. Supporting a new camera is a config update
with no change here. *Why this matters*: the alternative is per-vendor
implementations against curves whose definitions are not uniformly
published — unbounded work, and unverifiable.

**`FixedFunction` is refused, by name, at compile.** OCIO reports the op
list before anything executes, so a transform containing an unimplemented
op fails naming the op, the transform, and its endpoints, rather than
emitting a graph that is quietly wrong. *Why refuse rather than
approximate*: an approximation of a display rendering transform is a
picture nobody signed off on.

**Refused is not unreachable.** Both styles are closed-form — neither
needs a 3D LUT in OCIO's own GPU partition, which is the measure of
whether an op can be expressed analytically at all. Refusal is a scope
and risk decision, and the two styles differ sharply in both:

- `REC2100_SURROUND` (5 transforms) is a surround/system-gamma
  adjustment — a small op, and the only thing standing between this
  compiler and HLG display output.
- `ACES_OUTPUT_TRANSFORM_20` (24 transforms) is the ACES 2.0 output
  transform: tone scale, chroma compression, and gamut mapping. Published
  with a reference implementation, and substantial. It is also *the look*
  — the most visible math in any pipeline that runs it — so it is the
  worst candidate for an early port and the best candidate for a careful
  one.

**Refusal is by op, not by category.** 131 of the 159 transforms compile;
28 refuse. Most are the ACES output transforms, where a caller expects it
— but `Rec.2100-HLG - Display` refuses too, because HLG's surround
adjustment is `REC2100_SURROUND` rather than a transfer curve. A consumer
wanting plain HLG encoding is served; one wanting the ACES HLG display
rendering is not. Stating the boundary as an op set rather than as "view
transforms are unsupported" is what keeps that case from being a
surprise. `ocio2onnx census` reports the split for any config, against the
op set the compiler actually implements rather than a second list beside
it; `tools/census.py` is a shim over the same code.

## How ops emit §spec:op-emission
*Status: in progress*

Every op emits as ONNX arithmetic over parameters read from OCIO's
transform introspection. `Matrix` is a 1×1 convolution, `Range` a clamp
and an affine, the exponential and logarithmic ops elementwise chains.
ONNX has no per-element branching, so an op with a breakpoint evaluates
both sides and selects; that costs arithmetic, not correctness.

`Lut1D` is the exception, because most of them are half-domain: 34 of the
40 in the config hold 65536 entries indexed by the bit pattern of the
input rounded to float16, not by a normalized coordinate. Standard ONNX
has no bit-reinterpreting cast, so the index has to be arrived at some
other way.

**OCIO's own shaders are that other way.** OCIO has the same constraint —
it targets GLSL 1.2 and ES 1.0, which have no bitwise integer ops — and
reconstructs the index in float arithmetic: an exponent from
`floor(log2)`, a mantissa fraction, a denormal case, and a sign offset.
Every step has an ONNX counterpart, so the compiler reproduces that
derivation rather than inventing one. Reading the reference implementation
is cheaper than rediscovering it, and it is the same reasoning that makes
OCIO the oracle in §spec:verification: the authority already exists.

The graph then gathers the two adjacent entries and interpolates, matching
OCIO's CPU processor, which interpolates across half slots rather than
snapping to one.

*Rejected*: resampling a half-domain LUT onto a uniform grid. It trades an
exact lookup for an approximation the oracle would then report as error,
and buys nothing — the arithmetic index costs about ten ops.

The emitted table is flat. OCIO packs half-domain LUTs into a 4096×17 2D
texture because GPU texture widths cap below 65536; ONNX initializers have
no such limit, so neither the packing nor its stride carries over.

An inverse `Lut1D` inverts at compile time rather than searching at run
time. OCIO hands over the forward table in both directions, and a
monotonic table inverts once, offline, to a uniform-domain table the graph
reads as an ordinary gather and lerp. Whether that resampling holds
tolerance is the oracle's finding (§spec:verification), not an assumption
this section is entitled to make.

**Where OCIO ships a curve as a table, the compiler emits the table.**
`Rec.2100-PQ - Display` resolves to a `BuiltinTransform` that OCIO
implements as a half-domain LUT, and it stays a LUT at every optimization
level from none to all. PQ is closed-form as mathematics but not as
anything OCIO hands over. Emitting a closed-form PQ instead would mean
defining a transform, which §spec:non-goals forbids, and would read as
error against the oracle.

## Verification §spec:verification
*Status: complete*

OCIO's CPU processor is the oracle. For every transform the compiler
claims to support, a test generates an input lattice — including values
outside `[0, 1]`, near zero, and at each op's breakpoints — runs it
through both the CPU processor and the emitted graph, and asserts
agreement within 2e-5 absolute or 1e-4 relative, whichever is looser.

**The tolerance is stated both ways because either alone is wrong.** A
purely relative bound is meaningless where these transforms cross zero: a
matrix row that nearly cancels yields values at which any relative measure
explodes while the absolute error sits on the float32 noise floor. A
purely absolute bound fails at the other end, where an input at 65504
legitimately produces outputs around 1e8. The floor itself belongs to the
executor rather than the compiler — `Pow` differs between ONNX Runtime and
OCIO in the last digits — so a tighter bound would measure a math library
instead of the compiler's reading of the config.

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
*Status: complete*

The graph is emitted at float32 and declares that. A log or exponential
chain evaluated at float16 loses accuracy where the curve is steepest,
which is the toe — the part of the image a grade is most often reaching
for. A consumer whose executor selects reduced precision by default must
pin float32 for these graphs or accept a documented, measured error
against the oracle (§spec:verification).

float32 is not exact either, and the residual is not the compiler's to
remove: `Pow` and the transcendentals differ between executors and OCIO in
their last digits, which is what sets the verification tolerance rather
than any choice made here (§spec:verification).

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
