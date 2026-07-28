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
— eight op types appear. `FixedFunction` is counted as its two styles,
because they are unrelated transforms sharing a class and the compiler
emits one of them:

| Op | Occurrences |
| --- | --- |
| `Matrix` | 284 |
| `Range` | 52 |
| `Lut1D` | 40 |
| `Exponent` | 26 |
| `ExponentWithLinear` | 24 |
| `LogCamera` | 24 |
| `FixedFunction[ACES_OUTPUT_TRANSFORM_20]` | 24 |
| `FixedFunction[REC2100_SURROUND]` | 5 |
| `Log` | 4 |

**No transform in that config requires a 3D LUT.** The census counts ops;
what governs build order is how the 159 transforms partition:

| Transform class | Count |
| --- | --- |
| closed-form ops only | 111 |
| `Lut1D`, half-domain | 18 |
| `Lut1D`, uniform | 6 |
| refused (`ACES_OUTPUT_TRANSFORM_20`) | 24 |

Six closed-form emitters — `Matrix`, `Range`, `Exponent`,
`ExponentWithLinear`, `Log`, `LogCamera` — cover 111 of the 159 transforms
with no LUT machinery at all. `Lut1D` adds the remaining 24 and is by a
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

**An unimplemented op is refused, by name, at compile.** OCIO reports the
op list before anything executes, so a transform containing one fails
naming the op, the transform, and its endpoints, rather than emitting a
graph that is quietly wrong. *Why refuse rather than approximate*: an
approximation of a display rendering transform is a picture nobody signed
off on.

**The name is the style, not the class.** `FixedFunction` is two unrelated
transforms sharing a class, so the class is too coarse to key an emitter on
and too coarse to refuse by. Every place the compiler names an op — the
emitter registry, the census, a refusal — uses one label carrying the type
plus whatever attribute distinguishes one behaviour of it from another.
That is what lets `REC2100_SURROUND` be emitted while
`ACES_OUTPUT_TRANSFORM_20` goes on refusing, and it keeps the census
honest: a row half of which is emitted could be marked neither emitted nor
refused.

Both styles are closed-form — neither needs a 3D LUT in OCIO's own GPU
partition, which is the measure of whether an op can be expressed
analytically at all. What separated them was scope and risk, by an order of
magnitude in both:

- `REC2100_SURROUND` (5 transforms) is a surround/system-gamma adjustment
  scaling all three channels by a power of their luminance. Small, and the
  only thing that stood between this compiler and HLG display output, so
  it went first. It ships.
- `ACES_OUTPUT_TRANSFORM_20` (24 transforms) is the ACES 2.0 output
  transform: tone scale, chroma compression, and gamut mapping. Published
  with a reference implementation, and substantial. It is also *the look*
  — the most visible math in any pipeline that runs it — so it is the
  worst candidate for an early port and the best candidate for a careful
  one. Deferred behind a named trigger (§road:aces-output-transform).

**Refusal is by op, not by category.** 135 of the 159 transforms compile;
24 refuse, and every one is an ACES 2.0 display rendering. Stating the
boundary as an op set rather than as "view transforms are unsupported" is
what makes it predictable: a caller reads which op blocks them, and every
other view compiles. `ocio2onnx census` reports the split for any config,
against the op set the compiler actually implements rather than a second
list beside it; `tools/census.py` is a shim over the same code.

## How ops emit §spec:op-emission
*Status: in progress*

Every op emits as ONNX arithmetic over parameters read from OCIO's
transform introspection. `Matrix` is a 1×1 convolution, `Range` a clamp
and an affine, the exponential and logarithmic ops elementwise chains.
ONNX has no per-element branching, so an op with a breakpoint evaluates
both sides and selects; that costs arithmetic, not correctness.

**`REC2100_SURROUND` is the one op whose arithmetic crosses channels.** A
single luminance per pixel scales all three, so a per-channel reading of it
agrees on every neutral pixel and disagrees on every coloured one — a
misreading nothing but an oracle sampling coloured pixels would catch
(§spec:verification). Its fold onto `|Y|` and its luminance floor are read
off OCIO's own renderer rather than chosen, as `Log`'s floor is: the floor
is `1e-4` forward, and that value's image under the forward op inverse,
because the inverse clamps in the forward's output domain.

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
monotonic table inverts once, offline, onto the **half domain**. An
inverse therefore reaches the graph as an ordinary forward half-domain
table, read by the gather and lerp that already exists, and direction
costs no second code path and no run-time search. A table that does not
rise has no inverse to read off and is refused by name.

**The inversion grid is the half domain because the oracle rejected a
uniform one.** This section proposed a uniform-domain table and left the
question open; the oracle (§spec:verification) overruled it. Across the
lattices of the eight transforms carrying an inverse — 10134 samples
against the CPU processor — the half domain misses nothing, while a
uniform grid over the forward table's output range misses 2267 samples by
as much as 2.06. The reason is dynamic range. Those output ranges are
enormous — `ACEScc` spans [-5.7e-07, 96617.7], `Apple Log` [-0.056,
3.4e+38] — so a uniform grid puts nearly every sample above middle grey
and leaves the toe, which is most of a picture, to one interval. Sample
count does not rescue it: 65536 uniform samples still miss 2138. A half's
spacing is geometric, so it holds the same *relative* resolution in every
decade, which is the shape a log-encoded curve has.

Where the forward table is flat its inverse is ambiguous across a whole
interval, and the ends of that interval sit half a domain apart. Each flat
run collapses to the end the curve leaves it at — the edge of the interval
the table is invertible over — and outside that interval the inverse
clamps. Read off OCIO rather than chosen: OCIO answers -65504 with 0 for
`ref -> Apple Log`, whose decode is flat across [-65504, 0], and 65504
with 3.7597656 for `ref -> ADX10`, whose is flat across [3.7597656,
65504]. Taking the same end of every run puts one of those two far wrong.

**Where OCIO ships a curve as a table, the compiler emits the table.**
`Rec.2100-PQ - Display` resolves to a `BuiltinTransform` that OCIO
implements as a half-domain LUT, and it stays a LUT at every optimization
level from none to all. PQ is closed-form as mathematics but not as
anything OCIO hands over. Emitting a closed-form PQ instead would mean
defining a transform, which §spec:non-goals forbids, and would read as
error against the oracle.

## Verification §spec:verification
*Status: in progress*

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
legitimately produces outputs around 1e8.

**The oracle runs with fast math off.** OCIO's default CPU optimization
includes `OPTIMIZATION_FAST_LOG_EXP_POW`, an approximate `pow`, `log`, and
`exp` whose error exceeds this tolerance. Left on, it measures a math
library rather than the compiler's reading of the config: 110 of the 111
closed-form transforms verify against OCIO's fast path, and all 111 verify
against its accurate one. Which optimization flags the oracle runs under
is a correctness decision, not a tuning knob — the reference has to be
OCIO's accurate arithmetic, or a compiler error and an oracle error are
indistinguishable.

**`OPTIMIZATION_LUT_INV_FAST` stays on**, which is the same decision read
the other way. Cleared, OCIO's inverse `Lut1D` renderer parts company with
its own fast path beyond tolerance at 5 samples across the pinned config,
every one at an input of 65504 — and the fast path is the one that agrees
with the encoding, which puts 65504 at ACEScc 1.468 where the cleared path
answers 1.4987. A flag is cleared where it makes the reference less
accurate. Reaching a coverage number by loosening the oracle is a failure,
not a pass, and so is tightening it past the reference.

*Why an oracle rather than reference constants*: the failure mode here is
not a wrong constant, it is a subtly wrong *interpretation* — a direction
inverted, a breakpoint on the wrong side, a LUT indexed by `n` where the
reference uses `n-1`. OCIO's own FAQ records that last one costing roughly
3% gain against Nuke. Hand-checked values do not catch a systematic
misreading; a lattice against the reference does.

Coverage is asserted, not sampled: for every color space pair in the
pinned config, the compiler either emits a verified graph or refuses with
a named op.

### What counts as evidence §spec:evidence-floor

Some transforms overflow float32 somewhere in the lattice: 12 of the pinned
config's 135 verified transforms do, at up to 36 of 1248 samples. A
non-finite reference value carries no magnitude to measure a tolerance
against, so agreement at those samples is asserted on the *kind* of value
instead — `+inf`, `-inf`, and NaN are three distinct answers, and the graph
shall return the one the reference returned. Every sample is therefore
evidence: a sample is compared against the tolerance or matched against a
class, never dropped.

*Why not drop them*: a direction inverted at overflow, the reference falling
to `-inf` where the graph climbs to `+inf`, is exactly the misreading this
section exists to catch. Dropping the sample reads that as agreement.

A verified graph shall also have at least one finite sample compared. A
transform whose reference is non-finite across the whole lattice otherwise
compares nothing and reports agreement — a graph certified on no evidence,
which is worse than one that refuses, because "verified" is a claim the
consumer acts on. No transform in the pinned config is in that state today;
a config the compiler is handed is arbitrary (§spec:problem-statement), and
a NaN coefficient in a config-declared matrix is enough to reach it.

*Why a floor of one sample rather than a fraction of the lattice*: a
percentage would need a constant this document cannot derive from anything,
and would refuse a legitimately overflow-heavy transform added later. Once
every sample is checked one way or the other, the only quantity left worth
asserting is that the finite kind occurred at all.

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

float32 is not exact either, but the residual that showed up in practice
was not float32's. It came from the oracle's own approximate `pow`, and
clearing that flag removes it (§spec:verification). The caution above is
about the consumer's executor, not about the reference.

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
