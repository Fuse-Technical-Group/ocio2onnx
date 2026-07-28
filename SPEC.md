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
*Status: complete*

Coverage is measured, not assumed. Across
`studio-config-v4.0.0_aces-v2.0_ocio-v2.5` — every color space in both
directions against the reference, plus every display view, 159 transforms
— nine op labels appear. `FixedFunction` is counted as its two styles,
because they are unrelated transforms sharing a class:

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

**That census counts the ops OCIO runs, not the ops the config declares.**
A processor reports its op list as written; OCIO's renderer runs an
optimized rewrite of it, and the compiler walks the rewrite so that the
emitted graph and the oracle measure the same arithmetic
(§spec:verification). The difference is not cosmetic. The declared list
holds 284 matrices, and a display view holds two adjacent ones that
compose to an identity: emitting both round-trips a near-black channel to
about 1e-10 rather than to zero, which a mirrored gamma further down the
view turns into 6e-5 against a reference of exactly black.

**No transform in that config requires a 3D LUT.** How the 159 partition:

| Transform class | Count |
| --- | --- |
| closed-form ops only | 111 |
| `Lut1D`, half-domain | 37 |
| `Lut1D`, uniform | 3 |
| a fixed function, no table | 8 |

Six closed-form emitters — `Matrix`, `Range`, `Exponent`,
`ExponentWithLinear`, `Log`, `LogCamera` — cover 111 of the 159 with no
LUT machinery at all. `Lut1D` adds 40 more and is by a wide margin the
most involved emitter (§spec:op-emission). The two fixed functions appear
in 29 transforms, 21 of which carry a table as well and are counted under
it. Nothing in the config refuses.

OCIO's own GPU path needs a sampled texture for 48 of the 159 — more than
this compiler does, which is none. Eight of those carry no `Lut1D` op at
all: OCIO's shader baked a closed-form `Exponent` or `ExponentWithLinear`
into a texture. Where OCIO's shaders sample, this compiler evaluates.

**Direction is a second axis.** Ops arrive both ways: 18 of 26 `Exponent`,
17 of 24 `ExponentWithLinear`, 12 of 24 `LogCamera`, and 2 of 4 `Log` are
inverse. `Matrix`, `Range`, and `Lut1D` never are, because OCIO folds or
pre-inverts them before the compiler sees them. The inverse of a
closed-form op is closed-form, so the six emitters are about eleven code
paths — cheap, but not free.

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
plus whatever attribute distinguishes one behaviour of it from another. It
kept the census honest while one style emitted and the other refused, and
it is what an OCIO release adding a third style will land against.

Both styles are closed-form — neither needs a 3D LUT in OCIO's own GPU
partition, which is the measure of whether an op can be expressed
analytically at all. What separated them was scope and risk, by an order of
magnitude in both. `REC2100_SURROUND` (5 transforms) scales all three
channels by a power of their luminance, and was the only thing standing
between this compiler and HLG display output, so it went first.
`ACES_OUTPUT_TRANSFORM_20` (24 transforms) is *the look* — the most visible
math in any pipeline that runs it — which made it the worst candidate for
an early port and the best candidate for a careful one. It went last.

**Refusal is by op, not by category.** Every transform in the pinned config
compiles, so the boundary is not visible from inside it. It is still stated
as an op set rather than as a category, because that is what makes it
predictable for the arbitrary config a caller hands over
(§spec:problem-statement): the answer names the op that blocks them, and
nothing is refused for the kind of transform it is. `ocio2onnx census`
reports the split for any config against the op set the compiler actually
implements rather than a second list beside it; `tools/census.py` is a shim
over the same code.

## How ops emit §spec:op-emission
*Status: complete*

Every op emits as ONNX arithmetic over parameters read from OCIO's
transform introspection. `Matrix` is a 1×1 convolution, `Range` a clamp
and an affine, the exponential and logarithmic ops elementwise chains.
ONNX has no per-element branching, so an op with a breakpoint evaluates
both sides and selects; that costs arithmetic, not correctness.

**Two ops' arithmetic crosses channels.** `REC2100_SURROUND` scales all
three by a single luminance per pixel, so a per-channel reading of it
agrees on every neutral pixel and disagrees on every coloured one — a
misreading nothing but an oracle sampling coloured pixels would catch
(§spec:verification). Its fold onto `|Y|` and its luminance floor are read
off OCIO's own renderer rather than chosen, as `Log`'s floor is: the floor
is `1e-4` forward, and that value's image under the forward op inverse,
because the inverse clamps in the forward's output domain.

**`ACES_OUTPUT_TRANSFORM_20` crosses them further: the whole op runs inside
a colour appearance model.** Every intermediate is one scalar per pixel —
lightness, colourfulness, hue — rather than one per channel, so the graph
takes the channels apart after the second matrix and puts them back
together before the third. Its nine parameters expand at compile time into
four matrices, a tone curve, two compressions, and three hue-indexed
tables; only the arithmetic reaches the graph.

That expansion is read off OCIO's renderer rather than off the published
ACES 2.0 description, which differs from it exactly where a difference is
expensive to notice: which clamp applies in which domain, where the
absolute values sit, and — the one the oracle caught — which `std::max`
swallows a NaN. OCIO's tone scale reaches `std::max(0.f, NaN)` for every
pixel whose achromatic response is negative and answers black; ONNX `Max`
propagates the NaN, so the comparison is emitted rather than the operator,
and every `std::max` in the op keeps OCIO's argument order.

Two of the three tables cost a search per hue to build — a bisection
against the display gamut hull, a nested one for the hull's exponent, a
third for the reach gamut — so they are read off OCIO's own shader
description rather than rebuilt. Rebuilding them would put a search
tolerance between this compiler and its oracle for no gain, which is the
reasoning that makes OCIO the oracle at all. The hue table is derived,
because OCIO publishes it only inside generated shader text, which is not
an interface; it costs six bisections rather than seven hundred, and a
test holds the derivation against what that text declares.

*Rejected*: the inverse. It exists in OCIO and in no transform of the
pinned config, and it is not the forward path run backwards — its gamut
compression solves an approximation twice, the first pass only to estimate
the lightness the second needs. An inverse request is refused by name.

`Lut1D` is the exception among the rest, because most of them are
half-domain: 37 of the 40 in the config hold 65536 entries indexed by the
bit pattern of the input rounded to float16, not by a normalized
coordinate. Standard ONNX has no bit-reinterpreting cast, so the index has
to be arrived at some other way.

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

No `Lut1D` in the pinned config now arrives inverse: `OPTIMIZATION_LUT_INV_FAST`
is set, which §spec:verification requires for a different reason, and it
makes OCIO rewrite every invertible one into a forward half-domain table
before the compiler sees it. The compile-time inversion below therefore
serves op lists that reach the compiler still inverse, and the paragraphs
that follow record why its grid is the one it is.

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
legitimately produces outputs around 1e8.

**The compiler and the oracle read one op list.** OCIO reports a
processor's ops as the config declares them and its renderer runs an
optimized rewrite of that list. The two are the same transform and not the
same arithmetic, so the compiler walks the rewrite: one optimization level,
resolved once where a request binds to a processor, and read from there by
the compiler, the census, and the lattice alike. Compiling one list and
measuring against another leaves a residual that belongs to neither
(§spec:op-coverage).

**That level runs with fast math off.** OCIO's default CPU optimization
includes `OPTIMIZATION_FAST_LOG_EXP_POW`, an approximate `pow`, `log`, and
`exp` whose error exceeds this tolerance. Left on, it measures a math
library rather than the compiler's reading of the config: 110 of the 111
closed-form transforms verify against OCIO's fast path, and all 111 verify
against its accurate one. Which optimization flags are set is a correctness
decision, not a tuning knob — the reference has to be OCIO's accurate
arithmetic, or a compiler error and an oracle error are indistinguishable.

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
a named op. All 159 verify.

### What counts as evidence §spec:evidence-floor

Every sample is evidence: one is held against the tolerance where both sides
are finite, and matched against its *class* otherwise, never dropped. A
non-finite reference value carries no magnitude to measure a tolerance
against, so what is asserted there is the kind of answer — finite, `+inf`,
`-inf`, and NaN are four distinct answers, and the graph returns the one the
reference returned. Twelve of the pinned config's 159 transforms reach that
path, at up to 36 samples of a lattice of about 1250, and every one of those
387 samples agrees on its class.

*Why not drop them*: a direction inverted at overflow, the reference falling
to `-inf` where the graph climbs to `+inf`, is exactly the misreading this
section exists to catch. Dropping the sample reads that as agreement.

Agreement also needs at least one finite sample compared. A transform whose
reference is non-finite across the whole lattice otherwise compares nothing
and reports agreement — a graph certified on no evidence, which is worse than
one that refuses, because "verified" is a claim the consumer acts on. No
transform in the pinned config is in that state; a config the compiler is
handed is arbitrary (§spec:problem-statement), and a NaN coefficient in a
config-declared matrix reaches it in one step.

*Why a floor of one sample rather than a fraction of the lattice*: a
percentage would need a constant this document cannot derive from anything,
and would refuse a legitimately overflow-heavy transform added later. Once
every sample is checked one way or the other, the only quantity left worth
asserting is that the finite kind occurred at all.

## Dynamic properties §spec:dynamic-properties
*Status: complete*

OCIO exposes seven dynamic property types — `EXPOSURE`, `CONTRAST`,
`GAMMA`, and four `GRADING_*` families — plus ASC CDL as slope, offset,
power, and saturation. A transform carrying them compiles to a graph with
those values as **additional inputs**, so a consumer varies them per frame
without recompiling.

This is what a baked LUT cannot do, and it is the reason the emitted
artifact is a graph rather than a table: a live grade is the same compiled
transform with its parameters left free.

The scalar and per-channel properties emit. `ExposureContrast` carries
`EXPOSURE`, `CONTRAST`, and `GAMMA`, spelled as OCIO's own property types
spell them; a CDL carries `CDL_SLOPE`, `CDL_OFFSET`, `CDL_POWER`, and
`CDL_SATURATION`.

**The four curve-shaped `GRADING_*` families are deferred, and the deferral is
the shipped behaviour.** Their parameters are control points rather than
values, so they arrive with a tensor-valued input or not at all — a different
interface from the scalars above. The trigger is the first consumer that needs
curve-based grading rather than CDL. Until it fires, each of the four refuses
by name, by the same mechanism as any unimplemented op, at the command line and
in the census (§spec:op-coverage). Pinned by test: a deferral whose edge a
caller cannot see is indistinguishable from a wrong answer. `GradingHueCurve`
arrived in OCIO 2.5 and was refused without a line of this compiler changing,
which is what deriving the refusal from the emitter registry buys.

**The value decides what is live; OCIO's flag decides only what the value
cannot.** OCIO's declaration governs its own run-time plumbing — one dynamic
property of each type per processor, extras dropped with a warning — and a
graph has neither the limit nor the need for it. An input left unbound reads
the default the artifact carries, so a knob nobody turns costs a consumer
nothing, and each op keeps its own default rather than one of them being
discarded. A CDL is the case that settles it: OCIO declares no dynamic property
for one at all, and a CDL is the grade this section exists for. So a grade
sitting off its identity proves its own op exists and emits its knobs whatever
OCIO declares.

A grade sitting *at* its identity is the exception, because it is not
distinguishable from an absent op and OCIO removes it. There the declaration is
the only signal left, and OCIO preserves a declared-dynamic op through that
removal — so an identity grade stays live exactly when its author said it was
meant to be driven. Catching the undeclared case would mean carrying every
identity matrix and range in every config into the graph, which is a wider
price than the knob is worth.

**Liveness is decided before the compiler sees the op, so the optimizer is part
of this section.** OCIO reports the ops it would run, not the ops the config
declares, and its rewrites can spend a live parameter before there is anything
to attach an input to: at a unit power a CDL becomes a matrix and the grade
arrives as coefficients. That is the primary — slope, offset, and saturation at
a unit power — which is the commonest CDL there is and precisely the grade a
consumer means to drive. `OPTIMIZATION_SIMPLIFY_OPS` is therefore cleared. It
costs nothing to clear, the op count across the pinned config being unchanged,
and the rewrite it suppresses is not value-preserving in any case: OCIO applies
it at `OPTIMIZATION_LOSSLESS`, where it drops a clamping CDL's output clamp and
answers 1.111 for an input the op answers 1.0 on. Which flags are set is a
correctness decision rather than a tuning knob, the same reading as
§spec:verification's, and the set is pinned by test.

**The graph inverts what OCIO pre-inverts.** An inverse op's parameters are
still the forward grade OCIO reports — the knob is the CDL's slope, not the
reciprocal an inverse multiplies by — so the reciprocals a static emitter
would fold at compile time become arithmetic in the graph. So do the
reference's guards, which are not tidy and are load bearing: OCIO floors a
CDL parameter at 0.01 before reciprocating it, so a crushed channel inverts
to 100 rather than to an infinity, and it floors `contrast * gamma` at 0.001
— *before* the reciprocal for the pivoted exposure styles and *after* it for
the logarithmic one, which sends a negative product to 1000 in the first case
and to 0.001 in the second.

**The reference bounds what a sweep can establish.** OCIO's
`ExposureContrast` renderer uses an approximate power that
`OPTIMIZATION_FAST_LOG_EXP_POW` does not govern, so §spec:verification's
remedy of clearing a flag does not reach it. Its base error is around 1e-5
relative, which two things multiply: the inverse's exponent of
`1 / (contrast * gamma)`, past the tolerance once the product falls below
about 0.09; and cancellation downstream, where a CDL saturation of 1.3 after
the op turns 1.3e-5 into 5e-4 on one sample of the lattice. Both are the
reference's residual rather than the graph's, and both are pinned by tests so
that an OCIO release which removes them is visible rather than silent.

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
