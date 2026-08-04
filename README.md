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

### Against OCIO's own shader

OCIO's GPU path is the incumbent, and it is a fair question whether
compiling to ONNX costs accuracy against it. It does not — it gains some.
Both are scored against the same CPU processor, over the same lattice, at
the same tolerance:

```sh
ocio2onnx shader [--config URI]      # needs the shader extra and a GL 4.0 driver
```

```text
               this compiler   OCIO's shader
verified:                159             159
of:                      159             159
margin:               0.0639           0.314
```

Both agree with OCIO's CPU processor everywhere, so the pass counts do not
separate them. `margin` does: every deviation stated as a fraction of the
tolerance bound that governed it, which is the only figure comparable
across two candidates — `max_abs` and `max_rel` are each dominated by the
samples the *other* bound decided. The graph's worst sample anywhere in the
config uses 6.4% of its tolerance; OCIO's own shader uses 31.4%.

The reason is the one §spec:op-coverage names. OCIO's shader sampled a
texture for **48 of the 159** transforms, **8** of which carry no `Lut1D`
op to account for it: they are the ACES 2.0 output transforms, and the two
tables are the hue and gamut-cusp data its fixed function needs, which
OCIO's GPU path cannot evaluate in closed form. This compiler samples for
none of them. Where OCIO's shaders sample, this evaluates.

This is an accuracy comparison only. Throughput is the next section.

### What it costs per frame

```sh
nvidia-smi -lgc 1500,1500 && nvidia-smi -lmc 8001,8001   # pin the clocks first
ocio2onnx bench --display "sRGB - Display" --view "ACES 2.0 - SDR 100 nits (Rec.709)"
nvidia-smi -rgc && nvidia-smi -rmc                       # give them back
```

Without pinned clocks a GPU boosts on whatever headroom it has and the same
frame times differently cold and warm, so `bench` prints the clocks it
actually observed with every run. On an RTX A6000 at 1500 MHz, the ACES 2.0
output transform:

```text
                     1920x1080      3840x2160        setup
OCIO GLSL              0.157 ms       0.538 ms       ~65-120 ms
ONNX / TensorRT        1.162 ms       4.381 ms       ~2,200-4,100 ms
ONNX / CUDA           20.750 ms      77.539 ms       ~550-1,800 ms
```

**OCIO's shader is about 8x faster than the graph under TensorRT, and
140x faster than the ONNX Runtime CUDA provider.** The CUDA provider
fuses nothing and pays for every node one at a time. What TensorRT is
paying for is worth stating precisely, because two obvious answers are
both wrong: it is not framework overhead, and it is not register
pressure.

A near-trivial transform — one `Matrix` — runs 0.430 ms against 0.594 ms
at 4K, so the floor either side of the fence is within 1.4x. The rest is
this transform's arithmetic, and Nsight counts it on both sides: per 4K
frame the graph executes **1.62 billion** warp instructions against the
shader's **158 million**, which is 10x the instructions for 8x the time.
Nothing is being executed badly. The graph issues at a *higher* rate than
the shader manages, no kernel spills a register, and occupancy runs
49-87%. There is simply ten times as much of it.

Those two counts are the only figures compared across the two profilers,
and deliberately: each clamps the clock itself — 1.41 GHz for the
shader's capture against 1.80 GHz for the graph's — so their rates and
durations do not sit on the same axis, while a count of instructions does
not care what clock retired it. Pinning does not reach them either. The
frame times above are `bench`'s, where both sides run in one harness at
one pinned clock.

Where the extra goes is legible on both sides. The shader's throughput is
dominated by its special-function pipe and its texture cache answers 87%
of the reads it makes, so the two tables OCIO bakes cost it almost
nothing — and evaluating that same appearance model in closed form
instead (§spec:op-coverage) is one compute-bound kernel of 2.48 ms here.
TensorRT compiles the graph to one `Myelin` layer, but a layer is not a
launch: it is six kernels, and the other five run at 82-93% of memory
bandwidth handing a 99.5 MB intermediate along — traffic a fragment
shader never generates because its intermediates never leave registers.

Getting to six kernels is why `Matrix` emits nine multiplies over the
channels rather than the 1x1 `Conv` that is the same arithmetic in one
node (§spec:op-emission). Emitted as a convolution the graph was **2.256
ms and 8.870 ms** in ten layers — six `CaskConvolution` alternating with
four fused regions — and a trace names the cost more bluntly than "a
fusion barrier" does: that frame carried two whole-tensor permutation
kernels which together outweighed all seven of its pointwise kernels
combined. A convolution wants NHWC and the rest of the graph is NCHW, so
TensorRT converted the image and converted it back; its build log names
`Reformat` 116 times for that spelling and never for this one. All of it
for a spelling that changes nothing about the result: on the CPU the two
are bit-identical. The ONNX Runtime CUDA provider is the one loser, about
13% slower on this transform and 2x on a bare matrix, because it fuses
neither and the pointwise spelling is more nodes to dispatch.

Setup is the other half of it: TensorRT builds an engine per shape, so a
consumer that resizes pays a few seconds again, against a millisecond to
link a shader. Dropping the convolutions took that from six seconds to
two, which is a smaller claim than it sounds — it is still a rebuild.

None of that undoes §spec:problem-statement — the graph exists to run where
no graphics context does, and it is more accurate than the shader by the
section above. But it is slower, and by how much is a measured number
rather than a hope.

### Where the graph runs

Both commands take `--provider` — `cpu`, `cuda`, `tensorrt`, `dml`, or an
onnxruntime provider's full name — and default to the CPU, which is the
arithmetic the reference itself runs on. Only the graph's side moves: the
reference stays OCIO's CPU processor and the tolerance stays what it is, so
a sweep that passes on the CPU and fails on an accelerator has measured
that runtime's fused multiplies and transcendentals rather than the
compiler's reading of the config. The claim this project makes is the CPU
one; the rest is a way to ask what a consumer's own runtime will do.

A provider the installed onnxruntime does not offer is refused with the
list it does, and one that loads but cannot run is refused naming itself,
so no session silently becomes a CPU session under a GPU's name.

That answers for the session, not for every node in it. onnxruntime
appends the CPU behind whatever it was asked for and places on it anything
the named provider will not take — sometimes deliberately, as with the
shape ops it keeps off an accelerator on purpose. `--strict-provider`
refuses that too. It is not the default because it is a stronger claim
than most callers want: on this config's heaviest transform TensorRT
passes it and the CUDA provider does not.

Running on CUDA needs `onnxruntime-gpu` in place of `onnxruntime`, plus a
CUDA runtime and cuDNN 9 its loader can find. The two packages install the
same module and cannot share an environment, so the extra is separate and
`uv run --no-sync` is what keeps a sync from putting the CPU build back.

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
