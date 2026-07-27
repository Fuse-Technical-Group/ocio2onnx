# ocio2onnx — Roadmap

Derived from SPEC.md. Every item traces to a spec gap. Sections are
in build-dependency order. Completed work is removed; presence here
means the work is not done.

## Compiler core §road:compiler-core

Turn an OCIO transform into a verified ONNX graph for the closed-form ops,
which reach 111 of the 159 transforms in the ACES configs with no lookup
table at all (§spec:op-coverage). Nothing here interprets color: every
coefficient is read off OCIO's own transform introspection, and the
oracle decides whether the reading was right.

### Transform addressing §road:transform-addressing

Resolve a compile request — a config plus a source and target color space,
or a display and view — to an OCIO processor, and reject an unresolvable
name against the config that was actually loaded
(§spec:emitted-graph). Configs address by built-in URI
(`ocio://studio-config-…`) or by file path; the resolved built-in name is
recorded in the emitted graph's metadata so an artifact says which
database produced it, never `ocio://default`, which moves between
releases. The graph's interface is three channels with spatial dimensions
free (§spec:emitted-graph).

### Oracle verification harness §road:oracle-harness

Generate the input lattice — out-of-range, near-zero, and per-op
breakpoint values — and compare an emitted graph against OCIO's CPU
processor within tolerance (§spec:verification). Built ahead of the
emitters rather than after them: it is the failing test each emitter is
written against, and a misread parameter has no other way of surfacing.
Depends on §road:transform-addressing, and is exercised first on `Matrix`,
the one op whose correctness is legible by inspection.

### Closed-form op emitters §road:closed-form-emitters

Emit ONNX for `Matrix`, `Range`, `Exponent`, `ExponentWithLinear`, `Log`,
and `LogCamera` in both directions — roughly eleven code paths, since most
of these ops arrive inverse (§spec:op-emission). Emitted at float32, which
the graph declares (§spec:precision). Unlocks 111 transforms on its own.
Depends on §road:oracle-harness.

### Named refusal §road:named-refusal

Enumerate a processor's ops before emitting and refuse any the compiler
does not implement, naming the op, the transform, and its endpoints
(§spec:op-coverage). The check is written against the supported set, so a
future OCIO op is refused rather than silently dropped. Depends on
§road:closed-form-emitters.

**Verify:** Compile `Log3G10 REDWideGamutRGB` → `ACES2065-1` and confirm
the emitted graph agrees with the CPU processor across the lattice.
Compile a display view carrying `ACES_OUTPUT_TRANSFORM_20` and confirm it
refuses, naming the op. Run the harness across the whole pinned config and
confirm all 111 closed-form transforms verify, that the rest refuse by
name, and that none are skipped.

## Sampled lookups §road:sampled-luts

The 20 transforms carrying a `Lut1D` (§spec:op-coverage). Split from
§road:compiler-core because the closed-form set needs no table machinery
at all, and because this is the most involved emitter in the project
(§spec:op-emission). Each workstream adds coverage the harness confirms
independently.

### Uniform 1D LUTs §road:uniform-lut

Emit a forward `Lut1D` over a uniform domain as a gather and lerp,
unblocking ACEScc, CanonLog2, and CanonLog3 (§spec:op-emission). The
smallest of the three, and the one that settles how a table reaches the
graph as an initializer. Depends on §road:oracle-harness.

### Half-domain 1D LUTs §road:half-domain-lut

Emit a `Lut1D` over a half domain by reconstructing the float16 index
arithmetically, after OCIO's own GLSL, unblocking the PQ and ST2084
displays, ADX10, ADX16, and Apple Log (§spec:op-emission). 34 of the 40
`Lut1D` ops in the pinned config take this path. Depends on
§road:uniform-lut.

### Inverse 1D LUTs §road:inverse-lut

Invert a monotonic `Lut1D` at compile time onto a uniform domain and emit
the result as an ordinary table (§spec:op-emission), unblocking the 8
transforms that carry one. Whether the resampling holds tolerance is the
harness's finding rather than an assumption. Depends on
§road:half-domain-lut.

**Verify:** Compile `ACES2065-1` → `Rec.2100-PQ - Display` and confirm the
graph agrees with the CPU processor across the lattice, including inputs
falling between two adjacent half slots. Confirm the census partition now
places all 131 non-`FixedFunction` transforms in the verified bucket.

## Display rendering ops §road:display-rendering

The two `FixedFunction` styles §road:named-refusal rejects. Both are
closed-form — neither needs a 3D LUT in OCIO's own partition — so this is
sequencing, not a capability gap (§spec:op-coverage). They are separated
because their cost and their risk differ by an order of magnitude.

### Rec.2100 surround §road:rec2100-surround

Emit `REC2100_SURROUND`, a surround/system-gamma adjustment, unblocking
the 5 transforms that reach an HLG display (§spec:op-coverage). Small, and
the only op between this compiler and HLG output — worth taking ahead of
any scene-referred work, since a consumer wanting HLG today is refused by
the cheap op rather than the expensive one. Depends on
§road:oracle-harness, which is what makes a rendering op safe to add at
all.

### ACES output transform §road:aces-output-transform

Emit `ACES_OUTPUT_TRANSFORM_20` — tone scale, chroma compression, gamut
mapping — unblocking the 24 display views (§spec:op-coverage). Deferred
behind a named trigger: the first consumer running a scene-referred
working space and needing a display rendering. *Why last*: this transform
is the image's look, so an error in it is both the most visible failure
available and the least likely to be caught by anything except the oracle
sweep. Depends on §road:oracle-harness.

## Live parameters §road:live-parameters

Compile a transform's dynamic properties to graph inputs so a consumer
varies a grade per frame without recompiling (§spec:dynamic-properties) —
the capability that distinguishes an emitted graph from a baked table.

### Scalar dynamic properties §road:scalar-dynamics

Emit `EXPOSURE`, `CONTRAST`, and `GAMMA`, plus ASC CDL's slope, offset,
power, and saturation, as named graph inputs with their OCIO defaults
recorded (§spec:dynamic-properties). Depends on §road:oracle-harness: a
live parameter is verified by sweeping it against the oracle, not only at
its default.

### Curve-shaped grading properties §road:grading-curves

The four `GRADING_*` families, whose parameters are control points rather
than values (§spec:dynamic-properties). Deferred behind a named trigger:
the first consumer that needs curve-based grading rather than CDL. Until
then the compiler refuses them by the same mechanism as any unimplemented
op (§road:named-refusal).

## Packaging §road:packaging

Publish the compiler as an installable package with the oracle harness
runnable by anyone (§spec:problem-statement). The audience is real-time
CUDA-resident video, which has had no answer here; making the artifact
easy to obtain and easy to check is most of what makes it useful to
someone who is not us.

### Distribution §road:distribution

Package for PyPI with `opencolorio` and `onnx` as the only runtime
dependencies, both permissively licensed, and pin the OCIO version range
the op coverage was measured against (§spec:op-coverage).

### Coverage report §road:coverage-report

Emit the op census and the verify/refuse partition as a generated artifact
per OCIO release, so a consumer reads what is supported from measurement
rather than from prose (§spec:op-coverage, §spec:verification). This is
also how an OCIO upgrade surfaces a new op: the partition changes.
