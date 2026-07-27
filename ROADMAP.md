# ocio2onnx — Roadmap

Derived from SPEC.md. Every item traces to a spec gap. Sections are
in build-dependency order. Completed work is removed; presence here
means the work is not done.

## Compiler core §road:compiler-core

Turn an OCIO transform into a verified ONNX graph for the ops that make up
the ACES configs (§spec:op-coverage). Nothing here interprets color: every
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
releases.

### Op emitters §road:op-emitters

Emit ONNX for the seven supported ops — `Matrix`, `Range`, `Exponent`,
`ExponentWithLinear`, `Log`, `LogCamera`, `Lut1D` — reading parameters
from the transform introspection API (§spec:op-coverage). `Matrix` emits
as a 1×1 convolution, `Lut1D` as a gather and lerp, the rest as
elementwise arithmetic. Depends on §road:transform-addressing.

### Named refusal §road:named-refusal

Enumerate a processor's ops before emitting and refuse any the compiler
does not implement, naming the op, the transform, and its endpoints
(§spec:op-coverage). `FixedFunction` is the only op this rejects in the
ACES configs today; the check is written against the supported set, so a
future OCIO op is refused rather than silently dropped. Depends on
§road:op-emitters.

### Oracle verification harness §road:oracle-harness

Property-test every emitted graph against OCIO's CPU processor over a
generated input lattice, including out-of-range, near-zero, and
per-op breakpoint values (§spec:verification). The harness enumerates the
pinned config exhaustively and asserts the partition: every color space
pair either verifies within tolerance or refuses by name. Depends on
§road:named-refusal.

**Verify:** Compile `Log3G10 REDWideGamutRGB` → `ACES2065-1` and confirm
the emitted graph agrees with the CPU processor across the lattice.
Compile a display view carrying `ACES_OUTPUT_TRANSFORM_20` and confirm it
refuses, naming the op. Run the harness across the whole pinned config and
confirm every pair lands in one bucket or the other, with none skipped.

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
