# ocio2onnx — Roadmap

Derived from SPEC.md. Every item traces to a spec gap. Sections are
in build-dependency order. Completed work is removed; presence here
means the work is not done.

## Oracle evidence §road:oracle-evidence

Close the gap between what the oracle compares and what it certifies
(§spec:evidence-floor). The pinned config does not exercise the blind spot:
across it no transform disagrees on the class of a non-finite sample, and
none is non-finite everywhere. What reaches it is a config a *user* supplies
(§spec:problem-statement) — a NaN coefficient in a config-declared matrix
gets there in one step, and the harness would report a graph certified on no
evidence. It is a guard on the arbitrary input, not protection for the work
below it.

### Non-finite agreement and the evidence floor §road:nonfinite-agreement

`oracle.compare` drops every sample where the reference is non-finite, then
reports agreement when that leaves nothing compared: a config-declared
matrix carrying NaN coefficients makes `ocio2onnx compile --verify` print
`verified: 0 of 0 samples outside tolerance` over a 1248-sample lattice.
Classify each non-finite sample as `+inf`, `-inf`, or NaN and require the
graph's class to match the reference's, and require at least one finite
comparison before a graph verifies (§spec:evidence-floor). Both the
`--verify` flag and the `verify` sweep read the same predicate, so the
partition over the pinned config is the regression test — 159 verified and
0 refused, unchanged, with the 12 overflowing transforms now agreeing on
their non-finite samples rather than skipping them.

## Live parameters §road:live-parameters

Compile a transform's dynamic properties to graph inputs so a consumer
varies a grade per frame without recompiling (§spec:dynamic-properties) —
the capability that distinguishes an emitted graph from a baked table.

### Scalar dynamic properties §road:scalar-dynamics

Emit `EXPOSURE`, `CONTRAST`, and `GAMMA`, plus ASC CDL's slope, offset,
power, and saturation, as named graph inputs with their OCIO defaults
recorded (§spec:dynamic-properties). A live parameter is verified by
sweeping it against the oracle (§spec:verification), not only at its
default.

### Curve-shaped grading properties §road:grading-curves

The four `GRADING_*` families, whose parameters are control points rather
than values (§spec:dynamic-properties). Deferred behind a named trigger:
the first consumer that needs curve-based grading rather than CDL. Until
then the compiler refuses them by the same mechanism as any unimplemented
op (§spec:op-coverage).

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
