# ocio2onnx — Roadmap

Derived from SPEC.md. Every item traces to a spec gap. Sections are
in build-dependency order. Completed work is removed; presence here
means the work is not done.

## Live parameters §road:live-parameters

The scalar and per-channel properties compile to graph inputs
(§spec:dynamic-properties), so a consumer varies a grade per frame without
recompiling. What is left is the shape those inputs cannot take.

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
