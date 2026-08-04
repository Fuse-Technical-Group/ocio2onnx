"""Throughput: OCIO's own shader against the emitted graph on a GPU.

The companion measurement to `shader`, which asks how far each lands from the
reference. This one asks what each costs per frame, and it is a separate module
because it is a separate claim with its own ways of being wrong.

Three rules keep the two sides comparable, and every one of them was a way to
publish a wrong number:

- **Nothing crosses PCIe inside the timing.** The image is resident before the
  clock starts — a texture on one side, a bound device tensor on the other —
  because a pointwise transform moves less arithmetic than it would move bytes,
  and a benchmark that included the bus would report the bus.
- **Neither side may batch what the other must not.** Each frame is finished
  before the next is timed. Queuing a hundred draws behind one fence measures a
  depth of pipeline the other side is never given.
- **A time is only reported for work that was right.** Both sides are held
  against OCIO's CPU processor before their number is kept, so a kernel that
  silently produced zeroes cannot post the winning score.

What it does not measure: the layout each side wants (planar float32 against
interleaved RGBA), and the conversion between them. Neither pays for it here,
and a pipeline that needs one will pay for it once, outside the transform.

**Pin the clocks before believing any of it.** A GPU left to its own devices
boosts on whatever thermal and power headroom it happens to have, so the same
frame times a few percent apart between runs and further apart between a cold
card and a warm one. `PINNING` is the fix, and every run prints the clocks it
actually observed so a number carries the conditions it was taken under.
"""

from __future__ import annotations

import dataclasses
import subprocess
import time

import numpy as np
import onnx
import PyOpenColorIO as OCIO

from ocio2onnx.addressing import OPTIMIZATION_FLAGS
from ocio2onnx.builder import CHANNELS, INPUT
from ocio2onnx.shader import describe

__all__ = [
    "PINNING",
    "BenchError",
    "Timing",
    "frame",
    "gpu_state",
    "sizes",
    "time_graph",
    "time_shader",
]

#: How to hold the GPU still, and how to give it back. Pick a clock the card
#: can hold indefinitely rather than its boost ceiling: a boost figure decays
#: as the card warms, which is the variance this is meant to remove. Reading
#: `nvidia-smi -q -d SUPPORTED_CLOCKS` names the ones on offer.
#:
#: A card may settle below the memory clock it accepted — ECC costs some of it
#: — so the figure `gpu_state` reads back is the one a run was taken at, not
#: the one that was asked for. Which is why every run prints it.
PINNING = (
    "nvidia-smi -lgc 1500,1500 && nvidia-smi -lmc 8001,8001   (release: "
    "nvidia-smi -rgc && nvidia-smi -rmc)"
)

#: Enough frames that a single scheduling hiccup cannot carry the number, and
#: few enough that a 4K sweep over several providers still finishes.
ITERATIONS = 200
WARMUP = 20

#: How far a timed frame may sit from OCIO's CPU processor before its time is
#: thrown away. Looser than §spec:verification's tolerance on purpose: this is
#: a check that the work happened, and `shader`/`verify` are where accuracy is
#: decided. Tightening it here would silently turn a benchmark into an oracle.
RAN_AT_ALL = 1e-3

#: The two sizes a display pipeline actually runs at.
DEFAULT_SIZES = ((1920, 1080), (3840, 2160))


class BenchError(RuntimeError):
    """A measurement that cannot honestly be reported."""


@dataclasses.dataclass(frozen=True)
class Timing:
    """What one candidate cost at one size."""

    label: str
    width: int
    height: int
    milliseconds: float
    setup_milliseconds: float

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def rate(self) -> float:
        """Megapixels per second."""
        return self.megapixels / self.milliseconds * 1e3

    def __str__(self) -> str:
        return (
            f"{self.label:<22}{self.milliseconds:9.3f} ms{self.rate:11.1f} Mpix/s"
            f"   (setup {self.setup_milliseconds:.0f} ms)"
        )


def gpu_state() -> str:
    """The card and the clocks it is actually running at.

    Printed with every run rather than assumed, because a throughput figure
    taken on a boosting GPU is not a figure anyone can reproduce — and the
    difference does not announce itself in the numbers.
    """
    query = "--query-gpu=name,clocks.sm,clocks.mem"
    try:
        answer = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown — nvidia-smi did not answer, so nothing here is pinned"
    lines = [line for line in answer.stdout.splitlines() if line.strip()]
    if not lines:
        return "unknown — nvidia-smi named no device"
    return ", ".join(field.strip() for field in lines[0].split(","))


def sizes(requested: list[str] | None) -> tuple[tuple[int, int], ...]:
    """Parse ``WxH`` arguments, or hand back the defaults."""
    if not requested:
        return DEFAULT_SIZES
    parsed = []
    for text in requested:
        width, _, height = text.lower().partition("x")
        if not height.isdigit() or not width.isdigit():
            raise ValueError(f"{text!r} is not a size; write it as 1920x1080")
        parsed.append((int(width), int(height)))
    return tuple(parsed)


def frame(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """One image, planar for the graph and interleaved RGBA for the shader.

    The same pixels both ways, from a fixed seed, so a run is repeatable and
    the two candidates are not quietly given different work.
    """
    generator = np.random.default_rng(0x0C10)
    planar = generator.uniform(0.0, 1.0, (1, CHANNELS, height, width))
    planar = planar.astype(np.float32)
    interleaved = np.ones((height, width, 4), dtype=np.float32)
    interleaved[..., :CHANNELS] = planar[0].transpose(1, 2, 0)
    return planar, np.ascontiguousarray(interleaved)


def _expected(processor: OCIO.Processor, pixels: np.ndarray) -> np.ndarray:
    """What OCIO's CPU processor makes of an ``(N, 3)`` strip."""
    want = np.ascontiguousarray(pixels, dtype=np.float32)
    processor.getOptimizedCPUProcessor(OPTIMIZATION_FLAGS).applyRGB(want)
    return want


def _check(label: str, got: np.ndarray, want: np.ndarray) -> None:
    """Refuse to report a time for a frame that is not the transform."""
    if not np.allclose(got, want, atol=RAN_AT_ALL, rtol=RAN_AT_ALL):
        raise BenchError(
            f"{label} was timed but did not transform the image "
            f"(worst {np.abs(got - want).max():g}), so its time means nothing"
        )


def time_shader(
    processor: OCIO.Processor,
    width: int,
    height: int,
    pixels: np.ndarray,
    iterations: int = ITERATIONS,
    warmup: int = WARMUP,
) -> Timing:
    """Milliseconds per frame for OCIO's own generated GLSL."""
    from ocio2onnx import _gl

    started = time.perf_counter()
    desc = describe(processor)
    seconds, strip = _gl.time_draws(desc, pixels, width, height, iterations, warmup)
    setup = (time.perf_counter() - started) * 1e3 - seconds * iterations * 1e3

    edge = strip.shape[0]
    _check(
        "OCIO's shader",
        strip[:, :CHANNELS],
        _expected(processor, pixels[0, :edge, :CHANNELS]),
    )
    return Timing("OCIO GLSL", width, height, seconds * 1e3, max(setup, 0.0))


def time_graph(
    model: onnx.ModelProto,
    processor: OCIO.Processor,
    provider: str,
    width: int,
    height: int,
    planar: np.ndarray,
    iterations: int = ITERATIONS,
    warmup: int = WARMUP,
) -> Timing:
    """Milliseconds per frame for the emitted graph on one execution provider.

    The setup figure is the honest cost of the first frame — for TensorRT it is
    an engine built for this shape, which a consumer pays again at the next one.
    """
    import onnxruntime as ort

    from ocio2onnx.oracle import DEFAULT_PROVIDER, ProviderError, resolve_provider

    wanted = resolve_provider(provider)
    if wanted == DEFAULT_PROVIDER:
        raise BenchError("there is nothing to compare on the CPU; name a GPU")

    started = time.perf_counter()
    session = ort.InferenceSession(model.SerializeToString(), providers=[wanted])
    if wanted not in session.get_providers():
        raise ProviderError(f"{wanted} did not load")

    device = ort.OrtValue.ortvalue_from_numpy(planar, "cuda", 0)
    binding = session.io_binding()
    binding.bind_ortvalue_input(INPUT, device)
    binding.bind_output(session.get_outputs()[0].name, "cuda", 0)

    for _ in range(warmup):
        session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    setup = (time.perf_counter() - started) * 1e3

    started = time.perf_counter()
    for _ in range(iterations):
        session.run_with_iobinding(binding)
    binding.synchronize_outputs()
    seconds = (time.perf_counter() - started) / iterations

    edge = min(width, 64)
    got = binding.copy_outputs_to_cpu()[0][0, :, 0, :edge].transpose(1, 0)
    want = _expected(processor, planar[0, :, 0, :edge].transpose(1, 0))
    _check(wanted, got, want)

    label = wanted.replace("ExecutionProvider", "")
    return Timing(f"ONNX / {label}", width, height, seconds * 1e3, setup)
