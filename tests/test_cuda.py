"""The fused CUDA kernel artifact (§spec:cuda-kernel).

The ONNX graph is the portable artifact; the kernel exists for the one
transform class that defeats graph executors — the ACES 2.0 display renders,
whose fusion-hostile op chain costs a memory pass per segment under TensorRT.
The kernel is transpiled from the shader OCIO's own GPU renderer emits, so
the tests hold three claims: the source is self-contained, the transpiler
refuses what it cannot carry, and the compiled kernel agrees with the same
oracle every graph answers to — at the GPU tolerance, which is the band
OCIO's own GPU renderer occupies against its CPU renderer.
"""

import re

import numpy as np
import PyOpenColorIO as OCIO
import pytest

from ocio2onnx import cuda
from ocio2onnx.addressing import (
    OPTIMIZATION_FLAGS,
    Resolved,
    resolve_display_view,
)

#: The transform the kernel exists for: the heaviest display render in the
#: pinned config, and the one §spec:cuda-kernel names.
DISPLAY = "sRGB - Display"
ACES2_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"

#: A closed-form display render, so the transpiler is held on a shader with
#: no textures at all.
PLAIN_VIEW = "Un-tone-mapped"


@pytest.fixture(scope="module")
def aces2_sdr(config, config_uri):
    return resolve_display_view(
        config, DISPLAY, ACES2_VIEW, src="ACEScg", uri=config_uri
    )


@pytest.fixture(scope="module")
def untonemapped(config, config_uri):
    return resolve_display_view(
        config, DISPLAY, PLAIN_VIEW, src="ACEScg", uri=config_uri
    )


@pytest.fixture(scope="module")
def source(aces2_sdr):
    return cuda.kernel_source(aces2_sdr)


def bare(config, config_uri, transform, label="bare"):
    """A hand-built transform resolved the way ``compile_bare`` does."""
    return Resolved(
        processor=config.getProcessor(transform).getOptimizedProcessor(
            OPTIMIZATION_FLAGS
        ),
        config_name="bare",
        config_uri=config_uri,
        endpoints=label,
    )


class TestSource:
    def test_self_contained(self, source):
        """No include, no sampler, no GL call survives: a consumer compiles
        the file with NVRTC and nothing else."""
        assert "#include" not in source
        assert "uniform" not in source
        assert "texture(" not in source
        assert 'extern "C"' in source

    def test_both_entry_points(self, source):
        """Planar RGB at float32 and float16 (§spec:cuda-kernel). Alpha is
        absent, not passed through (§spec:emitted-graph)."""
        assert "apply_f32" in source
        assert "apply_f16" in source
        assert "alpha" not in source.lower()

    def test_tables_are_embedded(self, source):
        """Both published textures land as global arrays — global rather than
        constant memory, which serializes on divergent indices."""
        assert "__device__ const float ocio_reach_m_table_0_data[363]" in source
        assert "__device__ const float ocio_gamut_cusp_table_0_data[1089]" in source
        assert "__constant__" not in source

    def test_metadata_header(self, source, config_uri):
        """The artifact says what produced it, like the graph does."""
        assert config_uri in source
        assert f"{DISPLAY} / {ACES2_VIEW}" in source

    def test_float_literals_are_float(self, source):
        """An unsuffixed literal is a double, and one double in a chain drops
        the whole expression to double-precision arithmetic."""
        unsuffixed = re.compile(
            r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
        )
        for line in source.splitlines():
            code = line.split("//")[0]
            assert not unsuffixed.search(code), line

    def test_closed_form_shader_transpiles(self, untonemapped):
        """A shader with no tables at all comes through the same path."""
        source = cuda.kernel_source(untonemapped)
        assert 'extern "C"' in source
        assert "apply_f32" in source


class TestRefusal:
    def test_dynamic_property_refused(self, config, config_uri):
        """A dynamic property is a uniform, and the kernel bakes everything;
        the refusal names the mechanism rather than emitting a stale value."""
        transform = OCIO.ExposureContrastTransform()
        transform.makeExposureDynamic()
        with pytest.raises(cuda.UnsupportedShaderError, match="uniform"):
            cuda.kernel_source(bare(config, config_uri, transform))

    def test_interpolated_texture_refused(self, config, config_uri):
        """A linearly-interpolated texture has no transpiled equivalent yet;
        the refusal names the texture. The table is non-identity because an
        identity ``Lut1D`` optimizes away before any texture is published."""
        transform = OCIO.Lut1DTransform(length=8)
        for i in range(8):
            value = (i / 7) ** 0.5
            transform.setValue(i, value, value, value)
        with pytest.raises(cuda.UnsupportedShaderError, match="ocio_lut1d"):
            cuda.kernel_source(bare(config, config_uri, transform))


@pytest.fixture(scope="module")
def cuda_runtime():
    """Skip where the NVRTC library or a CUDA device is absent, so the
    execution tests fail only on answers, never on machinery."""
    pytest.importorskip("cuda.bindings")
    try:
        cuda.device_arch()
    except Exception as exc:  # any miss means "no GPU here", not a failure
        pytest.skip(f"no usable CUDA runtime: {exc}")


@pytest.mark.usefixtures("cuda_runtime")
class TestKernel:
    def test_kernel_agrees_with_oracle(self, aces2_sdr):
        result = cuda.verify(aces2_sdr)
        assert result.ok, str(result)

    def test_closed_form_kernel_agrees(self, untonemapped):
        result = cuda.verify(untonemapped)
        assert result.ok, str(result)

    def test_f16_entry_point_tracks_f32(self, aces2_sdr, source):
        """The half kernel is the same arithmetic behind quantized edges, so
        it agrees with the float kernel to half precision, not to the oracle
        tolerance (§spec:cuda-kernel)."""
        from ocio2onnx.oracle import lattice

        samples = lattice(aces2_sdr.processor)
        f32 = cuda.run_kernel(source, samples)
        f16 = cuda.run_kernel(source, samples, precision="f16")
        assert np.isfinite(f16).all()
        assert np.abs(f16 - f32).max() < 1e-2
