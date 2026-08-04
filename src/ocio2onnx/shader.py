"""OCIO's own generated shader, held against the same oracle (§spec:op-coverage).

The compiler's claim is not only that its graph agrees with OCIO, but that it
agrees by evaluating what OCIO's shaders sample: the GPU path bakes a texture
for 48 of the pinned config's 159 transforms, eight of them closed-form ops
with no ``Lut1D`` to justify it. That is a statement about accuracy, and it is
worth measuring rather than asserting.

So this module runs OCIO's GLSL through a real driver and scores it against
``oracle.cpu_reference`` — the same reference, the same lattice, the same
tolerance the compiler is held to. Two candidates, one oracle, and the
difference between them is the answer.

It answers only that question. A shader is measured here for how far it lands
from the reference, never for how fast it runs: throughput is a different
benchmark with different traps, and mixing them would make this one unreadable.
"""

from __future__ import annotations

import numpy as np
import PyOpenColorIO as OCIO

from ocio2onnx.addressing import OPTIMIZATION_FLAGS
from ocio2onnx.builder import CHANNELS

__all__ = ["LANGUAGE_NAME", "ShaderError", "sampled_textures", "shader_reference"]

#: The language to generate. GLSL 4.0 has ``sampler1D`` and ``texelFetch``,
#: which is what lets a sample reach the shader as the value the lattice holds
#: rather than as whatever a filter made of it.
LANGUAGE = OCIO.GPU_LANGUAGE_GLSL_4_0

#: What to call it in a report, from OCIO rather than from a string here.
LANGUAGE_NAME = OCIO.GpuLanguageToString(LANGUAGE)

#: Our own identifiers, prefixed so they cannot collide with the ``ocio`` ones
#: the generated text declares.
SOURCE = "ocio2onnx_source"
RESULT = "ocio2onnx_result"

#: A single triangle covering the viewport, from no vertex data at all. The
#: alternative is a quad and a buffer, which is more objects to leak.
VERTEX_SHADER = """#version 400
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
"""

#: ``texelFetch`` rather than ``texture``: this asks for one texel by integer
#: index, so no filtering, wrapping, or normalized-coordinate rounding sits
#: between the lattice and the shader. A sample must reach OCIO's code as the
#: float the reference was given, or the comparison measures our plumbing.
FRAGMENT_SHADER = """#version 400
uniform sampler2D {source};
out vec4 {result};

{generated}

void main()
{{
    {result} = {entry}(texelFetch({source}, ivec2(gl_FragCoord.xy), 0));
}}
"""


class ShaderError(RuntimeError):
    """OCIO's shader could not be run, or could not be run honestly."""


def describe(processor: OCIO.Processor) -> OCIO.GpuShaderDesc:
    """OCIO's shader for ``processor``, at the oracle's optimization level.

    The CPU reference is taken at ``OPTIMIZATION_FLAGS`` (``oracle``), so the
    shader is too. Comparing a differently optimized pair would measure the
    optimizer.
    """
    desc = OCIO.GpuShaderDesc.CreateShaderDesc(language=LANGUAGE)
    processor.getOptimizedGPUProcessor(OPTIMIZATION_FLAGS).extractGpuShaderInfo(desc)
    return desc


def sampled_textures(processor: OCIO.Processor) -> int:
    """How many textures OCIO's shader wants bound for this transform.

    The count §spec:op-coverage reports, read off the shader itself. Zero means
    OCIO's GPU path evaluates this transform in closed form, as the compiler
    does for all of them.
    """
    desc = describe(processor)
    return len(list(desc.getTextures())) + len(list(desc.get3DTextures()))


def shader_reference(processor: OCIO.Processor, samples: np.ndarray) -> np.ndarray:
    """Run ``samples`` through OCIO's generated GLSL on a real driver.

    Returns the same shape it was given, so the result drops into
    ``oracle.compare`` beside the graph's without reshaping.
    """
    from ocio2onnx import _gl

    desc = describe(processor)
    bound = list(desc.getUniforms())
    if bound:
        # A uniform is a dynamic property, and GL leaves an unset one at zero.
        # Binding defaults here would be inventing values OCIO did not give us
        # and reporting the result as OCIO's; refusing says so instead.
        names = ", ".join(uniform.name for uniform in bound)
        raise ShaderError(
            f"the shader declares uniforms this harness cannot bind: {names}"
        )

    shape = np.shape(samples)
    planes = np.reshape(samples, (shape[0], CHANNELS, -1))
    if planes.shape[0] != 1:
        raise ShaderError(f"one image at a time, not {planes.shape[0]}")

    width = planes.shape[2]
    pixels = np.ones((width, 4), dtype=np.float32)
    pixels[:, :CHANNELS] = planes[0].transpose(1, 0)

    result = _gl.render(desc, pixels, width)
    return np.reshape(
        result[:, :CHANNELS].transpose(1, 0).reshape(1, CHANNELS, 1, -1), shape
    )
