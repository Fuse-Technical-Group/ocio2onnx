"""The driver plumbing behind `shader` and `bench`, kept apart from the colour
question.

Nothing here knows what a transform is. It compiles the text OCIO generated,
binds the textures OCIO asked for, draws one fragment per sample, and reads the
result back at full float precision — so that its callers read as the questions
they are asking rather than as lists of GL calls.

The context is made once and kept. Creating one per transform would dominate a
159-transform sweep, and tearing one down mid-sweep would take the driver's
program cache with it. Everything else lives inside `scene`, which owns its
objects for exactly as long as one measurement needs them.
"""

from __future__ import annotations

import contextlib
import time

import glfw
import numpy as np
import PyOpenColorIO as OCIO
from OpenGL.error import GLError
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COMPILE_STATUS,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_NEAREST,
    GL_R32F,
    GL_RED,
    GL_RGB,
    GL_RGB32F,
    GL_RGBA,
    GL_RGBA32F,
    GL_TEXTURE0,
    GL_TEXTURE_1D,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glCheckFramebufferStatus,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDrawArrays,
    glFinish,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glPixelStorei,
    glReadPixels,
    glShaderSource,
    glTexImage1D,
    glTexImage2D,
    glTexParameteri,
    glUniform1i,
    glUseProgram,
    glViewport,
)

from ocio2onnx.shader import FRAGMENT_SHADER, RESULT, SOURCE, VERTEX_SHADER, ShaderError

#: The one context, made on first use.
_WINDOW = None

#: Texture unit zero carries the samples; OCIO's own textures start above it.
FIRST_OCIO_UNIT = 1


def context() -> None:
    """Make the offscreen context current, creating it once."""
    global _WINDOW
    if _WINDOW is not None:
        glfw.make_context_current(_WINDOW)
        return

    if not glfw.init():
        raise ShaderError("glfw could not initialise; there is no GL driver to ask")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1, 1, "ocio2onnx", None, None)
    if not window:
        glfw.terminate()
        raise ShaderError("no GL 4.0 core context; OCIO's GLSL 4.0 cannot be run here")
    _WINDOW = window
    glfw.make_context_current(_WINDOW)


def compile_stage(kind, source: str, what: str) -> int:
    """One shader stage, or the driver's own complaint about it."""
    stage = glCreateShader(kind)
    glShaderSource(stage, source)
    glCompileShader(stage)
    if not glGetShaderiv(stage, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(stage)
        glDeleteShader(stage)
        raise ShaderError(f"{what} did not compile: {_text(log)}")
    return stage


def link(fragment_source: str) -> int:
    """The linked program for one generated fragment shader."""
    stages = [
        compile_stage(GL_VERTEX_SHADER, VERTEX_SHADER, "the vertex shader"),
        compile_stage(GL_FRAGMENT_SHADER, fragment_source, "OCIO's generated shader"),
    ]
    program = glCreateProgram()
    for stage in stages:
        glAttachShader(program, stage)
    glLinkProgram(program)
    linked = glGetProgramiv(program, GL_LINK_STATUS)
    for stage in stages:
        glDeleteShader(stage)
    if not linked:
        log = glGetProgramInfoLog(program)
        glDeleteProgram(program)
        raise ShaderError(f"OCIO's generated shader did not link: {_text(log)}")
    return program


def program_for(desc: OCIO.GpuShaderDesc) -> int:
    """The linked program for a transform's generated shader."""
    return link(
        FRAGMENT_SHADER.format(
            source=SOURCE,
            result=RESULT,
            generated=desc.getShaderText(),
            entry=desc.getFunctionName(),
        )
    )


def _text(log) -> str:
    return log.decode(errors="replace").strip() if isinstance(log, bytes) else str(log)


def _filter(interpolation) -> int:
    """GL's name for the filtering OCIO asked for."""
    return GL_NEAREST if interpolation == OCIO.INTERP_NEAREST else GL_LINEAR


def bind_ocio_textures(program: int, desc: OCIO.GpuShaderDesc) -> list[int]:
    """Upload every texture OCIO declared and point its sampler at it.

    A 3D texture is refused rather than guessed at: the pinned config's shaders
    ask for none, so the path would ship untested and be believed anyway.
    """
    if list(desc.get3DTextures()):
        raise ShaderError("this harness binds 1D and 2D textures, not 3D LUTs")

    handles: list[int] = []
    for unit, texture in enumerate(desc.getTextures(), start=FIRST_OCIO_UNIT):
        values = np.ascontiguousarray(texture.getValues(), dtype=np.float32)
        red = texture.channel == OCIO.GpuShaderDesc.TEXTURE_RED_CHANNEL
        internal, layout = (GL_R32F, GL_RED) if red else (GL_RGB32F, GL_RGB)
        flat = _filter(texture.interpolation)

        handle = glGenTextures(1)
        handles.append(handle)
        target = (
            GL_TEXTURE_1D
            if texture.dimensions == OCIO.GpuShaderDesc.TEXTURE_1D
            else GL_TEXTURE_2D
        )
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(target, handle)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        if target == GL_TEXTURE_1D:
            glTexImage1D(
                target, 0, internal, texture.width, 0, layout, GL_FLOAT, values
            )
        else:
            glTexImage2D(
                target,
                0,
                internal,
                texture.width,
                texture.height,
                0,
                layout,
                GL_FLOAT,
                values,
            )
            glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, flat)
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, flat)
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)

        location = glGetUniformLocation(program, texture.samplerName)
        if location < 0:
            # Refused rather than skipped. A sampler left unpointed keeps unit
            # zero, where it would read the source image and be scored as
            # OCIO's answer — a wrong number that looks like a result, which is
            # the one failure this harness must not produce.
            raise ShaderError(
                f"the linked program has no {texture.samplerName!r} to point at "
                "its table, so OCIO's shader cannot be run as written"
            )
        glUniform1i(location, unit)
    return handles


def _image(width: int, height: int, pixels: np.ndarray | None) -> int:
    """An RGBA float texture, from ``pixels`` or left undefined."""
    handle = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, handle)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        width,
        height,
        0,
        GL_RGBA,
        GL_FLOAT,
        None if pixels is None else np.ascontiguousarray(pixels, dtype=np.float32),
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    return handle


@contextlib.contextmanager
def scene(desc: OCIO.GpuShaderDesc, pixels: np.ndarray, width: int, height: int):
    """One shader, one image, one float framebuffer — set up and torn down.

    Every object is owned here, so an error anywhere inside still gives them
    back. Yields once the draw would produce OCIO's answer for ``pixels``.
    """
    context()
    program = vertices = source = target = buffer = None
    textures: list[int] = []
    try:
        program = program_for(desc)
        glUseProgram(program)
        vertices = glGenVertexArrays(1)
        glBindVertexArray(vertices)

        glActiveTexture(GL_TEXTURE0)
        source = _image(width, height, pixels)
        location = glGetUniformLocation(program, SOURCE)
        if location < 0:
            raise ShaderError(f"the linked program has no {SOURCE!r} to bind samples")
        glUniform1i(location, 0)

        textures = bind_ocio_textures(program, desc)

        # Past every unit a sampler is pointed at. Creating the render target
        # needs it bound somewhere, and binding it over a unit the shader reads
        # would silently replace those samples with this empty texture — which
        # reads as OCIO returning zero for every input rather than as a mistake.
        glActiveTexture(GL_TEXTURE0 + FIRST_OCIO_UNIT + len(textures))
        target = _image(width, height, None)

        buffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, buffer)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0
        )
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise ShaderError(f"no complete float framebuffer at {width}x{height}")
        glViewport(0, 0, width, height)
        yield
    except GLError as exc:
        # A driver-level failure is the runtime declining, and the sweep above
        # handles a refusal. Letting PyOpenGL's own error out would end it.
        raise ShaderError(f"the GL driver refused the shader: {exc}") from exc
    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
        glBindVertexArray(0)
        glActiveTexture(GL_TEXTURE0)
        if buffer is not None:
            glDeleteFramebuffers(1, [buffer])
        doomed = [handle for handle in (target, source, *textures) if handle]
        if doomed:
            glDeleteTextures(doomed)
        if vertices is not None:
            glDeleteVertexArrays(1, [vertices])
        if program is not None:
            glDeleteProgram(program)


def render(desc: OCIO.GpuShaderDesc, pixels: np.ndarray, width: int) -> np.ndarray:
    """Draw one fragment per sample and read the result back at float32.

    ``pixels`` is ``(width, 4)`` RGBA. The return is the same, so the caller
    keeps the channel bookkeeping it already had.
    """
    with scene(desc, pixels, width, 1):
        glDrawArrays(GL_TRIANGLES, 0, 3)
        # Read back at the precision the attachment holds. Anything narrower
        # would round the shader's answer before the tolerance sees it.
        read = glReadPixels(0, 0, width, 1, GL_RGBA, GL_FLOAT)
        return np.frombuffer(read, dtype=np.float32).reshape(width, 4).copy()


def time_draws(
    desc: OCIO.GpuShaderDesc,
    pixels: np.ndarray,
    width: int,
    height: int,
    iterations: int,
    warmup: int,
) -> tuple[float, np.ndarray]:
    """Seconds per frame, and a strip of the last one to prove it happened.

    Each draw is finished before the next is timed. Queuing them all and
    finishing once would measure a pipeline no consumer has, and would time a
    depth of work the other side of the comparison is not allowed to batch.
    """
    with scene(desc, pixels, width, height):
        for _ in range(warmup):
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glFinish()

        started = time.perf_counter()
        for _ in range(iterations):
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glFinish()
        elapsed = time.perf_counter() - started

        read = glReadPixels(0, 0, min(width, 64), 1, GL_RGBA, GL_FLOAT)
        strip = np.frombuffer(read, dtype=np.float32).reshape(-1, 4).copy()
        return elapsed / iterations, strip
