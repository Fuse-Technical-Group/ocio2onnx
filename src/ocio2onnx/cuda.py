"""Transpile a processor's GPU shader to one fused CUDA kernel (§spec:cuda-kernel).

The graph is the portable artifact, but a graph executor pays a full-frame
memory pass per fusion segment, and the ACES 2.0 display renders carry a
fusion-hostile op chain — measured at 4K on TensorRT, 9 ms against the
0.3 ms a single pass costs. OCIO's own GPU renderer already fuses: it emits
the whole transform as one shader function with its tables published as
textures. This module asks OCIO for that shader and rewrites it into a
self-contained ``.cu`` file — tables embedded, no includes, two planar RGB
entry points — that a consumer compiles with NVRTC and launches on the
stream its tensors already live on.

The rewrite is textual and covers the dialect OCIO emits for closed-form
transforms and nearest-sampled 1D tables. What falls outside — dynamic
properties, which arrive as uniforms, and interpolated or multi-dimensional
textures — is refused by name before anything is emitted, by the same
philosophy as ``compiler.unsupported_ops`` (§spec:op-coverage).
"""

from __future__ import annotations

import re

import numpy as np
import PyOpenColorIO as OCIO

from ocio2onnx.addressing import OPTIMIZATION_FLAGS, Resolved
from ocio2onnx.builder import CHANNELS
from ocio2onnx.emitters import UnsupportedOpError
from ocio2onnx.oracle import Comparison, Tolerance, compare, cpu_reference, lattice

__all__ = [
    "GPU_TOLERANCE",
    "UnsupportedShaderError",
    "device_arch",
    "kernel_source",
    "run_kernel",
    "verify",
]


class UnsupportedShaderError(UnsupportedOpError):
    """A shader whose dialect this transpiler has no path for.

    An `emitters.UnsupportedOpError`, so a caller that already routes "this
    compiler will not emit it" — the CLI's exit code, a consumer's retry on
    the graph — routes this refusal the same way without new handling.
    """


#: What the kernel is held to (§spec:cuda-kernel). Wider than the graph's
#: ``oracle.TOLERANCE`` because GPU transcendentals are not libm: OCIO's own
#: GPU renderer lands ~3e-5 from its CPU renderer on these transforms, and
#: this kernel was measured at 5.8e-5 worst-case over two million stress
#: samples (1.2e-6 across the verification lattice). The band still catches
#: what verification exists to catch — a transposed matrix, an off-by-one
#: table index — which miss by orders of magnitude.
GPU_TOLERANCE = Tolerance(absolute=2e-4, relative=1e-3)

#: The shader language the transpiler reads. GLSL 4.0 is the dialect the
#: rewrites below were written against; other languages name the same math
#: with different types.
LANGUAGE = OCIO.GPU_LANGUAGE_GLSL_4_0

#: Threads per block for both entry points. One thread per pixel, no shared
#: memory, so occupancy is insensitive to this within the usual range.
BLOCK = 256

#: GLSL float literals are float32; the same spelling in C++ is a double,
#: and one double in a chain drops the whole expression to double-precision
#: arithmetic — a 64x throughput cliff on a consumer die, not a wrong answer.
_UNSUFFIXED_FLOAT = re.compile(
    r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])"
)

_UNIFORM_SAMPLER = re.compile(r"^uniform sampler\dD \w+Sampler;\n", re.M)

#: ``const float name[N] = float[N](...);`` — GLSL's array constructor,
#: rewritten to an aggregate initializer in global memory.
_CONST_ARRAY = re.compile(r"const float (\w+)\[(\d+)\] = float\[\d+\]\((.*?)\);", re.S)

#: A file-scope function definition, which device code must mark.
_FUNCTION = re.compile(r"^((?:float|int|bool|void|vec[234]|mat[34]) \w+\()", re.M)

#: The support code every generated kernel carries: GLSL's vector and matrix
#: types over CUDA's, the builtins the emitted dialect calls, and half
#: conversion by PTX so the file needs no header. Swizzle field reads and
#: writes become the ``rgb()``/``set_rgb()`` calls the rewrites produce —
#: ``x/y/z/w`` and ``r/g/b/a`` stay plain members via the union. Tables are
#: global memory deliberately: divergent indices (the cusp search) serialize
#: the constant cache 32-way, measured at 5x the whole kernel's cost.
_PRELUDE = r"""
struct vec2 {
    union { struct { float x, y; }; struct { float r, g; }; };
    __device__ vec2() { x = 0.f; y = 0.f; }
    __device__ vec2(float x_, float y_) { x = x_; y = y_; }
};

struct vec3 {
    union { struct { float x, y, z; }; struct { float r, g, b; }; };
    __device__ vec3() { x = 0.f; y = 0.f; z = 0.f; }
    __device__ explicit vec3(float s) { x = s; y = s; z = s; }
    __device__ vec3(float x_, float y_, float z_) { x = x_; y = y_; z = z_; }
    __device__ vec3 rgb() const { return *this; }
    __device__ void set_rgb(vec3 v) { x = v.x; y = v.y; z = v.z; }
    __device__ vec2 rg() const { return vec2(x, y); }
};

struct vec4 {
    union { struct { float x, y, z, w; }; struct { float r, g, b, a; }; };
    __device__ vec4() { x = 0.f; y = 0.f; z = 0.f; w = 0.f; }
    __device__ vec4(float x_, float y_, float z_, float w_) {
        x = x_; y = y_; z = z_; w = w_;
    }
    __device__ vec3 rgb() const { return vec3(x, y, z); }
    __device__ void set_rgb(vec3 v) { x = v.x; y = v.y; z = v.z; }
    __device__ vec2 rg() const { return vec2(x, y); }
};

__device__ vec3 operator+(vec3 a, vec3 b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ vec3 operator-(vec3 a, vec3 b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ vec3 operator*(vec3 a, vec3 b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ vec3 operator/(vec3 a, vec3 b) {
    return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__device__ vec3 operator+(float s, vec3 a) { return vec3(s + a.x, s + a.y, s + a.z); }
__device__ vec3 operator-(float s, vec3 a) { return vec3(s - a.x, s - a.y, s - a.z); }
__device__ vec3 operator*(float s, vec3 a) { return vec3(s * a.x, s * a.y, s * a.z); }
__device__ vec3 operator*(vec3 a, float s) { return s * a; }
__device__ vec3 operator/(vec3 a, float s) { return vec3(a.x / s, a.y / s, a.z / s); }
__device__ vec4 operator+(vec4 a, vec4 b) {
    return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ vec4 operator-(vec4 a, vec4 b) {
    return vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__device__ vec4 operator*(vec4 a, vec4 b) {
    return vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

// GLSL matrix constructors are column-major; storage keeps that order.
struct mat3 {
    float m[9];
    __device__ mat3(float a0, float a1, float a2, float a3, float a4,
                    float a5, float a6, float a7, float a8) {
        m[0] = a0; m[1] = a1; m[2] = a2; m[3] = a3; m[4] = a4;
        m[5] = a5; m[6] = a6; m[7] = a7; m[8] = a8;
    }
};
__device__ vec3 operator*(mat3 a, vec3 v) {
    return vec3(a.m[0] * v.x + a.m[3] * v.y + a.m[6] * v.z,
                a.m[1] * v.x + a.m[4] * v.y + a.m[7] * v.z,
                a.m[2] * v.x + a.m[5] * v.y + a.m[8] * v.z);
}
struct mat4 {
    float m[16];
    __device__ mat4(float a0, float a1, float a2, float a3, float a4,
                    float a5, float a6, float a7, float a8, float a9,
                    float a10, float a11, float a12, float a13, float a14,
                    float a15) {
        m[0] = a0; m[1] = a1; m[2] = a2; m[3] = a3;
        m[4] = a4; m[5] = a5; m[6] = a6; m[7] = a7;
        m[8] = a8; m[9] = a9; m[10] = a10; m[11] = a11;
        m[12] = a12; m[13] = a13; m[14] = a14; m[15] = a15;
    }
};
__device__ vec4 operator*(mat4 a, vec4 v) {
    return vec4(a.m[0] * v.x + a.m[4] * v.y + a.m[8] * v.z + a.m[12] * v.w,
                a.m[1] * v.x + a.m[5] * v.y + a.m[9] * v.z + a.m[13] * v.w,
                a.m[2] * v.x + a.m[6] * v.y + a.m[10] * v.z + a.m[14] * v.w,
                a.m[3] * v.x + a.m[7] * v.y + a.m[11] * v.z + a.m[15] * v.w);
}

// GLSL builtins over the types above. sign(0) is 0, which copysign is not.
__device__ float sign(float x) { return (float)((x > 0.f) - (x < 0.f)); }
__device__ vec3 sign(vec3 v) { return vec3(sign(v.x), sign(v.y), sign(v.z)); }
__device__ vec4 sign(vec4 v) {
    return vec4(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
}
__device__ float abs(float x) { return fabsf(x); }
__device__ vec3 abs(vec3 v) { return vec3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
__device__ vec4 abs(vec4 v) {
    return vec4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}
__device__ float floor(float x) { return floorf(x); }
__device__ float sqrt(float x) { return sqrtf(x); }
__device__ float log(float x) { return logf(x); }
__device__ float exp(float x) { return expf(x); }
__device__ float log2(float x) { return log2f(x); }
__device__ float exp2(float x) { return exp2f(x); }
__device__ float cos(float x) { return cosf(x); }
__device__ float sin(float x) { return sinf(x); }
__device__ float atan(float y, float x) { return atan2f(y, x); }
__device__ float atan(float x) { return atanf(x); }
__device__ float pow(float x, float y) { return powf(x, y); }
__device__ vec3 pow(vec3 v, vec3 e) {
    return vec3(powf(v.x, e.x), powf(v.y, e.y), powf(v.z, e.z));
}
__device__ vec4 pow(vec4 v, vec4 e) {
    return vec4(powf(v.x, e.x), powf(v.y, e.y), powf(v.z, e.z), powf(v.w, e.w));
}
__device__ float min(float a, float b) { return fminf(a, b); }
__device__ float max(float a, float b) { return fmaxf(a, b); }
__device__ vec3 min(vec3 a, vec3 b) {
    return vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__device__ vec3 max(vec3 a, vec3 b) {
    return vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__device__ float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ vec3 mix(vec3 a, vec3 b, float t) {
    return vec3(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t));
}
__device__ float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ vec4 greaterThan(vec4 a, vec4 b) {
    return vec4(a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w);
}
__device__ vec4 lessThan(vec4 a, vec4 b) {
    return vec4(a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w);
}

__device__ float half_to_float(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ unsigned short float_to_half(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
"""

#: The entry points. Planar RGB in and out, one thread per pixel; the fourth
#: channel exists only inside the call, at zero, and is never read back —
#: alpha bypasses the kernel exactly as it bypasses the graph
#: (§spec:emitted-graph). ``@FUNCTION@`` is the name the shader declares.
_KERNELS = r"""
extern "C" __global__ void apply_f32(const float* __restrict__ in,
                                     float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec4 c = @FUNCTION@(vec4(in[i], in[i + n], in[i + 2 * n], 0.f));
    out[i] = c.x;
    out[i + n] = c.y;
    out[i + 2 * n] = c.z;
}

extern "C" __global__ void apply_f16(const unsigned short* __restrict__ in,
                                     unsigned short* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec4 c = @FUNCTION@(vec4(half_to_float(in[i]), half_to_float(in[i + n]),
                             half_to_float(in[i + 2 * n]), 0.f));
    out[i] = float_to_half(c.x);
    out[i + n] = float_to_half(c.y);
    out[i + 2 * n] = float_to_half(c.z);
}
"""


def kernel_source(resolved: Resolved) -> str:
    """One self-contained ``.cu`` for a resolved transform, or a refusal.

    The shader is extracted at the same optimization level the oracle runs
    (`addressing.OPTIMIZATION_FLAGS`), so the kernel and the reference
    disagree only on arithmetic, never on which ops ran.
    """
    desc = OCIO.GpuShaderDesc.CreateShaderDesc(language=LANGUAGE)
    resolved.processor.getOptimizedGPUProcessor(
        OPTIMIZATION_FLAGS
    ).extractGpuShaderInfo(desc)
    _refuse_unsupported(desc)

    header = "\n".join(
        ["// Fused CUDA kernel. Compile with NVRTC; --use_fast_math holds the"]
        + ["// tolerance this file was verified at (§spec:cuda-kernel)."]
        + [f"// {key}: {value}" for key, value in sorted(resolved.metadata.items())]
    )
    tables = "".join(_table(texture) for texture in desc.getTextures())
    body = _transpile(desc.getShaderText())
    kernels = _KERNELS.replace("@FUNCTION@", desc.getFunctionName())
    return "\n".join([header, _PRELUDE, tables, body, kernels])


def _refuse_unsupported(desc) -> None:
    """Name everything the transpiler has no path for, then refuse once.

    Refusing before emission names them all, as the compiler does for ops
    (§spec:op-coverage).
    """
    blocked: list[str] = []
    for name, _ in desc.getUniforms():
        blocked.append(f"uniform {name!r} (a dynamic property)")
    for texture in desc.getTextures():
        if texture.dimensions != OCIO.GpuShaderDesc.TEXTURE_1D:
            blocked.append(f"texture {texture.textureName!r} (not one-dimensional)")
        elif texture.interpolation != OCIO.INTERP_NEAREST:
            blocked.append(f"texture {texture.textureName!r} (not nearest-sampled)")
    for texture in desc.get3DTextures():
        blocked.append(f"texture {texture.textureName!r} (three-dimensional)")
    if blocked:
        raise UnsupportedShaderError(
            f"{', '.join(blocked)} {'is' if len(blocked) == 1 else 'are'} not "
            "transpiled by this compiler; the ONNX graph carries this transform"
        )


def _table(texture) -> str:
    """One texture as a global array and the sampler the rewrite calls.

    A nearest-sampled 1D texture at coordinate ``u`` reads texel
    ``floor(u * width)``; the clamp is GL's edge behaviour.
    """
    values = np.asarray(texture.getValues(), dtype=np.float32)
    name = texture.textureName
    width = texture.width
    if texture.channel == OCIO.GpuShaderDesc.TEXTURE_RED_CHANNEL:
        texel = f"return vec4({name}_data[i], 0.f, 0.f, 1.f);"
    else:
        texel = (
            f"return vec4({name}_data[3 * i], {name}_data[3 * i + 1], "
            f"{name}_data[3 * i + 2], 1.f);"
        )
    entries = ", ".join(f"{value!r}f" for value in values.tolist())
    return (
        f"__device__ const float {name}_data[{len(values)}] = {{ {entries} }};\n"
        f"__device__ vec4 {name}_texel(float u) {{\n"
        f"    int i = (int)floorf(u * {width}.f);\n"
        f"    i = i < 0 ? 0 : (i > {width - 1} ? {width - 1} : i);\n"
        f"    {texel}\n"
        f"}}\n"
    )


def _transpile(glsl: str) -> str:
    """GLSL to CUDA C++, textually.

    The prelude carries the types, so most of the shader is already valid;
    what remains is sampler plumbing, GLSL's array constructor, ``__device__``
    marks, the two compound swizzles the dialect uses, and literal suffixes.
    Order matters twice: swizzle writes before swizzle reads, so the
    assignment's own target is not first rewritten into an rvalue, and
    literals last, so no earlier pattern has to anticipate the suffix.
    """
    out = _UNIFORM_SAMPLER.sub("", glsl)
    out = _CONST_ARRAY.sub(r"__device__ const float \1[\2] = {\3};", out)
    out = re.sub(r"texture\((\w+)Sampler,\s*", r"\1_texel(", out)
    out = _FUNCTION.sub(r"__device__ \1", out)
    out = re.sub(
        r"^(\s*)([A-Za-z_]\w*)\.rgb\s*=\s*(.*);",
        r"\1\2.set_rgb(\3);",
        out,
        flags=re.M,
    )
    out = re.sub(r"\.rgb\b", ".rgb()", out)
    out = re.sub(r"\.rg\b", ".rg()", out)
    return _UNSUFFIXED_FLOAT.sub(r"\1f", out)


def device_arch() -> str:
    """The compute capability NVRTC should target, read off device zero."""
    from cuda.bindings import driver as cu

    _check(cu.cuInit(0))
    device = _check(cu.cuDeviceGet(0))
    major = _check(
        cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
        )
    )
    minor = _check(
        cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
        )
    )
    return f"compute_{major}{minor}"


def compile_ptx(source: str, arch: str | None = None) -> bytes:
    """Compile a generated kernel to PTX, or fail with NVRTC's own log.

    ``--use_fast_math`` is not an option here but the contract: it selects
    the hardware log2/exp2 paths GLSL's ``pow`` uses, `GPU_TOLERANCE` was
    measured under it, and the accurate paths cost 3x for a difference the
    tolerance cannot see.
    """
    from cuda.bindings import nvrtc

    err, program = nvrtc.nvrtcCreateProgram(source.encode(), b"ocio2onnx.cu", 0, [], [])
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC: {err}")
    options = [
        f"--gpu-architecture={arch or device_arch()}".encode(),
        b"--use_fast_math",
    ]
    (result,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, size = nvrtc.nvrtcGetProgramLogSize(program)
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(f"NVRTC refused the generated kernel:\n{log.decode()}")
    _, size = nvrtc.nvrtcGetPTXSize(program)
    ptx = b" " * size
    nvrtc.nvrtcGetPTX(program, ptx)
    return ptx


def run_kernel(source: str, samples: np.ndarray, precision: str = "f32") -> np.ndarray:
    """Execute a generated kernel over lattice-shaped samples, on device zero.

    ``precision`` names the entry point. The half path quantizes the samples
    to float16 on the way in, which is the claim that entry point makes: the
    same arithmetic behind half-precision edges.
    """
    from cuda.bindings import driver as cu

    shape = np.shape(samples)
    planes = np.ascontiguousarray(np.reshape(samples, (CHANNELS, -1)), dtype=np.float32)
    count = planes.shape[1]
    host_in = planes if precision == "f32" else planes.astype(np.float16)

    ptx = compile_ptx(source)
    context = _check(cu.cuDevicePrimaryCtxRetain(_check(cu.cuDeviceGet(0))))
    _check(cu.cuCtxSetCurrent(context))
    module = _check(cu.cuModuleLoadData(ptx))
    kernel = _check(cu.cuModuleGetFunction(module, f"apply_{precision}".encode()))

    d_in = _check(cu.cuMemAlloc(host_in.nbytes))
    d_out = _check(cu.cuMemAlloc(host_in.nbytes))
    try:
        _check(cu.cuMemcpyHtoD(d_in, host_in.ctypes.data, host_in.nbytes))
        arguments = [
            np.array([int(d_in)], dtype=np.uint64),
            np.array([int(d_out)], dtype=np.uint64),
            np.array([count], dtype=np.int32),
        ]
        pointers = np.array([a.ctypes.data for a in arguments], dtype=np.uint64)
        _check(
            cu.cuLaunchKernel(
                kernel,
                (count + BLOCK - 1) // BLOCK,
                1,
                1,
                BLOCK,
                1,
                1,
                0,
                0,
                pointers.ctypes.data,
                0,
            )
        )
        host_out = np.empty_like(host_in)
        _check(cu.cuMemcpyDtoH(host_out.ctypes.data, d_out, host_out.nbytes))
    finally:
        cu.cuMemFree(d_in)
        cu.cuMemFree(d_out)
        cu.cuModuleUnload(module)
    return np.reshape(host_out.astype(np.float32), shape)


def verify(resolved: Resolved, source: str | None = None) -> Comparison:
    """Hold a kernel against the oracle at `GPU_TOLERANCE`.

    The same lattice and the same reference as the graph's verification
    (§spec:verification); only the tolerance differs, for the reason it
    documents. A caller holding the source it is about to write hands it
    over, so what is verified is the artifact.
    """
    samples = lattice(resolved.processor)
    return compare(
        cpu_reference(resolved.processor, samples),
        run_kernel(kernel_source(resolved) if source is None else source, samples),
        samples,
        tolerance=GPU_TOLERANCE,
    )


def _check(result):
    """Unpack a cuda-python result tuple, raising on any error code."""
    from cuda.bindings import driver as cu

    error, *rest = result if isinstance(result, tuple) else (result,)
    if error != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver: {error}")
    return rest[0] if len(rest) == 1 else rest
