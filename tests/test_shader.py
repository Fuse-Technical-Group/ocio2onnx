"""OCIO's own shader, scored against the oracle the compiler answers to.

Two questions, and only one of them needs a driver. What OCIO's GPU path has
to *sample* is a property of the shader it generates, readable anywhere; how
far the shader then *lands* from the reference needs a real GL context, so
those tests skip where there is none rather than failing a machine that was
never asked to have one.
"""

import numpy as np
import pytest

from ocio2onnx.addressing import resolve_display_view
from ocio2onnx.oracle import compare, cpu_reference, lattice, run_graph
from ocio2onnx.shader import ShaderError, sampled_textures

#: The view whose look is a fixed function over two sampled tables.
ACES_VIEW = ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)")

#: A closed-form display view: OCIO's shader needs no texture for it either.
OPEN_VIEW = ("sRGB - Display", "Un-tone-mapped")


@pytest.fixture(scope="session")
def view(config, config_uri):
    """The processor behind a display view, which is not a color space pair."""

    def view(pair):
        display, name = pair
        return resolve_display_view(config, display, name, uri=config_uri).processor

    return view


@pytest.fixture(scope="session")
def render():
    """OCIO's shader output, or a skip where no driver will run it."""
    pytest.importorskip("glfw", reason="the shader extra is not installed")
    pytest.importorskip("OpenGL", reason="the shader extra is not installed")
    from ocio2onnx.shader import shader_reference

    def render(processor, samples):
        try:
            return shader_reference(processor, samples)
        except ShaderError as exc:
            pytest.skip(f"no usable GL 4.0 context here: {exc}")

    return render


def test_a_matrix_needs_no_texture_from_either_of_us(config):
    """Where OCIO's own shader is closed-form there is nothing to compare on
    this axis, which is most of the config."""
    assert sampled_textures(config.getProcessor("ACEScg", "ACES2065-1")) == 0


def test_the_aces_output_transform_costs_ocio_two_tables(view):
    """The hue tables §spec:op-coverage counts. This compiler emits the same
    transform with no texture at all."""
    assert sampled_textures(view(ACES_VIEW)) == 2


def test_the_shader_reaches_the_reference_on_a_closed_form_transform(config, render):
    """First that the harness is sound: a matrix is the one op where a shader,
    a graph and the CPU cannot honestly disagree, so a failure here is this
    module's plumbing rather than OCIO's arithmetic."""
    processor = config.getProcessor("ACEScg", "ACES2065-1")
    samples = lattice(processor)
    result = compare(cpu_reference(processor, samples), render(processor, samples))
    assert result.ok, str(result)


def test_the_shader_returns_a_transformed_image_not_its_input(view, render):
    """A harness that silently rendered the source texture through would agree
    with nothing, but one that skipped the draw and read back a cleared buffer
    is the failure that looks like a result — an all-zero answer scores as a
    real disagreement rather than as the mistake it is. Both are ruled out."""
    processor = view(OPEN_VIEW)
    samples = lattice(processor)
    got = render(processor, samples)
    assert np.isfinite(got).any()
    assert not np.array_equal(got, samples)
    assert (got != 0.0).any()


def test_the_graph_lands_closer_to_the_reference_than_ocios_shader(
    view, compile_bare, render
):
    """The claim the whole module exists to test, on one transform where the
    gap is four orders wide. ``margin`` is what compares them: each deviation
    against the bound that governed it (§spec:verification)."""
    processor = view(OPEN_VIEW)
    samples = lattice(processor)
    want = cpu_reference(processor, samples)

    graph = compare(want, run_graph(compile_bare(processor), samples), samples)
    shader = compare(want, render(processor, samples), samples)

    assert graph.ok and shader.ok
    assert graph.margin < shader.margin
