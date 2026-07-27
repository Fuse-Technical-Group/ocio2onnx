"""The scaffold builds and its dependencies resolve."""

import onnx
import PyOpenColorIO as OCIO

import ocio2onnx


def test_package_is_installed():
    assert ocio2onnx.__version__


def test_runtime_dependencies_import():
    assert OCIO.GetVersion()
    assert onnx.__version__
