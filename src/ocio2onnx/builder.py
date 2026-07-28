"""Accumulate ONNX nodes and initializers into a model (§spec:emitted-graph).

The builder holds no color knowledge. Emitters name the arithmetic; this
module names the tensors, keeps them unique, and settles the graph's
interface: channels-first float32, batch and spatial dimensions free, plus one
input per live parameter (§spec:dynamic-properties).
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ocio2onnx.addressing import METADATA_PREFIX

#: The emitted graph carries three channels; alpha bypasses it rather than
#: flowing through identity arithmetic (§spec:emitted-graph).
CHANNELS = 3

#: The graph's interface: channels-first, batch and spatial dimensions free,
#: so one graph serves every resolution (§spec:emitted-graph).
IMAGE_SHAPE = ("N", CHANNELS, "H", "W")

#: The graph's input and output tensor names.
INPUT = "input"
OUTPUT = "output"

#: Opset 18 covers every operator the closed-form emitters need.
OPSET = 18

#: onnx 1.22 stamps an IR version onnxruntime 1.28 rejects, so pin one both
#: agree on rather than inherit whichever the writer happens to default to.
IR_VERSION = 10

#: The precision the graph is emitted at, and declares (§spec:precision).
PRECISION = "float32"
PRECISION_KEY = f"{METADATA_PREFIX}precision"

#: A per-channel constant broadcasts against ``(N, 3, H, W)``; a scalar is
#: rank one so it broadcasts against anything.
CHANNEL_SHAPE = (1, CHANNELS, 1, 1)
SCALAR_SHAPE = (1,)

#: What a per-channel live parameter is declared as. A consumer binding a
#: grade hands over three numbers; putting the reshape onto the image's channel
#: axis inside the graph keeps that shape out of its interface.
PARAMETER_SHAPE = (CHANNELS,)

#: Where the channels sit in ``IMAGE_SHAPE``, for an op that reduces across
#: them rather than elementwise along them.
CHANNEL_AXIS = 1


class GraphBuilder:
    """Nodes and initializers, accumulated in emission order."""

    def __init__(self) -> None:
        self._nodes: list[onnx.NodeProto] = []
        self._initializers: list[onnx.TensorProto] = []
        self._counts: collections.Counter[str] = collections.Counter()
        self._parameters: list[onnx.ValueInfoProto] = []

    def name(self, hint: str) -> str:
        """A tensor name unique within this graph."""
        index = self._counts[hint]
        self._counts[hint] += 1
        return f"{hint}_{index}"

    def constant(self, hint: str, value: Any, dtype: Any = np.float32) -> str:
        """An initializer holding ``value`` at its own shape.

        float32 unless a caller asks otherwise, which only an index does: the
        graph carries float32 and ONNX indexing operators take integers.
        """
        array = np.asarray(value, dtype=dtype)
        name = self.name(hint)
        self._initializers.append(numpy_helper.from_array(array, name))
        return name

    def per_channel(self, hint: str, values: Any, dtype: Any = np.float32) -> str:
        """A three-element constant shaped to broadcast across channels."""
        return self.constant(hint, np.reshape(values, CHANNEL_SHAPE), dtype)

    def scalar(self, hint: str, value: float) -> str:
        """A single-element constant that broadcasts against anything."""
        return self.constant(hint, np.reshape(value, SCALAR_SHAPE))

    def scalar_parameter(self, name: str, value: float) -> str:
        """A live scalar: a graph input defaulting to OCIO's own value.

        ONNX spells "input with a default" as an initializer sharing a graph
        input's name, which is what makes one compiled artifact serve every
        setting of the knob (§spec:dynamic-properties).
        """
        return self._parameter(name, np.reshape(value, SCALAR_SHAPE))

    def channel_parameter(self, name: str, values: Any) -> str:
        """A live per-channel parameter, declared flat and reshaped to
        broadcast across the image's channel axis."""
        declared = self._parameter(name, np.reshape(values, PARAMETER_SHAPE))
        return self.op(
            "Reshape",
            [declared, self.constant("channel_shape", CHANNEL_SHAPE, dtype=np.int64)],
        )

    def _parameter(self, name: str, array: np.ndarray) -> str:
        """Declare one graph input carrying ``array`` as its default.

        The name is the caller's where it is free and suffixed where it is not.
        Two graded ops in one transform hold different defaults, so a shared
        input would have to discard one of them; OCIO's own run time allows one
        dynamic property of each type per processor and drops the rest, which
        is a limit of that plumbing rather than of a graph.
        """
        taken = {value.name for value in self._parameters}
        unique, index = name, 1
        while unique in taken:
            index += 1
            unique = f"{name}_{index}"
        self._initializers.append(
            numpy_helper.from_array(np.asarray(array, dtype=np.float32), unique)
        )
        self._parameters.append(
            helper.make_tensor_value_info(unique, TensorProto.FLOAT, array.shape)
        )
        return unique

    def op(self, op_type: str, inputs: list[str], **attrs: Any) -> str:
        """Append a single-output node and return the tensor it produces."""
        output = self.name(op_type.lower())
        self._nodes.append(
            helper.make_node(op_type, inputs, [output], name=f"{output}_node", **attrs)
        )
        return output

    def mul(self, a: str, b: str) -> str:
        return self.op("Mul", [a, b])

    def add(self, a: str, b: str) -> str:
        return self.op("Add", [a, b])

    def sub(self, a: str, b: str) -> str:
        return self.op("Sub", [a, b])

    def div(self, a: str, b: str) -> str:
        return self.op("Div", [a, b])

    def pow(self, a: str, b: str) -> str:
        return self.op("Pow", [a, b])

    def reciprocal(self, x: str) -> str:
        return self.op("Reciprocal", [x])

    def to_int64(self, x: str) -> str:
        """Cast to int64, which is what a ``Gather`` index must be.

        Here rather than in an emitter so the ONNX type enumeration stays
        inside the module that speaks ONNX.
        """
        return self.op("Cast", [x], to=TensorProto.INT64)

    def where(self, condition: str, a: str, b: str) -> str:
        """Select elementwise. ONNX has no per-element branching, so an op with
        a breakpoint evaluates both sides and selects (§spec:op-emission)."""
        return self.op("Where", [condition, a, b])

    def build(
        self, output: str, name: str, metadata: dict[str, str]
    ) -> onnx.ModelProto:
        """Close the graph over ``output`` and return a checked model.

        The chain's last tensor carries an emission-order name, so an
        ``Identity`` gives the graph a stable output name for a consumer to
        bind to. It also makes an empty chain a valid graph.
        """
        self._nodes.append(
            helper.make_node("Identity", [output], [OUTPUT], name="output_node")
        )
        graph = helper.make_graph(
            self._nodes,
            name,
            [_value(INPUT), *self._parameters],
            [_value(OUTPUT)],
            self._initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
        model.ir_version = IR_VERSION
        helper.set_model_props(model, {**metadata, PRECISION_KEY: PRECISION})
        onnx.checker.check_model(model, full_check=True)
        return model


def parameters(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """A graph's live parameters and the OCIO defaults they carry.

    The knobs a consumer can vary per frame, in declaration order. Read off the
    model rather than off a side table: an input backed by an initializer is
    exactly what the format means by an overridable default, so the artifact
    answers for itself and a consumer that never loaded this package can ask
    the same question (§spec:dynamic-properties).
    """
    # Keyed off the inputs rather than off the initializers, so a half-domain
    # `Lut1D`'s 65536 entries are never decoded to report a handful of scalars.
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    return {
        value.name: numpy_helper.to_array(initializers[value.name])
        for value in model.graph.input
        if value.name in initializers
    }


def _value(name: str) -> onnx.ValueInfoProto:
    """The graph's interface: float32 at ``IMAGE_SHAPE``."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, IMAGE_SHAPE)
