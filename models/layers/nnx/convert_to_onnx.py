#!/usr/bin/env python3

from analyzer import Analyzer
from flax import nnx
from onnx import TensorProto, helper

_analyzer = Analyzer()


# Define the ONNX Resize converter
def convert_resize(
    input_name: str,
    output_name: str,
    info: dict,
    input_shape: list,
    input_type: int,
    output_type: int,
):
    # Get target dimensions from layer info
    target_height = info["target_height"]
    target_width = info["target_width"]

    # Create sizes constant node
    sizes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sizes"],
        value=helper.make_tensor(
            name="sizes",
            data_type=TensorProto.INT64,
            dims=[4],
            vals=[
                0,
                0,
                target_height,
                target_width,
            ],  # Use 0 for dynamic dimensions (N and C)
        ),
    )

    # Create the Resize node in ONNX format
    resize_node = helper.make_node(
        "Resize",  # Operation type
        inputs=[input_name, "sizes"],  # Input tensor and sizes
        outputs=[output_name],  # Output tensor
        mode="nearest",  # Resize mode
        coordinate_transformation_mode="asymmetric",  # Transformation mode
        nearest_mode="round_prefer_floor",
    )

    # Calculate output shape (resized height, width)
    output_shape = [
        input_shape[0],  # Keep N dynamic
        input_shape[1],  # Keep C dynamic
        target_height,
        target_width,
    ]

    return [sizes_node, resize_node], [], output_shape, input_type


def convert_norm(
    input_name: str,
    output_name: str,
    info: dict,
    input_shape: list,
    input_type: int,
    output_type: int,
):
    # Create a tensor for min_val
    min_val_tensor = helper.make_tensor(
        "min_val",  # Tensor name (string)
        TensorProto.FLOAT,  # Data type
        [1],  # Shape (scalar)
        [info["min_val"]],  # min_val value
    )

    # Create a tensor for max_val
    max_val_tensor = helper.make_tensor(
        "max_val",  # Tensor name (string)
        TensorProto.FLOAT,  # Data type
        [1],  # Shape (scalar)
        [info["max_val"]],  # max_val value
    )

    # Cast input to float if needed
    cast_node = None
    if input_type != TensorProto.FLOAT:
        cast_node = helper.make_node(
            "Cast", inputs=[input_name], outputs=["cast_output"], to=TensorProto.FLOAT
        )
        input_name = "cast_output"

    # Subtraction: x - min_val
    sub_node = helper.make_node(
        "Sub",  # Subtraction operation
        inputs=[input_name, "min_val"],  # Input tensor and min_val tensor names
        outputs=["subtracted_output"],
    )

    # Normalize factor (max_val - min_val)
    norm_factor_tensor = helper.make_tensor(
        "norm_factor",  # Tensor name
        TensorProto.FLOAT,  # Data type
        [1],  # Shape (single scalar value)
        [
            info["max_val"] - info["min_val"]
        ],  # Normalize factor value (max_val - min_val)
    )

    # Division: (x - min_val) / (max_val - min_val)
    div_node = helper.make_node(
        "Div",  # Division operation
        inputs=[
            "subtracted_output",
            "norm_factor",
        ],  # Result of subtraction and normalization factor
        outputs=[output_name],
    )

    # Calculate output shape (it remains the same as input shape for normalization)
    output_shape = input_shape

    # Return nodes and initializers
    nodes = [sub_node, div_node]
    if cast_node:
        nodes.insert(0, cast_node)

    return (
        nodes,
        [min_val_tensor, max_val_tensor, norm_factor_tensor],
        output_shape,
        TensorProto.FLOAT,  # Normalization always outputs float
    )


def convert(model: nnx.Module):
    # Get layer info from the analyzer (a)
    layer_info = _analyzer(model)

    # Create an empty list to store ONNX nodes and initializers
    onnx_nodes = []
    initializer_tensors = []

    input_name = "input"
    input_shape = ["N", "C", "H", "W"]
    input_type = TensorProto.INT8

    output_name = "output"
    output_type = TensorProto.FLOAT

    crr_input_shape = input_shape
    crr_input_type = input_type

    # Loop through the extracted layer info and convert specific layers
    for idx, layer in enumerate(layer_info):
        # Dynamically generate output tensor name for the current layer
        _output_name = (
            f"output_{idx}" if idx < len(layer_info) - 1 else output_name
        )  # Unique output name for each layer

        if "Resize" in layer["type"]:
            # Convert resize operation to an ONNX node
            nodes, initializers, output_shape, output_type = convert_resize(
                input_name,
                _output_name,
                layer["params"],
                crr_input_shape,
                crr_input_type,
                output_type,
            )
            onnx_nodes.extend(nodes)
            initializer_tensors.extend(initializers)
        elif "Norm" in layer["type"]:
            # Convert normalization operation to an ONNX node
            nodes, initializers, output_shape, output_type = convert_norm(
                input_name,
                _output_name,
                layer["params"],
                crr_input_shape,
                crr_input_type,
                output_type,
            )
            onnx_nodes.extend(nodes)
            initializer_tensors.extend(initializers)

        # After processing, the output of the current layer becomes the input for the next layer
        input_name = _output_name  # Update input name for the next layer
        crr_input_shape = output_shape
        crr_input_type = output_type

    output_tensor = helper.make_tensor_value_info(
        output_name, output_type, crr_input_shape
    )  # Use input_shape for output

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(
        nodes=onnx_nodes,  # Add the nodes
        name="NormGraph",
        inputs=[
            helper.make_tensor_value_info("input", input_type, input_shape)
        ],  # Input tensor(s) with the specified shape
        outputs=[output_tensor],  # Output tensor(s)
        initializer=initializer_tensors,  # Add initializers
    )

    # Create the ONNX model from the graph
    onnx_model = helper.make_model(
        graph, producer_name="onnx", opset_imports=[helper.make_opsetid("", 13)]
    )

    return onnx_model
