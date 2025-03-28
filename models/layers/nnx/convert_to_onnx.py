#!/usr/bin/env python3

from analyzer import Analyzer
from flax import nnx
from onnx import TensorProto, helper

_analyzer = Analyzer()


# Define the ONNX Resize converter
def convert_resize(input_name: str, output_name: str, info: dict, input_shape: list):
    # Create the Resize node in ONNX format
    resize_node = helper.make_node(
        "Resize",  # Operation type
        inputs=[input_name],  # Input tensor
        outputs=[output_name],  # Output tensor
        mode="nearest",  # Resize mode (could also be linear, cubic, etc.)
        sizes=[info["target_height"], info["target_width"]],  # Resize dimensions
        coordinate_transformation_mode="asymmetric",  # Transformation mode (you can change it as needed)
        nearest_mode='round_prefer_floor'
    )

    # Calculate output shape (resized height, width)
    output_shape = [
        input_shape[0],
        input_shape[1],
        info["target_height"],
        info["target_width"],
    ]

    return [resize_node], [], output_shape


def convert_norm(input_name: str, output_name: str, info: dict, input_shape: list):
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
    return (
        [sub_node, div_node],
        [min_val_tensor, max_val_tensor, norm_factor_tensor],
        output_shape,
    )


def convert(model: nnx.Module):
    # Get layer info from the analyzer (a)
    layer_info = _analyzer(model)

    # Create an empty list to store ONNX nodes and initializers
    onnx_nodes = []
    initializer_tensors = []

    input_name = "input"
    input_shape = ["N", "C", "H", "W"]

    output_name = "output"

    crr_input_shape = input_shape

    # Loop through the extracted layer info and convert specific layers
    for idx, layer in enumerate(layer_info):
        # Dynamically generate output tensor name for the current layer
        _output_name = (
            f"output_{idx}" if idx < len(layer_info) - 1 else output_name
        )  # Unique output name for each layer

        if "Resize" in layer["type"]:
            # Convert resize operation to an ONNX node
            nodes, initializers, output_shape = convert_resize(
                input_name, _output_name, layer["params"], crr_input_shape
            )
            onnx_nodes.extend(nodes)
            initializer_tensors.extend(initializers)
        elif "Norm" in layer["type"]:
            # Convert normalization operation to an ONNX node
            nodes, initializers, output_shape = convert_norm(
                input_name, _output_name, layer["params"], crr_input_shape
            )
            onnx_nodes.extend(nodes)
            initializer_tensors.extend(initializers)

        # After processing, the output of the current layer becomes the input for the next layer
        input_name = _output_name  # Update input name for the next layer
        crr_input_shape = output_shape

    output_tensor = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, crr_input_shape
    )  # Use input_shape for output

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(
        nodes=onnx_nodes,  # Add the nodes
        name="NormGraph",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        ],  # Input tensor(s) with the specified shape
        outputs=[output_tensor],  # Output tensor(s)
        initializer=initializer_tensors,  # Add initializers
    )

    # Create the ONNX model from the graph
    onnx_model = helper.make_model(graph, producer_name="onnx", opset_imports=[helper.make_opsetid("onnx", 11)])

    return onnx_model
