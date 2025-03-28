import os
import sys

import jax.numpy as jnp
import onnx
from flax import nnx as nn

# Ensure the parent directory is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import convert_to_onnx
from preproc import PreInputLayer, PreprocOp


# Example Model that uses PreInputLayer
class Model(nn.Module):
    config: list  # Configuration for preprocessing

    def __init__(self, config):
        self.config = config
        self.pre_input_layer = PreInputLayer(config=self.config)
        print(self.pre_input_layer)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Preprocess input and forward through the model."""
        return self.pre_input_layer(x)


# Example usage:
if __name__ == "__main__":
    # Create an example input (NxCxHxW), for instance: (100, 3, 1920, 1080)
    input_data = jnp.ones((100, 3, 1920, 1080))  # Shape: (100, 3, 1920, 1080)

    # Define configuration with any order of operations (resize, normalize)
    config = [
        {"op": PreprocOp.RESIZE, "params": (640, 640)},  # Resize to 640x640
        {"op": PreprocOp.NORM, "params": (0, 255)},  # Normalize between 0 and 255
    ]

    # Initialize the model with the config
    model = Model(config=config)

    output = model(input_data)

    print()
    print(output.shape)
    print()

    onnx_model = convert_to_onnx.convert(model=model)

    onnx.checker.check_model(onnx_model)  # Check if the model is valid

    import numpy as np
    import onnxruntime as ort

    # Now, let's test inference using ONNX Runtime with dummy data
    # Create some dummy input data matching the expected input shape
    input_data = np.random.rand(1, 3, 1920, 1080).astype(
        np.int8
    )  # Batch size 1, Channels 3, Height 640, Width 640

    # Create ONNX Runtime session using the model in memory (not saved to a file)
    session = ort.InferenceSession(onnx_model.SerializeToString())

    # Extract input tensor names
    input_name = session.get_inputs()[0].name

    # Run inference with input_data
    output_data = session.run(None, {input_name: input_data})

    # Print the outputs
    print("Inference result:", output_data)

    # onnx.save_model(onnx_model, "model.onnx")
