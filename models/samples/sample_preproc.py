import os
import sys

import jax
import jax.numpy as jnp
from flax import linen as nn

# Ensure the parent directory is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layers.preproc import PreInputLayer, PreprocOp


# Example Model that uses PreInputLayer
class Model(nn.Module):
    config: list  # Configuration for preprocessing

    def setup(self):
        """Initialize the preprocessing layer with the given configuration."""
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

    # Initialize the model parameters
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_data)

    # Forward pass through the model with the initialized parameters
    output = model.apply(params, input_data)

    print("Output shape:", output.shape)  # Should be (100, 3, 640, 640)
