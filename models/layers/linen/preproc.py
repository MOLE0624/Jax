from enum import Enum

import jax
import jax.image
import jax.numpy as jnp
from flax import linen as nn


# Define the Preprocessing Operations Enum
class PreprocOp(Enum):
    RESIZE = "resize"
    NORM = "norm"


# Define operations like resize and normalization
class PreResizeLayer(nn.Module):
    target_height: int
    target_width: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Resize the input to the target height and width."""
        return jax.image.resize(
            x,
            (x.shape[0], x.shape[1], self.target_height, self.target_width),
            method="nearest",
        )


class PreNormLayer(nn.Module):
    min_val: float
    max_val: float

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalize the input to a given range (e.g., [0, 255])."""
        return (x - self.min_val) / (self.max_val - self.min_val)


# The PreInputLayer which accepts a configuration for various operations
class PreInputLayer(nn.Module):
    config: list
    _debug_printed: bool = False  # Class-level flag to track if we've printed

    def setup(self):
        """Initialize the layers based on the configuration."""
        layers_list = []
        for op in self.config:
            if op["op"] == PreprocOp.RESIZE:
                target_height, target_width = op["params"]
                layers_list.append(
                    PreResizeLayer(
                        target_height=target_height, target_width=target_width
                    )
                )
            elif op["op"] == PreprocOp.NORM:
                min_val, max_val = op["params"]
                layers_list.append(PreNormLayer(min_val=min_val, max_val=max_val))

        self.layers = tuple(layers_list)

    def __str__(self):
        # Only print once using the class-level flag
        if not PreInputLayer._debug_printed:
            PreInputLayer._debug_printed = True
            layer_details = "\n".join(
                [
                    f"    {line}"
                    for layer in self.layers
                    for line in str(layer).splitlines()
                ]
            )
            return f"PreInputLayer\n{layer_details})"
        return ""  # Return simple string for subsequent calls

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply each operation in the order specified in the config."""
        for layer in self.layers:
            x = layer(x)
        return x
