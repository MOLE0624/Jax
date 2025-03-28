from enum import Enum

import jax
import jax.image
import jax.numpy as jnp
from flax import nnx


# Define the Preprocessing Operations Enum
class PreprocOp(Enum):
    RESIZE = "resize"
    NORM = "norm"


class PreResizeLayer(nnx.Module):
    target_width: int
    target_height: int

    def __init__(self, target_width: int, target_height: int):
        """Initialize target width and height."""
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Resize the input tensor using nearest-neighbor interpolation."""
        return jax.image.resize(
            x,
            (x.shape[0], x.shape[1], self.target_height, self.target_width),
            method="nearest",
        )


class PreNormLayer(nnx.Module):
    min_val: float
    max_val: float

    def __init__(self, min_val: float, max_val: float):
        """Initialize min and max normalization values."""
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalize the input tensor to a given range (e.g., [0, 1])."""
        return (x - self.min_val) / (self.max_val - self.min_val)


# The PreInputLayer which accepts a configuration for various operations
class PreInputLayer(nnx.Module):
    config: list
    _debug_printed: bool = False  # Class-level flag to track debug print

    def __init__(self, config):
        """Initialize the layers based on the configuration."""
        self.config = config
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

        # Use nnx.ModuleList to store layers
        self.layers = tuple(layers_list)

    def __str__(self):
        """Print debug info only once."""
        if not PreInputLayer._debug_printed:
            PreInputLayer._debug_printed = True
            layer_details = "\n".join(f"  {layer}" for layer in self.layers)
            return f"PreInputLayer(\n{layer_details}\n)"
        return ""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply each preprocessing operation sequentially."""
        for layer in self.layers:
            x = layer(x)
        return x
