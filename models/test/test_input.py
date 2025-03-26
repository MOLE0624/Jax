import os
import sys
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from flax import linen as nn

# Ensure the parent directory is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layers.preproc import PreInputLayer, PreprocOp


class Model(nn.Module):
    """Example model that applies preprocessing using PreInputLayer."""

    config: list

    def setup(self):
        self.pre_input_layer = PreInputLayer(config=self.config)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pre_input_layer(x)


class TestModel(unittest.TestCase):

    def setUp(self):
        """Initialize common test variables."""
        self.input_data = jnp.ones((100, 3, 1920, 1080))
        self.config = [
            {"op": PreprocOp.RESIZE, "params": (640, 640)},
            {"op": PreprocOp.NORM, "params": (0, 255)},
        ]
        self.rng = jax.random.PRNGKey(0)

    def test_forward_pass(self):
        """Ensure the model correctly preprocesses input data."""
        model = Model(config=self.config)
        params = model.init(self.rng, self.input_data)
        output = model.apply(params, self.input_data)

        # Check if the output shape is correct
        self.assertEqual(output.shape, (100, 3, 640, 640))

        # Check if the output data type is jnp.float32 (assuming normalization results in float values)
        self.assertEqual(
            output.dtype,
            jnp.float32,
            "Expected output dtype to be jnp.float32 after normalization",
        )

    def test_value_range(self):
        """Ensure that the output values are within the expected normalized range."""
        model = Model(config=self.config)
        params = model.init(self.rng, self.input_data)
        output = model.apply(params, self.input_data)

        # Check if values are in the expected range after normalization (assuming 0-1 normalization)
        self.assertTrue(output.min() >= 0.0, "Expected all output values to be >= 0.0")
        self.assertTrue(output.max() <= 1.0, "Expected all output values to be <= 1.0")

    @patch("builtins.print")
    def test_print_initialization(self, mock_print):
        """Ensure PreInputLayer does not print during initialization."""
        input_data = jnp.ones((100, 3, 1920, 1080))

        config = [
            {"op": PreprocOp.RESIZE, "params": (640, 640)},
            {"op": PreprocOp.NORM, "params": (0, 255)},
        ]

        model = Model(config=config)

        rng = jax.random.PRNGKey(0)
        model.init(rng, input_data)  # Initialize model

        # Assert that print() was never called
        mock_print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
