import os
from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from jax import random
from jax.example_libraries import stax

# Load MNIST test data
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images / 255.0
test_images = test_images.astype(np.float32)


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


test_labels_onehot = _one_hot(test_labels, 10)

# Define the model architecture (same as in train.py)
init_random_params, predict = stax.serial(
    stax.Flatten,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(10),
    stax.LogSoftmax,
)


def load_best_model():
    """Load the best model from checkpoints based on test accuracy."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoints")

    # Create checkpoint manager
    manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax.checkpoint.PyTreeCheckpointer()
    )

    # Get all checkpoints
    checkpoints = manager.all_steps()
    if not checkpoints:
        raise ValueError("No checkpoints found!")

    # Load each checkpoint and find the best one
    best_acc = 0
    best_params = None
    best_epoch = 0

    for step in checkpoints:
        checkpoint_data = manager.restore(step)
        if checkpoint_data["test_acc"] > best_acc:
            best_acc = checkpoint_data["test_acc"]
            best_params = checkpoint_data["params"]
            best_epoch = checkpoint_data["epoch"]

    print(
        f"Loaded best model from epoch {best_epoch} with test accuracy: {best_acc:.4f}"
    )
    return best_params


def visualize_predictions(params, num_samples=5):
    """Visualize model predictions on random test samples."""
    # Randomly select samples
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    selected_images = test_images[indices]
    selected_labels = test_labels[indices]

    # Get predictions
    predictions = predict(params, selected_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]

    for i, (ax, img, true_label, pred_label) in enumerate(
        zip(axes, selected_images, selected_labels, predicted_labels)
    ):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)

    plt.tight_layout()
    plt.show()


def main():
    # Load the best model
    params = load_best_model()

    # Visualize predictions
    visualize_predictions(params)

    # Calculate overall test accuracy
    test_predictions = predict(params, test_images)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    accuracy = np.mean(test_pred_labels == test_labels)
    print(f"\nOverall test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
