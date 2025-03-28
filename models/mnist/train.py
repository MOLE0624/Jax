import functools
import itertools
import os
import time
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import numpy.random as npr
import orbax.checkpoint
import tensorflow as tf
from jax import grad, jit, random
from jax.example_libraries import optimizers, stax


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

train_labels = _one_hot(train_labels, 10)
test_labels = _one_hot(test_labels, 10)


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


init_random_params, predict = stax.serial(
    stax.Flatten,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(1024),
    stax.Relu,
    stax.Dense(10),
    stax.LogSoftmax,
)

rng = random.PRNGKey(0)


step_size = 0.001
num_epochs = 10
batch_size = 128
momentum_mass = 0.9


num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)


def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]


batches = data_stream()

opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)


@jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)


_, init_params = init_random_params(rng, (-1, 28 * 28))
opt_state = opt_init(init_params)
itercount = itertools.count()


def save_model(params, epoch, train_acc, test_acc):
    """Save model checkpoint using Orbax."""
    # Get the absolute path of the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint manager
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5)
    manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax.checkpoint.PyTreeCheckpointer(), options
    )

    # Prepare checkpoint data
    checkpoint_data = {
        "params": params,
        "epoch": epoch,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "timestamp": datetime.now().isoformat(),
    }

    # Save checkpoint
    manager.save(epoch, checkpoint_data)
    print(f"Saved checkpoint for epoch {epoch}")


print("\nStarting training...")
for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
        opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

    # Save model after each epoch
    save_model(params, epoch, train_acc, test_acc)
