"""Utilities for loading the MNIST dataset as NumPy arrays."""

import numpy as np
from torchvision.datasets import MNIST


def load_mnist(
    root: str = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST if needed and return its training and test arrays.

    Images are flattened to 784-element ``float32`` arrays scaled to [0, 1].
    Labels are ``int64`` arrays containing the digit class for each image.

    Args:
        root: Directory in which torchvision stores the MNIST files.

    Returns:
        A tuple containing training images, training labels, test images, and
        test labels, in that order.
    """

    train_ds = MNIST(root=root, train=True, download=True)
    test_ds = MNIST(root=root, train=False, download=True)

    # The image tensors have shape (N, 28, 28) and values in [0, 255].
    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)

    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test
