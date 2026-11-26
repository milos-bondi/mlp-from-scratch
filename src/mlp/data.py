"""Caricamento di MNIST come array NumPy."""
import numpy as np
from torchvision.datasets import MNIST


def load_mnist(root: str = "data"):
    """Returns (X_train, y_train, X_test, y_test).

    X: float32 in [0, 1], shape (N, 784). y: int64, shape (N,).
    """

    train_ds = MNIST(root=root, train=True, download=True)
    test_ds = MNIST(root=root, train=False, download=True)

    # train_ds.data è un tensore uint8 di shape (60000, 28, 28) con valori in [0, 255]
    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)

    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test
