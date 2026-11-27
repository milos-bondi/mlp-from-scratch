"""Neural network layers with manually implemented forward and backward passes."""

import numpy as np


class Linear:
    """Apply a fully connected affine transformation, ``y = x @ W + b``.

    Inputs have shape ``(batch, in_features)`` and outputs have shape
    ``(batch, out_features)``.
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        """Initialize weights and biases for the requested feature sizes."""
        rng = np.random.default_rng(seed)
        self.W = rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / in_features),
            size=(in_features, out_features),
        )
        self.b = np.zeros(shape=(out_features,))
        self.dW = None
        self.db = None
        self._x = None  # Cached input used to calculate gradients in backward().

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the affine transformation and cache its input."""
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Compute parameter gradients and return the gradient with respect to input."""
        self.dW = self._x.T @ grad_out
        self.db = np.sum(grad_out, axis=0)
        return grad_out @ self.W.T


class ReLU:
    """Apply the rectified linear unit activation elementwise."""

    def __init__(self):
        """Initialize the activation mask cache."""
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``max(x, 0)`` elementwise and cache the active positions."""
        self._mask = (x > 0)
        return np.where(self._mask, x, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Return the input gradient, zeroed where the forward input was negative."""
        return grad_out * self._mask
