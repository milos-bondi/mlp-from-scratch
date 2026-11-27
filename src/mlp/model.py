"""Sequential multilayer perceptron model."""
from __future__ import annotations

import numpy as np

from mlp.layers import Linear, ReLU


class MLP:
    """Compose layers and propagate inputs and gradients through them."""

    def __init__(self, layers: list[Linear | ReLU]):
        """Create a model from layers in forward-pass order."""
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pass a batch of inputs through each layer and return the output."""
        input = x.copy()
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Propagate an output gradient backward through all layers."""
        gradient = grad.copy()
        inverted_layers = self.layers[::-1]
        for layer in inverted_layers:
            gradient = layer.backward(gradient)
        return gradient


    def step(self, lr: float) -> None:
        """Update each linear layer's parameters using its stored gradients."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
            
