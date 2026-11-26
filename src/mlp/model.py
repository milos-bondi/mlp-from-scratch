"""Il modello: una sequenza di layer."""
from __future__ import annotations
import numpy as np
from src.mlp.layers import Linear, ReLU

class MLP:
    def __init__(self, layers: list[Linear|ReLU]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        input = x.copy()
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, grad: np.ndarray) -> None:
        gradient = grad.copy()
        inverted_layers = self.layers[::-1]
        for layer in inverted_layers:
            gradient = layer.backward(gradient)
        return gradient
        

    def step(self, lr: float) -> None:
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
            
