"""Il modello: una sequenza di layer."""
import numpy as np


class MLP:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: passa x attraverso tutti i layer")

    def backward(self, grad: np.ndarray) -> None:
        raise NotImplementedError("TODO: propaga grad all'indietro (ordine inverso!)")

    def step(self, lr: float) -> None:
        raise NotImplementedError("TODO: SGD, W -= lr * dW per ogni layer con parametri")
