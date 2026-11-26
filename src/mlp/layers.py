"""Layer della rete. Ogni layer ha forward() e backward()."""
import numpy as np


class Linear:
    """y = x @ W + b

    x: (batch, in_features)  ->  y: (batch, out_features)
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # TODO: inizializza self.W con He initialization: N(0, sqrt(2 / in_features))
        # TODO: inizializza self.b a zeri, shape (out_features,)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self._x = None  # salva l'input nel forward: ti serve nel backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: calcola x @ W + b e salva x")

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out = dL/dy. Calcola self.dW, self.db e ritorna dL/dx."""
        raise NotImplementedError("TODO")


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO")
