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
        self.W = rng.normal(loc=0.0, scale=np.sqrt(2/in_features), size=(in_features, out_features))
        self.b = np.zeros(shape=(out_features,))
        self.dW = None
        self.db = None
        self._x = None  # salva l'input nel forward: ti serve nel backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x@self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out = dL/dy. Calcola self.dW, self.db e ritorna dL/dx."""
        self.dW = self._x.T @ grad_out 
        self.db = np.sum(grad_out, axis=0)
        return grad_out@ self.W.T


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return np.where(self._mask, x, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask
