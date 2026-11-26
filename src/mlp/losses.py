"""Softmax + cross-entropy combinate (numericamente stabili)."""
import numpy as np


class SoftmaxCrossEntropy:
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """logits: (batch, classes), targets: (batch,) int. Returns avg Loss."""
        raise NotImplementedError("TODO")

    def backward(self) -> np.ndarray:
        """Returns dL/dlogits, shape (batch, classes). Result: (softmax - one_hot) / batch"""
        raise NotImplementedError("TODO")
