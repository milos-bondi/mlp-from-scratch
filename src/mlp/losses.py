"""Softmax + cross-entropy combinate (numericamente stabili)."""
import numpy as np


class SoftmaxCrossEntropy:

    def __init__(self):
        self.softmax = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """logits: (batch, classes), targets: (batch,) int. Returns avg Loss."""
        self.targets = targets
        batch_size = logits.shape[0]

        # 1. Softmax
        stable_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = stable_exp / np.sum(stable_exp, axis=1, keepdims=True)

        self.softmax = softmax
        correct_probs = softmax[np.arange(batch_size), targets]

        # 3. Cross-entropy
        loss = -np.mean(np.log(correct_probs + 1e-15))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Returns dL/dlogits, shape (batch, classes). Result: (softmax - one_hot) / batch"""
        batch_size = self.softmax.shape[0]
        grad = self.softmax.copy()
        grad[np.arange(batch_size), self.targets] -= 1.0
        return grad / batch_size


